import logging
import json
import contextlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM,Qwen2Tokenizer, StoppingCriteriaList
from peft import LoraConfig, TaskType, get_peft_model
from .wav2vec import Wav2Vec2Model
from .utils import StoppingCriteriaSub


class ALLM(nn.Module):
    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        qwen_path="",
        wav2vec2_path="/ds-slt/audio_weights/xlsr2_300m.pt",
        freeze_wav2vec2=True,
        
        speech_qwen_proj_model="",
        freeze_speech_qwen_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,

        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.lora = lora
        self.multi_prompt = multi_prompt
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.low_resource = low_resource

        logging.info('Loading Qwen Tokenizer')
        self.qwen_tokenizer = Qwen2Tokenizer.from_pretrained(qwen_path, use_fast=False)
        self.qwen_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.qwen_tokenizer.padding_side = "right"
        logging.info('Loading Qwen Model')
        if self.low_resource:
            self.qwen_model = Qwen2ForCausalLM.from_pretrained(
                qwen_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={"": device_8bit},
            )
        else:
            self.qwen_model = Qwen2ForCausalLM.from_pretrained(
                qwen_path,
                torch_dtype=torch.float16,
            )

        self.qwen_model.resize_token_embeddings(len(self.qwen_tokenizer))
        for name, param in self.qwen_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading Qwen Done')

        if self.lora:
            self.peft_config = LoraConfig(
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.qwen_model = get_peft_model(self.qwen_model, self.peft_config)
            self.qwen_model.print_trainable_parameters()
            logging.info('LoRA Training')

        assert wav2vec2_path
        logging.info('Loading Wav2Vec2 Model')
        self.speech_encoder = Wav2Vec2Model(wav2vec2_path)
        self.speech_encoder_hidden_size = 1024
        self.ln_speech = nn.LayerNorm(self.speech_encoder_hidden_size)
        if freeze_wav2vec2:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()
            logging.info("freeze Wav2Vec2")
        
        self.speech_qwen_proj_model = nn.Linear(self.speech_encoder_hidden_size, self.qwen_model.config.hidden_size)
        if speech_qwen_proj_model:
            logging.info("Loading speech Qwen proj from {}".format(speech_qwen_proj_model))
            speech_qwen_proj_weight = torch.load(speech_qwen_proj_model, map_location="cpu")
            self.load_state_dict(speech_qwen_proj_weight['model'], strict=False)


        # prepare prompts
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")

    def _encode_auditory_feature(self, speech_embeds):
        with self.maybe_autocast():
            speech_embeds = self.ln_speech(speech_embeds)
            speech_embeds = self.speech_qwen_proj_model(speech_embeds)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        return speech_embeds, speech_atts

    def encode_speech(self, input_values):
        """
        Encode speech using preprocessed Wav2Vec2 inputs
        
        Args:
            input_values: Preprocessed Wav2Vec2 input (already on correct device)
            attention_mask: Preprocessed Wav2Vec2 attention mask
        """

        speech_embeds = self.speech_encoder(input_values.float())
        return self._encode_auditory_feature(speech_embeds)
    
    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    if "<SpeechHere>" in p:
                        b, a = p.split("<SpeechHere>")
                    else:
                        b = ""
                        a = p
                    p_before.append(b)
                    p_after.append(a)
                
                # Tokenize before prompts - don't pad
                p_before_tokens = self.qwen_tokenizer(
                    p_before, 
                    return_tensors="pt", 
                    add_special_tokens=False,
                    padding=False
                ).to(embeds.device)
                
                # Tokenize after prompts WITH padding
                p_after_tokens = self.qwen_tokenizer(
                    p_after, 
                    return_tensors="pt", 
                    add_special_tokens=False,
                    padding=True,
                    return_attention_mask=True
                ).to(embeds.device)
                
                # Get embeddings
                if self.lora:
                    embed_func = self.qwen_model.model.model.embed_tokens
                else:
                    embed_func = self.qwen_model.model.embed_tokens
                    
                p_before_embeds = embed_func(p_before_tokens.input_ids)
                p_after_embeds = embed_func(p_after_tokens.input_ids)
                
                batch_size = embeds.shape[0]
                
                # Expand if needed to match batch size
                if p_before_embeds.shape[0] == 1 and batch_size > 1:
                    p_before_embeds = p_before_embeds.expand(batch_size, -1, -1)
                    p_before_tokens.attention_mask = p_before_tokens.attention_mask.expand(batch_size, -1)
                
                if p_after_embeds.shape[0] == 1 and batch_size > 1:
                    p_after_embeds = p_after_embeds.expand(batch_size, -1, -1)
                    p_after_tokens.attention_mask = p_after_tokens.attention_mask.expand(batch_size, -1)
                
                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
                
                return wrapped_embeds, wrapped_atts
            else:
                batch_size = embeds.shape[0]
                if "<SpeechHere>" in prompt:
                    p_before, p_after = prompt.split("<SpeechHere>")
                else:
                    p_before = ""
                    p_after = prompt

                p_before_tokens = self.qwen_tokenizer(
                    p_before, 
                    return_tensors="pt", 
                    add_special_tokens=False
                ).to(embeds.device)
                
                p_after_tokens = self.qwen_tokenizer(
                    p_after, 
                    return_tensors="pt", 
                    add_special_tokens=False
                ).to(embeds.device)
                
                if self.lora:
                    embed_func = self.qwen_model.model.model.embed_tokens
                else:
                    embed_func = self.qwen_model.model.embed_tokens
                    
                p_before_embeds = embed_func(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = embed_func(p_after_tokens.input_ids).expand(batch_size, -1, -1)
                
                p_before_mask = p_before_tokens.attention_mask.expand(batch_size, -1)
                p_after_mask = p_after_tokens.attention_mask.expand(batch_size, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_mask, atts, p_after_mask], dim=1)
                
                return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts
    
    def compute_policy_logprobs(self, samples, texts, prompts=None):
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        if self.prompt_dict and prompts is None:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])
        elif prompts is not None:
            prompt = prompts
        else:
            prompt = None

        input_values = samples["input_values"]

        speech_embeds, speech_atts = self.encode_speech(input_values)

        if self.prompt_dict and prompt is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=True if prompts is not None else self.multi_prompt)

        text = [t + self.qwen_tokenizer.eos_token for t in texts]
        to_regress_tokens = self.qwen_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(speech_embeds.device)
        to_regress_embeds = self.qwen_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.qwen_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.qwen_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] ],
                dtype=torch.long
            ).to(speech_embeds.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = torch.cat([speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([speech_atts, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.qwen_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            logits = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous()
            labels = targets[:, empty_targets.size(1):].contiguous()
            log_probs = torch.log_softmax(logits, dim=-1)

            mask = labels != -100

            safe_labels = labels.clone()
            safe_labels[~mask] = 0      # any valid token id

            token_log_probs = (
                log_probs.gather(-1, safe_labels.unsqueeze(-1))
                .squeeze(-1)
            )

            token_log_probs = token_log_probs.masked_fill(~mask, 0.0)

            return token_log_probs.sum(dim=1)
        
    def forward(self, samples, verbose=False):
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

        # use speech/audio encoder to encode speech/audio
        input_values = samples["input_values"]

        speech_embeds, speech_atts = self.encode_speech(input_values)

        # wrap speech_embeds with prompts
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM
        text = [t + self.qwen_tokenizer.eos_token for t in samples["text"]]
        # text = samples['text']
        to_regress_tokens = self.qwen_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(speech_embeds.device)
        to_regress_embeds = self.qwen_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.qwen_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.qwen_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] ],
                dtype=torch.long
            ).to(speech_embeds.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = speech_embeds.shape[0]

        inputs_embeds = torch.cat([ speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([ speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.qwen_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.qwen_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}

    def generate(self, samples, generate_cfg, prompts=None):
        input_values = samples.get("input_values", None)
        speech_embeds, speech_atts = self.encode_speech(input_values)

        if prompts is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)


        embeds =  speech_embeds
        attns = speech_atts

        # im_end_token_id = self.qwen_tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        # stop_words_ids = [self.qwen_tokenizer.eos_token_id, im_end_token_id]

        # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.qwen_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 4),
            # stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.2),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
            pad_token_id=self.qwen_tokenizer.pad_token_id,
            eos_token_id=self.qwen_tokenizer.eos_token_id,
        )
        text = self.qwen_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # text = [ t.replace('<|endoftext|>', '').replace('<|im_end|>','').strip() for t in text]

        return text

    @classmethod
    def from_config(cls, config):
        qwen_path = config.get("qwen_path")
        wav2vec2_path = config.get("wav2vec2_path")
        freeze_wav2vec2 = config.get("freeze_wav2vec2", True)
        speech_qwen_proj_model = config.get("speech_qwen_proj_model", "")
        freeze_speech_qwen_proj = config.get("freeze_speech_qwen_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        model = cls(
            qwen_path=qwen_path,
            wav2vec2_path=wav2vec2_path,
            freeze_wav2vec2=freeze_wav2vec2,
            speech_qwen_proj_model=speech_qwen_proj_model,
            freeze_speech_qwen_proj=freeze_speech_qwen_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load SALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model
