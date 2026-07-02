# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from email.mime import audio
import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2FeatureExtractor


class SALMONNDataset(Dataset):
    def __init__(self, ann_path, wav2vec2_path):
        super().__init__()

        self.annotation = json.load(open(ann_path, "r"))
        
        # Use Wav2Vec2FeatureExtractor instead of WhisperFeatureExtractor
        self.wav_processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec2_path)

    def _normalize_audio(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        audio = np.asarray(audio)

        if audio.ndim == 0:
            audio = audio.reshape(1)
        elif audio.ndim > 1:
            # Collapse common 2D layouts to a single waveform.
            if audio.ndim == 2 and audio.shape[0] != 1 and audio.shape[1] != 1:
                audio = audio[:, 0]
            else:
                audio = audio.reshape(-1)

        audio = audio.astype(np.float32, copy=False)
        if audio.size == 0:
            audio = np.zeros(16000, dtype=np.float32)

        return audio.reshape(-1)

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        raw_audios = [self._normalize_audio(s["raw_wav"]) for s in samples]

        try:
            processed = self.wav_processor(
                raw_audios,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
        except ValueError:
            processed_items = []
            for audio in raw_audios:
                item = self.wav_processor(
                    audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                processed_items.append(item)

            max_len = max(item["input_values"].shape[-1] for item in processed_items)
            input_values = []
            attention_mask = []
            for item in processed_items:
                values = item["input_values"][0]
                mask = item["attention_mask"][0]
                pad = max_len - values.shape[-1]
                if pad > 0:
                    values = torch.nn.functional.pad(values, (0, pad), value=0.0)
                    mask = torch.nn.functional.pad(mask, (0, pad), value=0)
                input_values.append(values)
                attention_mask.append(mask)

            processed = {
                "input_values": torch.stack(input_values),
                "attention_mask": torch.stack(attention_mask),
            }

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        ids = [s["id"] for s in samples]

        return {
            "input_values": processed["input_values"],
            "attention_mask": processed["attention_mask"],
            "text": text,
            "task": task,
            "Q": Q,
            "id": ids,
        }
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        # Load audio
        audio, sr = sf.read(ann["path"])
        audio = self._normalize_audio(audio)
        
        # Expand audio if needed
        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, _ = sf.read(p)
                expand_audio = self._normalize_audio(expand_audio)
                sil = np.zeros(1600, dtype=np.float32)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)
        
        # Pad audio to at least 1s
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=np.float32)
            audio = np.concatenate((audio, sil), axis=0)
        
        # Truncate audio to at most 30s
        audio = audio[:sr * 30]

        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        return {
            "raw_wav": audio,  # Keep raw audio for potential use
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }