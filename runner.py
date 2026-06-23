# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py

import os
import json
import time
import datetime
from pathlib import Path
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_rank, get_world_size
from logger import MetricLogger, SmoothedValue
from utils import get_dataloader, prepare_sample
from optims import get_optimizer, LinearWarmupCosineLRScheduler


class Runner:
    def __init__(self, cfg, model, datasets, job_id):
        self.config = cfg

        # log
        self.output_dir = Path(self.config.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)

        # settings
        self.device = torch.device(self.config.config.run.device)
        self.use_distributed = self.config.config.run.use_distributed
        self.start_epoch = 0
        self.max_epoch = self.config.config.run.optims.max_epoch
        self.evaluate_only = self.config.config.run.evaluate
        self.cuda_enabled = (self.device.type == "cuda")

        # test prompt
        self.prompt_template = self.config.config.model.get("prompt_template", "")
        test_prompt_path = self.config.config.model.get("test_prompt_path", "")
        if test_prompt_path:
            try:
                with open(test_prompt_path, "r") as f:
                    self.test_prompt_dict = json.load(f)
            except:
                print("Failed to load test prompt! Try to use utf-8 encoding.")
                with open(test_prompt_path, "r", encoding="utf-8") as f:
                    self.test_prompt_dict = json.load(f)
            for k in self.test_prompt_dict.keys():
                self.test_prompt_dict[k] = self.prompt_template.format(self.test_prompt_dict[k])

        else:
            self.test_prompt_dict = None

        # model
        self._model = model
        self._model.to(self.device)
        if self.use_distributed:
            self.model = DDP(
                self._model, device_ids=[self.config.config.run.gpu]
            )
        else:
            self.model = self._model

        # dataloaders
        self.train_loader = get_dataloader(datasets["train"], self.config.config.run, is_train=True, use_distributed=self.use_distributed)
        self.valid_loader = get_dataloader(datasets["valid"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)
        self.test_loader = get_dataloader(datasets["test"], self.config.config.run, is_train=False, use_distributed=self.use_distributed)

        # scaler
        self.use_amp = self.config.config.run.get("amp", False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # optimizer & scheduler
        self.iters_per_epoch = len(self.train_loader) if self.config.config.run.epoch_based else self.config.config.run.iters_per_epoch
        self.optimizer = get_optimizer(self.model, self.config.config.run.optims)
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=self.config.config.run.optims.min_lr,
            init_lr=self.config.config.run.optims.init_lr,
            warmup_steps=self.config.config.run.optims.warmup_steps,
            warmup_start_lr=self.config.config.run.optims.get("warmup_start_lr", -1),
        )

        self.log_config()

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def train_epoch(self, epoch):
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)

        for i in metric_logger.log_every(range(self.iters_per_epoch), self.config.config.run.log_freq, header=header, logger=self.log_writter, start_step=epoch*self.iters_per_epoch):
            if i >= self.iters_per_epoch:
                break

            samples = next(self.train_loader)
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            self.scheduler.step(cur_epoch=epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss = self.model(samples)["loss"]

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.config.config.run.accum_grad_iters == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @torch.no_grad()
    def valid_epoch(self, epoch, split, decode=False, save_json=False):
        model = self.unwrap_dist_model(self.model)
        model.eval()

        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Eval: data epoch: [{}]".format(epoch)

        results = []
        save_freq = self.config.config.run.get("save_result_freq", 100)
        batch_count = 0

        # Metrics for binary classification
        tp = 0  # True Positives (predicted bonafide, actual bonafide)
        tn = 0  # True Negatives (predicted spoof, actual spoof)
        fp = 0  # False Positives (predicted bonafide, actual spoof)
        fn = 0  # False Negatives (predicted spoof, actual bonafide)
        total_samples = 0
        total_loss = 0.0
        
        # For token-level accuracy (when decode=False)
        total_correct = 0
        total_tokens = 0

        for samples in metric_logger.log_every(dataloader, self.config.config.run.log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                forward_result = model(samples, verbose=True)
            
            loss = forward_result.get("loss", 0)
            total_loss += loss.item()
            
            # Get ground truth labels from the text (they contain "bonafide" or "spoof")
            ground_truth_texts = samples["text"]
            
            res = {
                "id": samples["id"],
                "ground_truth": ground_truth_texts,
                "loss": loss.item(),
            }

            if decode:
                if model.prompt_dict:
                    if self.test_prompt_dict is None:
                        prompts = None
                    else:
                        prompts = [self.test_prompt_dict[s] for s in samples["task"]]
                        if "Q" in samples:
                            prompts = [p.format(q) if "{}" in p else p for p, q in zip(prompts, samples["Q"])]
                else:
                    prompts = None

                text = model.generate(samples, self.config.config.run, prompts=prompts)
                res["text"] = text
                res["prompt"] = prompts
                res["task"] = samples["task"]
                
                # Extract binary labels from predictions and ground truth
                for i, (pred_text, gt_text) in enumerate(zip(text, ground_truth_texts)):
                    pred_label = self._extract_label(pred_text)
                    true_label = self._extract_label(gt_text)
                    
                    if pred_label == 1 and true_label == 1:
                        tp += 1
                    elif pred_label == 0 and true_label == 0:
                        tn += 1
                    elif pred_label == 1 and true_label == 0:
                        fp += 1
                    elif pred_label == 0 and true_label == 1:
                        fn += 1
                    total_samples += 1
            else:
                # When decode=False, use token-level accuracy from forward pass
                correct = forward_result.get("correct", 0)
                total = forward_result.get("total", 0)
                total_correct += correct
                total_tokens += total
                res["acc"] = (correct / total).item() if total > 0 else 0
                res["total"] = total

            results.append(res)
            batch_count += 1
            
            # Save results incrementally to avoid memory issues
            if save_json and batch_count % save_freq == 0:
                self.save_result_incremental(results, self.output_dir, "eval_{}_epoch_{}_incremental".format(split, epoch), clear_after_save=True)
                results = []

        if is_dist_avail_and_initialized():
            dist.barrier()

        if save_json:
            if results:
                self.save_result_incremental(results, self.output_dir, "eval_{}_epoch_{}_incremental".format(split, epoch), clear_after_save=False)
            self.merge_incremental_results(self.output_dir, "eval_{}_epoch_{}_incremental".format(split, epoch), "eval_{}_epoch_{}".format(split, epoch))

        # Compute metrics based on decode mode
        if decode:
            # Binary classification metrics
            if total_samples > 0:
                if is_dist_avail_and_initialized():
                    tp_tensor = torch.tensor(tp).float().cuda()
                    tn_tensor = torch.tensor(tn).float().cuda()
                    fp_tensor = torch.tensor(fp).float().cuda()
                    fn_tensor = torch.tensor(fn).float().cuda()
                    total_tensor = torch.tensor(total_samples).float().cuda()
                    loss_tensor = torch.tensor(total_loss).float().cuda()
                    
                    dist.all_reduce(tp_tensor)
                    dist.all_reduce(tn_tensor)
                    dist.all_reduce(fp_tensor)
                    dist.all_reduce(fn_tensor)
                    dist.all_reduce(total_tensor)
                    dist.all_reduce(loss_tensor)
                    
                    tp = int(tp_tensor.item())
                    tn = int(tn_tensor.item())
                    fp = int(fp_tensor.item())
                    fn = int(fn_tensor.item())
                    total_samples = int(total_tensor.item())
                    total_loss = loss_tensor.item()
                
                accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
                balanced_accuracy = 0.5 * (tp / (tp + fn) if (tp + fn) > 0 else 0) + 0.5 * (tn / (tn + fp) if (tn + fp) > 0 else 0)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                ret = {
                    "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0,
                    "agg_metrics": balanced_accuracy,  # Use balanced accuracy as the main metric
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "total_samples": total_samples
                }
            else:
                ret = {"loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0, "agg_metrics": 0}
        else:
            # Token-level accuracy (when decode=False)
            if is_dist_avail_and_initialized():
                correct_tensor = torch.tensor(total_correct).float().cuda()
                total_tensor = torch.tensor(total_tokens).float().cuda()
                loss_tensor = torch.tensor(total_loss).float().cuda()
                
                dist.all_reduce(correct_tensor)
                dist.all_reduce(total_tensor)
                dist.all_reduce(loss_tensor)
                
                total_correct = correct_tensor.item()
                total_tokens = total_tensor.item()
                total_loss = loss_tensor.item()
            
            token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
            
            ret = {
                "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0,
                "agg_metrics": token_accuracy,  # Use token accuracy as agg_metrics when not decoding
                "token_accuracy": token_accuracy,
                "correct": int(total_correct),
                "total": int(total_tokens)
            }

        return ret

    def _extract_label(self, text):
        """
        Extract binary label from raw text output.
        Returns 1 for 'bonafide', 0 for 'spoof'.
        """
        if not text:
            return 0
        
        text_lower = text.lower().strip()
        
        # Direct word matching
        if 'bonafide' in text_lower:
            return 1
        elif 'spoof' in text_lower:
            return 0
    
        return 0  # Default to spoof
    
    def save_result_incremental(self, result, result_dir, filename, clear_after_save=True):
        """
        Save results incrementally in JSONL format (one JSON object per line).
        This prevents data loss if the process crashes during long inference runs.
        """
        result_file = os.path.join(
            result_dir, "%s_rank%d.jsonl" % (filename, get_rank())
        )
        
        try:
            with open(result_file, "a", encoding="utf-8") as f:
                for item in result:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logging.info(f"Incremental save: {len(result)} results appended to {result_file}")
        except Exception as e:
            logging.warning(f"Error saving to {result_file}. Error: {e}")

    @main_process
    def merge_incremental_results(self, result_dir, incremental_filename, final_filename):
        """
        Merge incremental JSONL results from all ranks into final JSON file.
        """
        final_result_file = os.path.join(result_dir, "%s.json" % final_filename)
        merged_results = []
        
        logging.info("Starting to merge incremental results.")
        
        if is_dist_avail_and_initialized():
            num_ranks = get_world_size()
        else:
            num_ranks = 1
        
        for rank in range(num_ranks):
            result_file = os.path.join(
                result_dir, "%s_rank%d.jsonl" % (incremental_filename, rank)
            )
            if os.path.exists(result_file):
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                merged_results.append(json.loads(line))
                    logging.info(f"Merged {len(merged_results)} results from {result_file}")
                except Exception as e:
                    logging.warning(f"Error reading {result_file}. Error: {e}")
            else:
                logging.warning(f"Incremental result file not found: {result_file}")
        
        try:
            with open(final_result_file, "w", encoding="utf-8") as f:
                json.dump(merged_results, f, ensure_ascii=False, indent=2)
            logging.info(f"Final result file saved to {final_result_file} with {len(merged_results)} results")
        except Exception as e:
            logging.warning(f"Error saving {final_result_file}. Error: {e}")

    def save_result(self, result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        try:
            json.dump(result, open(result_file, "w"), ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Error saving {result_file}. Error: {e}")
            json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.info("rank %d starts merging results." % get_rank())
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                try:
                    res = json.load(open(result_file, "r"))
                except Exception as e:
                    logging.warning(f"Error reading {result_file}. Error: {e}")
                    res = json.load(open(result_file, "r", encoding="utf-8"))
                result += res

            try:
                json.dump(result, open(final_result_file, "w"), ensure_ascii=False)
            except Exception as e:
                logging.warning(f"Error saving {final_result_file}. Error: {e}")
                json.dump(result, open(final_result_file, "w", encoding="utf-8"), ensure_ascii=False)

            print("result file saved to %s" % final_result_file)

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break

            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")

            # validating phase
            logging.info("Validating Phase")
            valid_log = self.valid_epoch(cur_epoch, "valid", decode=True, save_json=True)
            if valid_log is not None:
                if is_main_process():
                    agg_metrics = valid_log["agg_metrics"]
                    if agg_metrics > best_agg_metric:
                        best_agg_metric = agg_metrics
                        best_epoch = cur_epoch

                        self.save_checkpoint(cur_epoch, is_best=True)

                    valid_log.update({"best_epoch": best_epoch})
                    self.log_stats(valid_log, split_name="valid")

            self.save_checkpoint(cur_epoch, is_best=False)

            if self.use_distributed:
                dist.barrier()

        # testing phase
        if self.evaluate_only:
            test_log = self.valid_epoch("best", "test", decode=True, save_json=True)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
