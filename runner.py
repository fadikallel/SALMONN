# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py
import librosa

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

from local_dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_rank, get_world_size
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
                self._model, device_ids=[self.config.config.run.gpu],find_unused_parameters=True
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
                loss = self.model(samples, verbose=True)["loss"]

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
        save_freq = self.config.config.run.get("save_result_freq", 100)  # Save every 100 samples by default
        batch_count = 0
        
        # Accumulate metrics incrementally
        accumulated_loss = 0.0
        accumulated_correct = 0.0
        accumulated_n_sample = 0
        accumulated_n_token = 0
        
        for samples in metric_logger.log_every(dataloader, self.config.config.run.log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                forward_result = model(samples, verbose=True)
            loss = forward_result.get("loss", 0)
            correct = forward_result.get("correct", 0)
            total = forward_result.get("total", 1)
            res = {
                "id": samples["id"],
                "ground_truth": samples["text"],
                "loss": loss.item(),
                "acc": (correct / total).item(),
                "total": total,
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

            results.append(res)
            batch_count += 1
            
            # Accumulate metrics
            item_loss = res["loss"]
            item_n_sample = len(res["id"])
            item_correct = res["acc"] * res["total"]
            item_n_token = res["total"]
            
            accumulated_loss += item_loss * item_n_sample
            accumulated_n_sample += item_n_sample
            accumulated_correct += item_correct
            accumulated_n_token += item_n_token
            
            # Save results incrementally to avoid losing all data if process crashes
            if save_json and batch_count % save_freq == 0:
                self.save_result_incremental(results, self.output_dir, "eval_{}_epoch_{}_incremental".format(split, epoch), clear_after_save=True)
                results = []

        if is_dist_avail_and_initialized():
            dist.barrier()

        if save_json:
            # Save remaining results
            if results:
                self.save_result_incremental(results, self.output_dir, "eval_{}_epoch_{}_incremental".format(split, epoch), clear_after_save=False)
            # Merge all incremental results into final JSON
            self.merge_incremental_results(self.output_dir, "eval_{}_epoch_{}_incremental".format(split, epoch), "eval_{}_epoch_{}".format(split, epoch))

        # Prepare tensors for distributed reduction
        res = {
            "loss": torch.tensor(accumulated_loss).float().cuda(),
            "n_sample": torch.tensor(accumulated_n_sample).float().cuda(),
            "correct": torch.tensor(accumulated_correct).float().cuda(),
            "n_token": torch.tensor(accumulated_n_token).float().cuda(),
        }

        if is_dist_avail_and_initialized():
            dist.all_reduce(res["loss"])
            dist.all_reduce(res["n_sample"])
            dist.all_reduce(res["correct"])
            dist.all_reduce(res["n_token"])

        ret = {"loss": 0, "agg_metrics": 0}
        ret["loss"] = (res["loss"] / res["n_sample"]).item() if res["n_sample"].item() > 0 else 0
        ret["agg_metrics"] = (res["correct"] / res["n_token"]).item() if res["n_token"].item() > 0 else 0

        return ret

    def save_result_incremental(self, result, result_dir, filename, clear_after_save=True):
        """
        Save results incrementally in JSONL format (one JSON object per line).
        This prevents data loss if the process crashes during long inference runs.
        """
        result_file = os.path.join(
            result_dir, "%s_rank%d.jsonl" % (filename, get_rank())
        )
        
        # Append results to JSONL file
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
        
        # Read from all rank-specific JSONL files
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
        
        # Save final merged result
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
            valid_log = self.valid_epoch(cur_epoch, "valid", decode=False, save_json=False)
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
