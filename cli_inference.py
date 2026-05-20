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

import argparse

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models import load_model
from utils import prepare_one_sample
from dataset import SALMONNDataset
from runner import Runner
from utils import now
from local_dist_utils import  init_distributed_mode

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

model = load_model(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

# wav_path = input("Your Wav Path:\n")
# prompt = input("Your Prompt:\n")
wav_path = "/ds-slt/audio/fkallel/LibriSpeech/train-clean-100/103/1240/103-1240-0001.flac"
prompt = "Recognize the speech and give me the transcription."

samples = prepare_one_sample(wav_path, wav_processor)
prompt = [
    cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
]
print("Output:")
# for environment with cuda>=117
with torch.cuda.amp.autocast(dtype=torch.float16):
    print(model.generate(samples, cfg.config.generate, prompts=prompt)[0])
# # print(model.generate(samples, cfg.config.generate, prompts=prompt)[0])
job_id = now()

# load config
args.cfg_path = "configs/qwen_config.yaml"

cfg = Config(args)
run_config = cfg.config.run
model_config = cfg.config.model
model_config.ckpt = "outputs/202605190932/checkpoint_best.pth" 
data_config = cfg.config.datasets
init_distributed_mode(run_config)

# build model
model = load_model(model_config)

# build datasets
datasets = {
    "train": SALMONNDataset(data_config.train_ann_path, data_config.whisper_path),
    "valid": SALMONNDataset(data_config.valid_ann_path, data_config.whisper_path),
    "test": SALMONNDataset(data_config.test_ann_path, data_config.whisper_path),
}

# build runner
runner = Runner(cfg, model, datasets, job_id)

runner.valid_epoch("0","valid",decode=True,save_json=True)