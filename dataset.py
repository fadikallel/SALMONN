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

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        # Stack input_values (already preprocessed by Wav2Vec2)
        raw_audios = [s["raw_wav"] for s in samples]
        
        # Process ALL audios in one batch
        # This is more efficient than processing one-by-one
        processed = self.wav_processor(
            raw_audios,  # List of numpy arrays
            sampling_rate=16000,  # Should be same for all
            padding=True,  # Pad to max length in batch
            return_tensors="pt",  # Return PyTorch tensors
            return_attention_mask=True
        )
        
        # Extract text and metadata
        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        ids = [s["id"] for s in samples]
        
        return {
            "input_values": processed["input_values"],  # Already batched and padded
            "attention_mask": processed["attention_mask"],  # Already batched and padded
            "text": text,
            "task": task,
            "Q": Q,
            "id": ids,
        }
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        # Load audio
        audio, sr = sf.read(ann["path"])
        if len(audio.shape) == 2:  # stereo to mono
            audio = audio[:, 0]
        
        # Expand audio if needed
        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, _ = sf.read(p)
                if len(expand_audio.shape) == 2:
                    expand_audio = expand_audio[:, 0]
                sil = np.zeros(1600, dtype=float)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)
        
        # Pad audio to at least 1s
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=float)
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