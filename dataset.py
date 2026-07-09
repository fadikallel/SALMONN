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

# from email.mime import audio
import json

import torch
from torch.utils.data import Dataset
# from torch.nn.utils.rnn import pad_sequence
# import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor
import librosa

class SALMONNDataset(Dataset):
    def __init__(self, ann_path, whisper_path):
        super().__init__()

        self.annotation = json.load(open(ann_path, "r"))
        
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)


    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)
        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        ids = [s["id"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ids,
        }
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        # Load audio
        audio, sr = librosa.load(ann["path"], sr=16000)
        if len(audio.shape) == 2: # stereo to mono
            audio = audio[:, 0]        
        # Expand audio if needed
        if "expand_wav" in ann:
            for p in ann["expand_wav"]:
                expand_audio, _ = librosa.load(p, sr=16000)
                if len(expand_audio.shape) == 2: # stereo to mono
                    expand_audio = expand_audio[:, 0]
                sil = np.zeros(1600, dtype=np.float32)
                audio = np.concatenate((audio, sil, expand_audio), axis=0)
        
        # Pad audio to at least 1s
        if len(audio) < sr:
            sil = np.zeros(sr - len(audio), dtype=np.float32)
            audio = np.concatenate((audio, sil), axis=0)
        
        # Truncate audio to at most 30s
        audio = audio[:sr * 30]
        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        return {
            "text": text,
            "task": task,
            "Q": Q,
            "spectrogram": spectrogram,
            "id": ann["path"],
        }