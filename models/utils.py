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

import torch
from transformers import StoppingCriteria

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1, tokenizer=None):
        super().__init__()
        self.stops = stops
        self.encounters = encounters
        self.tokenizer = tokenizer
        self.stop_counter = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0, -1].item()
        if last_token in self.stops:
            return True

        
        # Additional check for text patterns if tokenizer is provided
        if self.tokenizer is not None and input_ids.shape[1] > 10:
            text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                        
            # Stop at USER/ASSISTANT patterns
            if "USER:" in text or "ASSISTANT:" in text:
                return True
                        
            # Stop if we see the same pattern repeating (like multiple fake answers)
            if text.count("<answer>fake</answer>") > 1:
                return True
            if text.count("<answer>real</answer>") > 1:
                return True
        
        return False