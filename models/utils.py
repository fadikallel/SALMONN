import torch
from transformers import StoppingCriteria

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[]):
        super().__init__()
        # Convert stop sequences to tensors for comparison
        self.stops = [torch.tensor(stop, dtype=torch.long) for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Create a boolean tensor of shape (batch_size,), initialized to False
        # True means "stop this sequence", False means "keep generating"
        stopped_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        
        for i in range(input_ids.shape[0]):
            for stop in self.stops:
                stop_len = stop.shape[0]
                
                # Only check if the sequence is long enough to contain the stop word
                if input_ids.shape[1] >= stop_len:
                    # Check if the last N tokens match the stop sequence exactly
                    if torch.equal(input_ids[i, -stop_len:], stop.to(input_ids.device)):
                        stopped_mask[i] = True
                        break # Found a stop word, no need to check other stop words for this sample
                        
        return stopped_mask # Return the tensor, NOT a single boolean!