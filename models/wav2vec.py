import fairseq
import torch 
import torch.nn as nn
class Wav2Vec2Model(nn.Module):
    
    def __init__(self, wav2vec_path):
        super(Wav2Vec2Model, self).__init__()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
        self.model = model[0]   
        return 
    
    def forward(self, input_data):
        input_data = input_data.squeeze(1)
        emb = self.model(input_data, mask=False, features_only=True)
        return emb['x']
