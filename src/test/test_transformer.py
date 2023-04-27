import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from transformer_baseline.configurations.config import ConfigFile
from transformer_baseline.model.transformer import Transformer




config = ConfigFile()

config.vocab_size = 11
config.num_decoder_layers = 6
config.num_encoder_layers = 6
config.max_seq_length = 12

#print(config)

# let 0 be sos token and 1 be eos token
src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1], 
                    [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])
target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1], 
                       [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])


model = Transformer(config)
model_output = model(src,target)
print(model_output)
