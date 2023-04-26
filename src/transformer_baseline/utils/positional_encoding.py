import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """Implementation of the Positional Encoder used to encode the position of the word-tokens 
        in the sequence into the learned embedding.
    """

    def __init__(self, config: object):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1).float()
        position = torch.arange(0, config.max_seq_length).unsqueeze(1)

        _2i = torch.arange(0, config.model_dim, step=2).float()
        div = position / (10000 ** (_2i / config.model_dim))

        pe = torch.zeros(config.max_seq_length, 1, config.model_dim)
        pe[:, 0, 0::2] = torch.sin(div)
        pe[:, 0, 1::2] = torch.cos(div)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:

        x = self.pe[:x.size(1),:]
        x = self.dropout(x)

        return x.transpose(0,1)

