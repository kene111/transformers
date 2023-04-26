import torch
from torch import nn, Tensor
from ..blocks.encoder_block import EncoderBlock
from ..embeddings.embeddings import TransformerEmbedding

class Encoder(nn.Module):
    """ Encoder class implenting the full encoding process as shown in the research paper """
    def __init__(self, config: object):
        super().__init__()
        self.embedding = TransformerEmbedding(config)
        self.encoder_layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)])


    def forward(self,x: Tensor, mask: Tensor) -> Tensor:
        x = self.embedding(x)
        for enc in self.encoder_layers:
            x = enc(x, mask)

        return x

