import torch
from torch import nn, Tensor
from ..embeddings.word_token_embedding import Embedding
from ..utils.positional_encoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    """Implementation of the Embedding layer in the 'Attention is all you need' research paper"""
    def __init__(self, config: object):
        super().__init__()
        self.word_embedder = Embedding(config)
        self.positional_encoder = PositionalEncoding(config)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: Tensor) -> Tensor:
        x_embeddings = self.word_embedder(x)
        x_positions = self.positional_encoder(x)
        x = x_embeddings + x_positions 
        x = self.dropout(x)
        return x


