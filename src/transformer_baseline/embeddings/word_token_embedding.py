import torch
from torch import nn


class Embedding(nn.Embedding):
    """Embedding class inheriting from the pytorch nn.Embedding class"""
    def __init__(self, config):
        super().__init__(config.vocab_size, config.model_dim, padding_idx=1)