import torch
from torch import nn, Tensor


class LayerNormalization(nn.Module):
    """ LayerNormalization class inherting from the nn.Module for the purpose of normalizing a layer """
    def __init__(self, config: object):
        super().__init__()
        model_dim = config.model_dim
        self.layer_normalizer = nn.LayerNorm(model_dim)

    def forward(self, x_tensor: Tensor) -> Tensor:
        normalized_layer = self.layer_normalizer(x_tensor)
        return normalized_layer