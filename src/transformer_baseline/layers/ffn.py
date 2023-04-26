import torch
from torch import nn, Tensor


class PointWiseFFN(nn.Module):
    """Implementation of the Point-Wise Feed Forward Network according to the 'Attention is all you need' research paper"""
    def __init__(self, config: object):
        super().__init__()
        self.linear_layer_1 = nn.Linear(config.model_dim, config.hidden_layer)
        self.linear_layer_2 = nn.Linear(config.hidden_layer, config.model_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()


    def forward(self, x_tensor: Tensor) -> Tensor:

        x_tensor = self.linear_layer_1(x_tensor)
        x_tensor = self.relu(x_tensor)
        x_tensor = self.dropout(x_tensor)
        x_tensor = self.linear_layer_2(x_tensor)

        return x_tensor