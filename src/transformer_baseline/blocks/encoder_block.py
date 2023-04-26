import torch
from torch import nn, Tensor
from typing import Optional
from ..layers.ffn import PointWiseFFN
from ..layers.mha import MultiHeadAttenion
from ..layers.layer_normalizer import LayerNormalization



class EncoderBlock(nn.Module):
    """Implementation of the Encoder block according to the 'Attention is all you need' research paper"""
    def __init__(self, config: object):
        super().__init__()
        self.norm_layer_1 = LayerNormalization(config)
        self.mha = MultiHeadAttenion(config)
        self.norm_layer_2 = LayerNormalization(config)
        self.ffn = PointWiseFFN(config)


    def forward(self, x: Tensor, mask:Optional[Tensor]=None) -> Tensor:

        # first residual in the encoder block
        residual_1 = x

        q = x
        k = x
        v  = x

        # perfom multi-head attention on the inputs
        learned_linear_weights = self.mha(q,k,v, mask)

        # add & norm layer
        add_ = learned_linear_weights + residual_1
        normalized_output = self.norm_layer_1(add_)

        # second residual in the encoder block
        residual_2 = normalized_output

        # point-wise FFN
        ffn_output = self.ffn(normalized_output)

        # add & norm layer
        add_ = ffn_output + residual_2
        normalized_output_2 = self.norm_layer_2(add_)

        return normalized_output_2


