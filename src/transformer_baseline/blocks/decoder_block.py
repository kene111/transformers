import torch
from typing import Optional
from torch import nn, Tensor
from ..layers.ffn import PointWiseFFN
from ..layers.mha import MultiHeadAttenion
from ..layers.layer_normalizer import LayerNormalization



class DecoderBlock(nn.Module):
    """Implementation of the Decoder block according to the 'Attention is all you need' research paper"""
    def __init__(self, config: object):
        super().__init__()
        self.norm_layer_1 = LayerNormalization(config)
        self.mha_1 = MultiHeadAttenion(config)
        self.norm_layer_2 = LayerNormalization(config)
        self.enc_dec = MultiHeadAttenion(config)
        self.ffn = PointWiseFFN(config)
        self.norm_layer_3 = LayerNormalization(config)


    def forward(self, dec_input: Tensor, enc_out:Optional[Tensor]=None, target_mask:Optional[Tensor]=None, src_mask:Optional[Tensor]=None) -> Tensor:

        # first residual in the decoder block
        residual_1 = dec_input

        q = dec_input
        k = dec_input
        v  = dec_input

        # perfom masked multi-head attention on the inputs
        x = self.mha_1(q,k,v, target_mask)

        # add & norm layer
        add_ = x + residual_1
        norm_output = self.norm_layer_1(add_)

        # second residual in the decoder block
        residual_2 = norm_output

        if enc_out is not None:
            # q is the input from the decoder
            q = norm_output
            # k and v is the input from the encoder
            k = enc_out
            v = enc_out

            # perform mulit-head attention
            x = self.enc_dec(q,k,v, src_mask)

        # add & norm layer
        add_ = residual_2 + x
        norm_output_2 = self.norm_layer_2(add_)

        # third residual in the decoder block
        residual_3 = norm_output_2

        # point-wise FFN
        learned_weights = self.ffn(norm_output_2)

        # add & norm layer
        add_ = learned_weights + residual_3
        norm_output_3 = self.norm_layer_3(add_)

        return norm_output_3




