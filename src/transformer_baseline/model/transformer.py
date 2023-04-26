import torch
from torch import nn, Tensor
from ..model.decoder import Decoder
from ..model.encoder import Encoder
from ..utils.masks import Mask


class Transformer(nn.Module):
    """ Transformer class implenting the end to end architecture as shown in the research paper """
    def __init__(self, config: object):
        super().__init__()
        self.decoder = Decoder(config)
        self.encoder = Encoder(config)
        self.linear = nn.Linear(config.model_dim, config.vocab_size)
        self.softmax= nn.Softmax()
        self.create_mask = Mask()

    def forward(self, _input: Tensor, _output: Tensor) -> Tensor:

        enc_mask = None#self.create_mask.get_mask(_input)
        enc_out = self.encoder(_input, enc_mask)

        dec_mask = self.create_mask.get_mask(_output)
        dec_no_lookahead_mask = self.create_mask.no_look_ahead_mask(_output)
        dec_output = self.decoder(_output, enc_out=enc_out, target_mask=dec_no_lookahead_mask, src_mask=enc_mask)

        linear_output = self.linear(dec_output)
        softmax_output = self.softmax(linear_output)

        return softmax_output