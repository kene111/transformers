import torch
from typing import Optional
from torch import nn, Tensor
from ..blocks.decoder_block import DecoderBlock
from ..embeddings.embeddings import TransformerEmbedding

class Decoder(nn.Module):
    """ Decoder class implenting the full decoding process as shown in the research paper """         
    def __init__(self, config: object):
        super().__init__()
        self.embedding = TransformerEmbedding(config)
        self.decoder_layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])

    def forward(self, dec_input: Tensor, enc_out: Optional[Tensor] = None, target_mask: Optional[Tensor] = None, src_mask: Optional[Tensor] = None) -> Tensor:

        dec_input = self.embedding(dec_input)

        for dec in self.decoder_layers:
            dec_input = dec(dec_input, enc_out=enc_out, target_mask=target_mask, src_mask=src_mask)

        return dec_input

