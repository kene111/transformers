import torch
from torch import  Tensor
class Mask:
    """Implementation of the mask used in the decoder block."""
    def __init__(self):
        pass

    def get_mask(self, input: Tensor) -> Tensor:
        mask = input.ne(1)
        #mask = torch.where(mask == True, mask, 0.0)
        return mask

    def no_look_ahead_mask(self, tensor: Tensor) -> Tensor:

        batch_size, seq_len= tensor.size()
        mask = torch.tril(torch.ones(seq_len, seq_len)).expand(
            batch_size, 1, seq_len, seq_len
        )

        return mask