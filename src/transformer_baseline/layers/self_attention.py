import torch
from torch import nn, Tensor
from typing import Optional


class SelfAttention(nn.Module):
    """Implementation of the Scaled Dot Product Attention layer according to the 'Attention is all you need' research paper"""
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,q_tensor: Tensor,k_tensor: Tensor,v_tensor : Tensor, mask:Optional[Tensor]=None) -> Tensor:

        # get dimension of any tensor
        batch_size, head, sequence_length, tensor_dim = k_tensor.size()

        # step 1 : Transpose the key tensor
        k_tensor = k_tensor.transpose(3,2)

        # Step 2 : Matrix Multiplication of query tensor and key tensor
        QT_t = torch.matmul(q_tensor, k_tensor)

        # step 3 : Scale the New Matrix
        self_attention_score = QT_t / torch.sqrt(torch.tensor([tensor_dim]))

        # Optional Step : Apply mask if needed
        if mask is not None:
            self_attention_score = self_attention_score.masked_fill(mask == 0, float("-inf"))

        # step 4 : Apply softmax to score
        softmax_score = self.softmax(self_attention_score)

        # step 5 : Matrix Multiplication of  softmax score and value tensor
        self_attention_output =  torch.matmul(softmax_score, v_tensor)

        return self_attention_output

            
