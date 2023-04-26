import torch
from torch import nn, Tensor
from typing import Optional
from ..layers.self_attention import SelfAttention

class MultiHeadAttenion(nn.Module):
    """Implementation of the Multi Head Attention layer according to the 'Attention is all you need' research paper"""
    def __init__(self, config: object):
        super().__init__()
        self.num_head = config.num_attention_head
        self.self_attention = SelfAttention()
        self.q_linear_layer = nn.Linear(config.model_dim, config.model_dim)
        self.k_linear_layer = nn.Linear(config.model_dim, config.model_dim)
        self.v_linear_layer = nn.Linear(config.model_dim, config.model_dim)
        self.output_linear_layer = nn.Linear(config.model_dim, config.model_dim)
        self.reduced_dim  = config.multi_head_attention_reduced_dimension


    def linear_projection(self, learned_weights: Tensor) -> Tensor:
        """Function used to linear project or reshape the tensors.
        
        Parameters
        ----------
        learned_weights : Tensor
            Output tensor of the linear layer 
        Returns
        -------
        learned_weights : Tensor
           reshaped tensor
        """
        batch_size, sequence_length, tensor_dim = learned_weights.size()
        learned_weights = learned_weights.view(batch_size, sequence_length, self.num_head, self.reduced_dim).transpose(1, 2)

        return learned_weights


    def concat(self, tensors: Tensor) -> Tensor:
        """Function used to reshape tensor back to original size
        
        Parameters
        ----------
        tensors : Tensor
            Output from the self attention layer
        Returns
        -------
        tensors : Tensor
            Original size Tensor.
        """
        batch_size, head, sequence_length, d_tensor = tensors.size()
        model_dim = head * d_tensor

        tensors = tensors.transpose(1, 2).contiguous().view(batch_size, sequence_length, model_dim)
        return tensors



    def forward(self, q_tensor: Tensor, k_tensor: Tensor, v_tensor: Tensor, mask: Optional[Tensor]=None) -> Tensor:

        # Step 1: Get Learned weights from the incoming Tensors
        learned_weights_q = self.q_linear_layer(q_tensor)
        learned_weights_k = self.k_linear_layer(k_tensor)
        learned_weights_v = self.v_linear_layer(v_tensor)

        # Step 2: Project learned weights linearly
        q_projections = self.linear_projection(learned_weights_q)
        k_projections = self.linear_projection(learned_weights_k)
        v_projections = self.linear_projection(learned_weights_v)

        # Step 3 : Perform self attention in parallel
        attention_scores = self.self_attention(q_projections, k_projections, v_projections, mask=mask)


        # step 4 : Concat Scores
        attention_scores =  self.concat(attention_scores)

        # Step 5 : Pass Concated scores through a linear layer
        attention_scores = self.output_linear_layer(attention_scores)

        return attention_scores



