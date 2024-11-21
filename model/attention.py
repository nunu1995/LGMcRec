import torch
import torch.nn as nn

class AttentionAlign(nn.Module):
     """
    A neural network module to compute attention-based mappings for embeddings.
    """
    def __init__(self, input_dim, output_dim):
         """
        Initialize AttentionAlign with input and output dimensions.

        Args:
            input_dim (int): Dimension of the input embeddings.
            output_dim (int): Dimension of the output embeddings.
        """
        super(AttentionAlign, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Perform forward pass to compute attention-weighted mappings.

        Args:
            x (Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            Tensor: Mapped tensor after applying attention and activation.
        """
        att_weights = torch.softmax(self.attention(x), dim=1)
        mapped = att_weights * x[:, :att_weights.size(1)]
        return self.activation(mapped)
