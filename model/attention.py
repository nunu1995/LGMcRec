import torch
import torch.nn as nn

class AttentionAlign(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionAlign, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        att_weights = torch.softmax(self.attention(x), dim=1)
        mapped = att_weights * x[:, :att_weights.size(1)]
        return self.activation(mapped)
