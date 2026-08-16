import torch
import torch.nn as nn

from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, mask=None):
        attn_out = self.attention(self.ln1(x), mask)
        x = x + attn_out

        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out

        return x