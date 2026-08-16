import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(Q, K, V, mask=None):

    # 1. compute raw attention scores
    K_t = K.transpose(-2, -1)
    scores = Q @ K_t

    # 2. scale by sqrt(D)
    D = Q.shape[-1]
    scores = scores / math.sqrt(D)

    # 3. apply mask if provided
    # repo convention: True = blocked
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # 4. softmax over keys
    weights = torch.softmax(scores, dim=-1)

    # 5. weighted sum of V
    output = weights @ V

    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, self.d_model)

        output = self.out_proj(attn_output)

        return output