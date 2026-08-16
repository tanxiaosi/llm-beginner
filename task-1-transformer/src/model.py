import torch
import torch.nn as nn

from src.block import TransformerBlock


class TinyDecoderModel(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        self.block = TransformerBlock(d_model, n_heads)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, mask=None):
        B, T = token_ids.shape

        token_vecs = self.token_embedding(token_ids)

        positions = torch.arange(T, device=token_ids.device)
        pos_vecs = self.position_embedding(positions)

        x = token_vecs + pos_vecs

        x = self.block(x, mask)

        logits = self.lm_head(x)
        return logits