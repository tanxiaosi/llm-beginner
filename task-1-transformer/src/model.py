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


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_heads, num_classes):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        self.block = TransformerBlock(d_model, n_heads)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, token_ids, padding_mask=None):
        B, T = token_ids.shape

        token_vecs = self.token_embedding(token_ids)
        positions = torch.arange(T, device=token_ids.device)
        pos_vecs = self.position_embedding(positions)
        x = token_vecs + pos_vecs

        if padding_mask is None:
            valid = torch.ones((B, T), dtype=torch.bool,
                               device=token_ids.device)
            attention_mask = None
        else:
            padding_mask = padding_mask.to(dtype=torch.bool,
                                           device=token_ids.device)
            valid = ~padding_mask
            attention_mask = padding_mask[:, None, None, :]

        x = self.block(x, attention_mask)

        valid = valid.unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

        logits = self.classifier(pooled)
        return logits
