import pandas as pd
import torch

df = pd.read_parquet("task-1-transformer/data/train.parquet")

chars = sorted(set("".join(df["text"].astype(str))))

vocab = {"<PAD>": 0, "<UNK>": 1}
vocab.update({ch: i + 2 for i, ch in enumerate(chars)})

def tokenize(text):
    return [vocab.get(ch, vocab["<UNK>"]) for ch in text]

texts = df.iloc[:3]["text"].astype(str).tolist()
labels = df.iloc[:3]["label"].tolist()

tokenized = [tokenize(text[:32]) for text in texts]

max_len = max(len(ids) for ids in tokenized)

pad_id = vocab["<PAD>"]

padded = [
    ids + [pad_id] * (max_len - len(ids))
    for ids in tokenized
]

padding_mask = [
    [False] * len(ids) + [True] * (max_len - len(ids))
    for ids in tokenized
]

input_ids = torch.tensor(padded, dtype=torch.long)
padding_mask = torch.tensor(padding_mask, dtype=torch.bool)
labels_tensor = torch.tensor(labels, dtype=torch.long)

print("input_ids shape:", input_ids.shape)
print("padding_mask shape:", padding_mask.shape)
print("labels shape:", labels_tensor.shape)

print("input_ids dtype:", input_ids.dtype)
print("padding_mask dtype:", padding_mask.dtype)
print("labels dtype:", labels_tensor.dtype)

print("labels:", labels_tensor)

assert input_ids.shape == (3, 32)
assert padding_mask.shape == (3, 32)
assert labels_tensor.shape == (3,)

assert input_ids.dtype == torch.long
assert padding_mask.dtype == torch.bool
assert labels_tensor.dtype == torch.long

assert padded[2].count(pad_id) == 12
assert padding_mask[2].sum().item() == 12

val_df = pd.read_parquet("task-1-transformer/data/validation.parquet")
val_ids = [tokenize(text) for text in val_df["text"].astype(str)]
total_unk = sum(ids.count(vocab["<UNK>"]) for ids in val_ids)

assert total_unk > 0
print("validation UNK tokens:", total_unk)

print("tokenization checks passed")