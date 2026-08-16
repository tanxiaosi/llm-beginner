import torch
import torch.nn as nn

from src.model import TinyDecoderModel

model = TinyDecoderModel(
    vocab_size=100,
    max_len=20,
    d_model=8,
    n_heads=2
)

input_ids = torch.tensor([
    [3, 7, 9, 2],
    [4, 1, 8, 6],
    [2, 5, 4, 9]
])

target = torch.tensor([
    [7, 9, 2, 5],
    [1, 8, 6, 3],
    [5, 4, 9, 1]
])

val_input = torch.tensor([
    [6, 2, 7, 4]
])

val_target = torch.tensor([
    [2, 7, 4, 8]
])

T = input_ids.shape[1]

mask = torch.triu(
    torch.ones(T, T, dtype=torch.bool),
    diagonal=1
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)
loss_fn = nn.CrossEntropyLoss()

for step in range(30):
    optimizer.zero_grad()

    logits = model(input_ids, mask)

    loss = loss_fn(
        logits.reshape(-1, 100),
        target.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    print(step, loss.item())

with torch.no_grad():
    val_logits = model(val_input, mask)

    val_loss = loss_fn(
        val_logits.reshape(-1, 100),
        val_target.reshape(-1)
    )

print("train loss:", loss.item())
print("val loss:", val_loss.item())