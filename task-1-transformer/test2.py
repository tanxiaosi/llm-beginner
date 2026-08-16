import torch
import torch.nn as nn

from src.model import TinyDecoderModel


# 1. Create the model and loss function
model = TinyDecoderModel(
    vocab_size=100,
    max_len=20,
    d_model=8,
    n_heads=2
)

loss_fn = nn.CrossEntropyLoss()


# 2. Create and shift the sequences
train_sequences = torch.tensor([
    [3, 7, 9, 2, 5],
    [4, 1, 8, 6, 3],
    [2, 5, 4, 9, 1],
    [7, 3, 6, 8, 2],
    [5, 9, 1, 4, 7],
])

val_sequences = torch.tensor([
    [6, 2, 7, 4, 8],
    [8, 5, 3, 1, 6],
])

train_input = train_sequences[:, :-1]
train_target = train_sequences[:, 1:]

val_input = val_sequences[:, :-1]
val_target = val_sequences[:, 1:]


# 3. Add the reusable function here
def compute_loss(input_ids, targets):
    T = input_ids.shape[1]

    mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool),
        diagonal=1
    )

    logits = model(input_ids, mask)

    loss = loss_fn(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1)
    )

    return loss


# 4. Evaluate the initial losses here
with torch.no_grad():
    initial_train_loss = compute_loss(train_input, train_target)
    initial_val_loss = compute_loss(val_input, val_target)

print("initial train loss:", initial_train_loss.item())
print("initial val loss:", initial_val_loss.item())

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

for step in range(61):
    model.train()

    optimizer.zero_grad()

    train_loss = compute_loss(
        train_input,
        train_target
    )

    train_loss.backward()
    optimizer.step()

    if step % 20 == 0:
        model.eval()

        with torch.no_grad():
            measured_train_loss = compute_loss(
                train_input,
                train_target
            )

            measured_val_loss = compute_loss(
                val_input,
                val_target
            )

        print(
            step,
            "train:", measured_train_loss.item(),
            "val:", measured_val_loss.item()
        )