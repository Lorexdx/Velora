import torch
import torch.nn as nn

from models.transformer import VeloraTransformer

VOCAB_SIZE = 30000

model = VeloraTransformer(
    vocab_size=VOCAB_SIZE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

loss_fn = nn.CrossEntropyLoss()

input_ids = torch.randint(
    0,
    VOCAB_SIZE,
    (2, 16)
)

target_ids = torch.randint(
    0,
    VOCAB_SIZE,
    (2, 16)
)

optimizer.zero_grad()

logits = model(input_ids)

loss = loss_fn(
    logits.view(-1, VOCAB_SIZE),
    target_ids.view(-1)
)

print("Loss przed:", loss.item())

loss.backward()

optimizer.step()

logits_after = model(input_ids)

loss_after = loss_fn(
    logits_after.view(-1, VOCAB_SIZE),
    target_ids.view(-1)
)

print("Loss po:", loss_after.item())
