import torch

from models.transformer import VeloraTransformer

VOCAB_SIZE = 32000

model = VeloraTransformer(
    vocab_size=VOCAB_SIZE
)

# Przykładowy batch
input_ids = torch.randint(
    0,
    VOCAB_SIZE,
    (2, 16)
)

# Targety przesunięte o 1 token
target_ids = torch.randint(
    0,
    VOCAB_SIZE,
    (2, 16)
)

logits = model(input_ids)

loss_fn = torch.nn.CrossEntropyLoss()

loss = loss_fn(
    logits.view(-1, VOCAB_SIZE),
    target_ids.view(-1)
)

print("Input:", input_ids.shape)
print("Target:", target_ids.shape)
print("Logits:", logits.shape)
print("Loss:", loss.item())
