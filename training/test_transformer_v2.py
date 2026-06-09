import torch

from models.transformer import VeloraTransformer

VOCAB_SIZE = 30000

model = VeloraTransformer(
    vocab_size=VOCAB_SIZE
)

x = torch.randint(
    0,
    VOCAB_SIZE,
    (1, 10)
)

output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)

params = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Parametry: {params:,}")
