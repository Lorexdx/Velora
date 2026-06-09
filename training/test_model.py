import torch

from models.transformer import VeloraTransformer

model = VeloraTransformer(vocab_size=32000)

x = torch.randint(0, 32000, (1, 10))

output = model(x)

print(output.shape)