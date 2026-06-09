from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import VeloraDataset
from models.transformer import VeloraTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset = VeloraDataset(
    corpus_path=PROJECT_ROOT / "data" / "final" / "velora_corpus.txt",
    vocab_path=str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
    merges_path=str(PROJECT_ROOT / "tokenizers" / "merges.txt"),
    seq_length=128
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = VeloraTransformer(
    vocab_size=32000
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

loss_fn = nn.CrossEntropyLoss()

for step, (input_ids, target_ids) in enumerate(dataloader):

    logits = model(input_ids)

    loss = loss_fn(
        logits.view(-1, 32000),
        target_ids.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(f"Step {step} | Loss: {loss.item():.4f}")

    if step >= 9:
        break
