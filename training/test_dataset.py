from pathlib import Path

from training.dataset import VeloraDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset = VeloraDataset(
    corpus_path=PROJECT_ROOT / "data" / "final" / "velora_corpus.txt",
    vocab_path=str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
    merges_path=str(PROJECT_ROOT / "tokenizers" / "merges.txt"),
    seq_length=128
)

print()
print("Długość datasetu:", len(dataset))

input_ids, target_ids = dataset[0]

print()
print("Input shape :", input_ids.shape)
print("Target shape:", target_ids.shape)

print()
print("Pierwsze 10 input tokenów:")
print(input_ids[:10])

print()
print("Pierwsze 10 target tokenów:")
print(target_ids[:10])

print()
print("Największy token:")
print(max(dataset.tokens))