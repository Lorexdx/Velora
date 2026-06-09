from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

corpus_file = PROJECT_ROOT / "data" / "final" / "velora_corpus.txt"

tokenizer = ByteLevelBPETokenizer(
    str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
    str(PROJECT_ROOT / "tokenizers" / "merges.txt")
)

with open(corpus_file, "r", encoding="utf-8") as f:
    text = f.read(1000)

encoding = tokenizer.encode(text)

print("Tokenów:", len(encoding.ids))
print()
print("Pierwsze 50 ID:")
print(encoding.ids[:50])