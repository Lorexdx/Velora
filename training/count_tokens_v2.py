from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tokenizer = ByteLevelBPETokenizer(
    str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
    str(PROJECT_ROOT / "tokenizers" / "merges.txt")
)

corpus_file = (
    PROJECT_ROOT
    / "data"
    / "final"
    / "velora_corpus_v2.txt"
)

with open(
    corpus_file,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()

tokens = tokenizer.encode(text).ids

print("Tokenów:", len(tokens))