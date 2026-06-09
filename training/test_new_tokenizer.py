from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tokenizer = ByteLevelBPETokenizer(
    str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
    str(PROJECT_ROOT / "tokenizers" / "merges.txt")
)

text = "Programowanie to sztuka rozwiązywania problemów."

encoding = tokenizer.encode(text)

print("Tokenów:", len(encoding.ids))
print(encoding.ids)