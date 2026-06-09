from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

corpus_file = PROJECT_ROOT / "data" / "final" / "velora_corpus.txt"

with open(corpus_file, "r", encoding="utf-8") as f:
    text = f.read()

characters = len(text)
words = len(text.split())

print(f"Znaki: {characters:,}")
print(f"Słowa: {words:,}")
print(f"Rozmiar: {corpus_file.stat().st_size / 1024 / 1024:.2f} MB")