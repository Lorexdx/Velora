from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

files = [
    PROJECT_ROOT / "data" / "final" / "velora_corpus.txt",
    PROJECT_ROOT / "data" / "final" / "instruction_corpus_v2.txt",
    PROJECT_ROOT / "data" / "downloaded" / "wikipedia_chat_polish.txt"
]

total_mb = 0

for file in files:

    size_mb = file.stat().st_size / 1024 / 1024

    print(
        f"{file.name}: "
        f"{size_mb:.2f} MB"
    )

    total_mb += size_mb

print()
print(f"RAZEM: {total_mb:.2f} MB")