from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

downloaded_dir = PROJECT_ROOT / "data" / "downloaded"
output_file = PROJECT_ROOT / "data" / "final" / "velora_corpus.txt"

with open(output_file, "w", encoding="utf-8") as out:

    wikipedia = downloaded_dir / "wikipedia_pl.txt"

    if wikipedia.exists():
        with open(wikipedia, "r", encoding="utf-8") as f:
            out.write(f.read())
            out.write("\n")

    books_pl_dir = downloaded_dir / "books_pl"
    books_en_dir = downloaded_dir / "books_en"

    if books_pl_dir.exists():
        for book in books_pl_dir.glob("*.txt"):
            with open(book, "r", encoding="utf-8") as f:
                out.write(f.read())
                out.write("\n")

    if books_en_dir.exists():
        for book in books_en_dir.glob("*.txt"):
            with open(book, "r", encoding="utf-8") as f:
                out.write(f.read())
                out.write("\n")

print(f"Gotowe: {output_file}")