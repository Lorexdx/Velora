from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOADED_DIR = PROJECT_ROOT / "data" / "downloaded"
OUTPUT_PATH = PROJECT_ROOT / "data" / "final" / "velora_base_corpus.txt"


def append_text_file(output, source_path):
    with open(source_path, "r", encoding="utf-8", errors="ignore") as source:
        shutil.copyfileobj(source, output, length=1024 * 1024)

    output.write("\n\n")


def main():
    wikipedia_path = DOWNLOADED_DIR / "wikipedia_pl.txt"
    books_dir = DOWNLOADED_DIR / "books_pl"

    if not wikipedia_path.exists():
        raise FileNotFoundError(f"Brak pliku: {wikipedia_path}")

    book_paths = sorted(books_dir.glob("*.txt"))

    if not book_paths:
        raise FileNotFoundError(f"Brak polskich ksiazek w: {books_dir}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="\n") as output:
        append_text_file(output, wikipedia_path)

        for book_path in book_paths:
            append_text_file(output, book_path)

    print(f"Zapisano: {OUTPUT_PATH}")
    print(f"Polskie ksiazki: {len(book_paths)}")
    print(f"Rozmiar: {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
