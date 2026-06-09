from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = PROJECT_ROOT / "data" / "final" / "velora_mixed_corpus.txt"

SOURCES = [
    {
        "path": PROJECT_ROOT / "data" / "downloaded" / "wikipedia_pl.txt",
        "max_ratio": 0.70
    },
    {
        "path": PROJECT_ROOT / "data" / "final" / "instruction_corpus_v2.txt",
        "max_ratio": 1.00
    },
    {
        "path": PROJECT_ROOT / "data" / "final" / "velora_chat_corpus.txt",
        "max_ratio": 1.00
    },
]


def read_limited_text(path, max_ratio):
    size = path.stat().st_size
    max_chars = int(size * max_ratio)

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read(max_chars)


def normalize_spacing(text):
    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]

    return "\n".join(clean_lines)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as output:
        for source in SOURCES:
            path = source["path"]
            max_ratio = source["max_ratio"]

            if not path.exists():
                print(f"Pominieto brakujacy plik: {path}")
                continue

            text = read_limited_text(path, max_ratio)
            text = normalize_spacing(text)

            output.write(text)
            output.write("\n\n")

            print(
                f"Dodano: {path.name} "
                f"({max_ratio:.0%}, {len(text):,} znakow)"
            )

    print(f"Gotowe: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
