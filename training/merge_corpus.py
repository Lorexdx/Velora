from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

files = [
    PROJECT_ROOT / "data" / "final" / "velora_corpus.txt",
    PROJECT_ROOT / "data" / "final" / "instruction_corpus_v2.txt",
    PROJECT_ROOT / "data" / "downloaded" / "wikipedia_chat_polish.txt"
]

output_file = (
    PROJECT_ROOT
    / "data"
    / "final"
    / "velora_corpus_v2.txt"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as out:

    for file in files:

        print("Łączenie:", file.name)

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            out.write(f.read())
            out.write("\n\n")

print()
print("Gotowe:", output_file)