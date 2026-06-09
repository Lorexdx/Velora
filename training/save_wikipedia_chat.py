from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset = load_dataset(
    "netrixsa/wikipedia_chat_polish"
)

print("Rekordów:", len(dataset["train"]))

output_file = (
    PROJECT_ROOT
    / "data"
    / "downloaded"
    / "wikipedia_chat_polish.txt"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    for row in dataset["train"]:
        f.write(row["text"])
        f.write("\n\n")

print("Zapisano:", output_file)