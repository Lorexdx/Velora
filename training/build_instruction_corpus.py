from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_file = PROJECT_ROOT / "data" / "downloaded" / "dolly_15k.jsonl"
output_file = PROJECT_ROOT / "data" / "final" / "instruction_corpus.txt"

count = 0

with open(input_file, "r", encoding="utf-8") as infile:
    with open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            row = json.loads(line)

            instruction = row["instruction"].strip()
            response = row["response"].strip()

            outfile.write(f"Użytkownik: {instruction}\n")
            outfile.write(f"Velora: {response}\n\n")

            count += 1

print("Przetworzono:", count)
print("Zapisano:", output_file)