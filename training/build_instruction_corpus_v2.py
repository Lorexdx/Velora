from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dolly_file = PROJECT_ROOT / "data" / "downloaded" / "dolly_15k.jsonl"
owca_file = PROJECT_ROOT / "data" / "downloaded" / "owca.jsonl"

output_file = PROJECT_ROOT / "data" / "final" / "instruction_corpus_v2.txt"

count = 0

with open(output_file, "w", encoding="utf-8") as outfile:

    for source_file in [dolly_file, owca_file]:

        with open(source_file, "r", encoding="utf-8") as infile:

            for line in infile:
                row = json.loads(line)

                instruction = row["instruction"].strip()
                response = row["response"].strip()

                outfile.write(f"Użytkownik: {instruction}\n")
                outfile.write(f"Velora: {response}\n\n")

                count += 1

print(f"Przetworzono: {count}")
print(f"Zapisano: {output_file}")