from datasets import load_dataset
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

output_dir = PROJECT_ROOT / "data" / "downloaded"
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "emplocity/owca",
    split="train"
)

output_file = output_dir / "owca.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for row in dataset:
        record = {
            "instruction": row["instruction"],
            "response": row["output"]
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Rekordów:", len(dataset))
print("Zapisano:", output_file)