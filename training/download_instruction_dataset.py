from datasets import load_dataset
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

output_dir = PROJECT_ROOT / "data" / "downloaded"
output_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "databricks/databricks-dolly-15k",
    split="train"
)

output_file = output_dir / "dolly_15k.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for row in dataset:
        record = {
            "instruction": row["instruction"],
            "response": row["response"]
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Gotowe:", output_file)
print("Przykład:")
print(dataset[0]["instruction"])