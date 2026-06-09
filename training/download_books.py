from datasets import load_dataset

dataset = load_dataset(
    "oscar-corpus/OSCAR-2301",
    language="pl",
    split="train[:0.05%]")

with open("data/downloaded/oscar_pl.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        text = item.get("text", "")
        if text:
            f.write(text + "\n")