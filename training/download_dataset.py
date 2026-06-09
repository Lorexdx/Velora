from datasets import load_dataset

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.pl",
    split="train[:1%]"
)

with open("data/downloaded/wikipedia_pl.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        text = item.get("text", "")
        if text:
            f.write(text + "\n")