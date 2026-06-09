from datasets import load_dataset

dataset = load_dataset(
    "emplocity/owca",
    split="train[:10]"
)

print("Rekordów:", len(dataset))
print()
print("Instrukcja:")
print(dataset[0]["instruction"])
print()
print("Odpowiedź:")
print(dataset[0]["output"])