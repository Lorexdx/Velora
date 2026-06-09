from datasets import load_dataset

dataset = load_dataset("netrixsa/wikipedia_chat_polish")

print(dataset["train"][0])