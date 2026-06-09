from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
    "tokenizers/vocab.json",
    "tokenizers/merges.txt"
)

text = "Cześć, jestem Velora."

encoded = tokenizer.encode(text)

print("Tokeny:")
print(encoded.tokens)

print("\nID:")
print(encoded.ids)