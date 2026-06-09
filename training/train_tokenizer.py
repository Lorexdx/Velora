from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print("Start treningu tokenizera...")

corpus_file = (
    PROJECT_ROOT
    / "data"
    / "final"
    / "velora_corpus_v2.txt"
)

print("Plik istnieje:", corpus_file.exists())
print("Plik:", corpus_file)

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[str(corpus_file)],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>"
    ]
)

print("Zapisywanie...")

tokenizer.save_model(
    str(PROJECT_ROOT / "tokenizers")
)

print("GOTOWE")