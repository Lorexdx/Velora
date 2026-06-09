from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["data/downloaded/wikipedia_pl.txt"],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>"
    ]
)

tokenizer.save_model("tokenizers")