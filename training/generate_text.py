import argparse
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

from models.transformer import VeloraTransformer
from training.save_checkpoint import (
    BEST_CHECKPOINT,
    CHAT_BEST_CHECKPOINT,
    CHAT_LATEST_CHECKPOINT,
    CHECKPOINT_DIR,
    LATEST_CHECKPOINT
)
from training.train_velora import SEQ_LENGTH, VOCAB_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with Velora.")
    parser.add_argument("--prompt", type=str, default="Czesc")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--checkpoint",
        choices=["best", "latest", "chat_best", "chat_latest"],
        default="best"
    )

    return parser.parse_args()


def resolve_checkpoint(checkpoint_choice):
    checkpoint_names = {
        "best": BEST_CHECKPOINT,
        "latest": LATEST_CHECKPOINT,
        "chat_best": CHAT_BEST_CHECKPOINT,
        "chat_latest": CHAT_LATEST_CHECKPOINT
    }
    preferred_name = checkpoint_names[checkpoint_choice]
    preferred_path = CHECKPOINT_DIR / preferred_name

    if preferred_path.exists():
        return preferred_path

    fallback_path = CHECKPOINT_DIR / LATEST_CHECKPOINT

    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        f"No checkpoint found in {CHECKPOINT_DIR}. Train the model first."
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    model = VeloraTransformer(
        vocab_size=config.get("vocab_size", VOCAB_SIZE),
        block_size=config.get("seq_length", SEQ_LENGTH)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def load_tokenizer():
    return ByteLevelBPETokenizer(
        str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
        str(PROJECT_ROOT / "tokenizers" / "merges.txt")
    )


def filter_top_k(logits, top_k):
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits

    values, _ = torch.topk(logits, top_k)
    cutoff = values[-1]

    return logits.masked_fill(logits < cutoff, float("-inf"))


def filter_top_p(logits, top_p):
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative_probabilities = torch.cumsum(probabilities, dim=-1)
    sorted_mask = cumulative_probabilities > top_p
    sorted_mask[1:] = sorted_mask[:-1].clone()
    sorted_mask[0] = False

    indices_to_remove = sorted_indices[sorted_mask]

    return logits.scatter(
        dim=-1,
        index=indices_to_remove,
        src=torch.full_like(indices_to_remove, float("-inf"), dtype=logits.dtype)
    )


def sample_next_token(logits, temperature, top_k, top_p):
    temperature = max(temperature, 1e-6)
    logits = logits / temperature
    logits = filter_top_k(logits, top_k)
    logits = filter_top_p(logits, top_p)
    probabilities = torch.softmax(logits, dim=-1)

    return torch.multinomial(probabilities, num_samples=1).item()


def trim_after_stop_sequences(text, stop_sequences):
    if not stop_sequences:
        return text

    stop_positions = [
        text.find(stop_sequence)
        for stop_sequence in stop_sequences
        if stop_sequence in text
    ]

    if not stop_positions:
        return text

    return text[:min(stop_positions)]


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    device,
    stop_sequences=None
):
    tokens = tokenizer.encode(prompt).ids
    prompt_length = len(prompt)

    for _ in range(max_new_tokens):
        context = tokens[-model.block_size:]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[0, -1]
        next_token_id = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        tokens.append(next_token_id)

        if stop_sequences:
            generated_suffix = tokenizer.decode(tokens)[prompt_length:]

            if any(stop_sequence in generated_suffix for stop_sequence in stop_sequences):
                break

    generated_text = tokenizer.decode(tokens)

    if stop_sequences:
        prefix = generated_text[:prompt_length]
        suffix = trim_after_stop_sequences(
            generated_text[prompt_length:],
            stop_sequences
        )

        return f"{prefix}{suffix}"

    return generated_text


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer()

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )

    print()
    print("Checkpoint:")
    print(checkpoint_path)
    print()
    print("Prompt:")
    print(args.prompt)
    print()
    print("Wynik:")
    print(generated_text)


if __name__ == "__main__":
    main()
