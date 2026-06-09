import argparse

import torch

from training.generate_text import (
    generate,
    load_model,
    load_tokenizer,
    resolve_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Velora in terminal.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--checkpoint",
        choices=["best", "latest"],
        default="best"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model = load_model(checkpoint_path, device)
    tokenizer = load_tokenizer()

    print("Velora chat")
    print(f"Checkpoint: {checkpoint_path}")
    print("Wpisz /exit, zeby zakonczyc.")
    print()

    while True:
        prompt = input("Ty: ").strip()

        if prompt.lower() in {"/exit", "exit", "quit"}:
            break

        if not prompt:
            continue

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )

        print()
        print("Velora:")
        print(response)
        print()


if __name__ == "__main__":
    main()
