import argparse
import re

import torch

from training.generate_text import (
    generate,
    load_model,
    load_tokenizer,
    resolve_checkpoint,
    trim_after_stop_sequences
)

STOP_SEQUENCES = [
    "<|",
    "<tool",
    "\nTy:",
    "\nUser:",
    "\nUzytkownik:",
    "\nAssistant:",
    "\nVelora:"
]


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with Velora in terminal.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument(
        "--checkpoint",
        choices=["best", "latest", "chat_best", "chat_latest"],
        default="chat_best"
    )

    return parser.parse_args()


def build_chat_prompt(user_message):
    return (
        "Rozmowa z prywatnym AI o imieniu Velora.\n"
        "Velora odpowiada krotko, konkretnie i po polsku.\n\n"
        f"Uzytkownik: {user_message}\n"
        "Velora:"
    )


def clean_response(generated_text, chat_prompt):
    response = generated_text[len(chat_prompt):]
    response = trim_after_stop_sequences(response, STOP_SEQUENCES)
    response = response.replace("\r\n", "\n")
    response = re.sub(r"\n{3,}", "\n\n", response)
    response = response.strip()

    if not response:
        return "(brak odpowiedzi)"

    return response


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
        user_message = input("Ty: ").strip()

        if user_message.lower() in {"/exit", "exit", "quit"}:
            break

        if not user_message:
            continue

        chat_prompt = build_chat_prompt(user_message)
        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=chat_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            stop_sequences=STOP_SEQUENCES
        )
        response = clean_response(generated_text, chat_prompt)

        print()
        print("Velora:")
        print(response)
        print()


if __name__ == "__main__":
    main()
