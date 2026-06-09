from pathlib import Path
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.transformer import VeloraTransformer
from training.dataset import VeloraDataset
from training.save_checkpoint import (
    BEST_CHECKPOINT,
    CHAT_BEST_CHECKPOINT,
    CHAT_LATEST_CHECKPOINT,
    CHECKPOINT_DIR,
    load_checkpoint,
    save_checkpoint
)
from training.train_velora import compute_loss, evaluate, get_learning_rate
from utils.console import (
    format_device,
    format_evaluation_step,
    format_training_interrupted,
    format_training_step
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VOCAB_SIZE = 32000
SEQ_LENGTH = 128
BATCH_SIZE = 4
LEARNING_RATE = 3e-5
MIN_LEARNING_RATE = 5e-6
WARMUP_STEPS = 200
MAX_STEPS = 5000
SAVE_EVERY = 250
EVAL_EVERY = 500
EVAL_BATCHES = 20
VAL_RATIO = 0.10
GRAD_CLIP_NORM = 1.0
CHAT_CORPUS_PATH = PROJECT_ROOT / "data" / "final" / "velora_chat_corpus.txt"
TOKEN_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "velora_chat_corpus_tokens.npy"

TRAINING_CONFIG = {
    "mode": "chat_finetune",
    "vocab_size": VOCAB_SIZE,
    "seq_length": SEQ_LENGTH,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "min_learning_rate": MIN_LEARNING_RATE,
    "warmup_steps": WARMUP_STEPS,
    "max_steps": MAX_STEPS,
    "save_every": SAVE_EVERY,
    "eval_every": EVAL_EVERY,
    "eval_batches": EVAL_BATCHES,
    "val_ratio": VAL_RATIO,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "chat_corpus_path": str(CHAT_CORPUS_PATH),
    "token_cache_path": str(TOKEN_CACHE_PATH)
}


def build_dataloaders():
    dataset = VeloraDataset(
        corpus_path=CHAT_CORPUS_PATH,
        vocab_path=str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
        merges_path=str(PROJECT_ROOT / "tokenizers" / "merges.txt"),
        seq_length=SEQ_LENGTH,
        max_chars=None,
        token_cache_path=TOKEN_CACHE_PATH
    )

    val_size = max(1, int(len(dataset) * VAL_RATIO))
    train_size = len(dataset) - val_size

    train_dataset = VeloraDataset(
        corpus_path=CHAT_CORPUS_PATH,
        vocab_path=str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
        merges_path=str(PROJECT_ROOT / "tokenizers" / "merges.txt"),
        seq_length=SEQ_LENGTH,
        max_chars=None,
        token_cache_path=TOKEN_CACHE_PATH,
        start_idx=0,
        end_idx=train_size + SEQ_LENGTH + 1
    )
    val_dataset = VeloraDataset(
        corpus_path=CHAT_CORPUS_PATH,
        vocab_path=str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
        merges_path=str(PROJECT_ROOT / "tokenizers" / "merges.txt"),
        seq_length=SEQ_LENGTH,
        max_chars=None,
        token_cache_path=TOKEN_CACHE_PATH,
        start_idx=train_size,
        end_idx=len(dataset.tokens)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


def build_scheduler(optimizer):
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return max(1e-8, (step + 1) / WARMUP_STEPS)

        progress = min(
            1.0,
            (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = MIN_LEARNING_RATE / LEARNING_RATE

        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_base_model(model, device):
    base_path = CHECKPOINT_DIR / BEST_CHECKPOINT

    if not base_path.exists():
        raise FileNotFoundError(
            f"Missing base checkpoint: {base_path}. Train Velora first."
        )

    checkpoint = torch.load(base_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Wczytano bazowy model: {base_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(format_device(device))

    train_loader, val_loader = build_dataloaders()

    model = VeloraTransformer(
        vocab_size=VOCAB_SIZE,
        block_size=SEQ_LENGTH
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = build_scheduler(optimizer)

    start_step, best_loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        checkpoint_name=CHAT_LATEST_CHECKPOINT
    )

    if start_step == 0:
        load_base_model(model, device)

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    current_step = start_step
    current_loss = None

    def persist_checkpoint(step, loss, is_best=False):
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            loss=loss,
            best_loss=best_loss,
            config=TRAINING_CONFIG,
            is_best=is_best,
            latest_checkpoint_name=CHAT_LATEST_CHECKPOINT,
            best_checkpoint_name=CHAT_BEST_CHECKPOINT
        )

    try:
        for step, (input_ids, target_ids) in enumerate(train_loader, start=start_step):
            current_step = step
            step_started_at = time.perf_counter()

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            loss = compute_loss(model, loss_fn, input_ids, target_ids)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            elapsed = max(time.perf_counter() - step_started_at, 1e-8)
            tokens_per_second = input_ids.numel() / elapsed
            learning_rate = get_learning_rate(optimizer)

            if step % SAVE_EVERY == 0:
                persist_checkpoint(step, current_loss)

            if step % EVAL_EVERY == 0:
                val_loss = evaluate(model, loss_fn, val_loader, device)
                is_best = val_loss < best_loss

                if is_best:
                    best_loss = val_loss

                persist_checkpoint(step, val_loss, is_best=is_best)
                print(format_evaluation_step(step, current_loss, val_loss, best_loss))

            print(format_training_step(
                step,
                current_loss,
                learning_rate,
                tokens_per_second
            ))

            if step >= MAX_STEPS:
                break
    except KeyboardInterrupt:
        print(format_training_interrupted())

        if current_loss is not None:
            persist_checkpoint(current_step, current_loss)


if __name__ == "__main__":
    main()
