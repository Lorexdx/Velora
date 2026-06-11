import argparse
import math
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.transformer import VeloraTransformer
from training.dataset import VeloraDataset
from training.save_checkpoint import load_checkpoint, save_checkpoint
from utils.console import (
    format_device,
    format_evaluation_step,
    format_training_interrupted,
    format_training_step
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VOCAB_SIZE = 32000
SEQ_LENGTH = 256
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-5
WARMUP_STEPS = 1000
MAX_STEPS = 50000
SAVE_EVERY = 1000
EVAL_EVERY = 1000
EVAL_BATCHES = 20
VAL_RATIO = 0.05
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 2
TOKEN_CACHE_PATH = (
    PROJECT_ROOT / "data" / "cache" / "velora_base_corpus_tokens.npy"
)
CORPUS_PATH = PROJECT_ROOT / "data" / "final" / "velora_base_corpus.txt"

# Kept for modules that import the default training configuration.
TRAINING_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "seq_length": SEQ_LENGTH,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "min_learning_rate": MIN_LEARNING_RATE,
    "warmup_steps": WARMUP_STEPS,
    "max_steps": MAX_STEPS,
    "save_every": SAVE_EVERY,
    "eval_every": EVAL_EVERY,
    "eval_batches": EVAL_BATCHES,
    "val_ratio": VAL_RATIO,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "token_cache_path": str(TOKEN_CACHE_PATH)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Velora from scratch on the base corpus."
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS
    )
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=MIN_LEARNING_RATE
    )
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start from new weights and ignore velora_latest.pt."
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision on CUDA."
    )

    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    if args.seq_length < 2:
        parser.error("--seq-length must be at least 2")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps cannot be negative")
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    if args.eval_every < 1:
        parser.error("--eval-every must be at least 1")
    if args.eval_batches < 1:
        parser.error("--eval-batches must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.min_learning_rate > args.learning_rate:
        parser.error("--min-learning-rate cannot exceed --learning-rate")

    return args


def build_dataloaders(
    seq_length=SEQ_LENGTH,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
):
    dataset = VeloraDataset(
        corpus_path=CORPUS_PATH,
        vocab_path=str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
        merges_path=str(PROJECT_ROOT / "tokenizers" / "merges.txt"),
        seq_length=seq_length,
        max_chars=None,
        token_cache_path=TOKEN_CACHE_PATH
    )

    token_count = len(dataset.tokens)
    train_end = int(token_count * (1.0 - VAL_RATIO))

    train_dataset = VeloraDataset(
        corpus_path=CORPUS_PATH,
        vocab_path=str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
        merges_path=str(PROJECT_ROOT / "tokenizers" / "merges.txt"),
        seq_length=seq_length,
        max_chars=None,
        token_cache_path=TOKEN_CACHE_PATH,
        start_idx=0,
        end_idx=train_end,
        stride=seq_length
    )
    val_dataset = VeloraDataset(
        corpus_path=CORPUS_PATH,
        vocab_path=str(PROJECT_ROOT / "tokenizers" / "vocab.json"),
        merges_path=str(PROJECT_ROOT / "tokenizers" / "merges.txt"),
        seq_length=seq_length,
        max_chars=None,
        token_cache_path=TOKEN_CACHE_PATH,
        start_idx=train_end,
        end_idx=token_count,
        stride=seq_length
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available()
    }

    if num_workers > 0:
        loader_options["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_options
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_options
    )

    return train_loader, val_loader, token_count


def compute_loss(model, loss_fn, input_ids, target_ids):
    logits = model(input_ids)

    return loss_fn(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1)
    )


def get_amp_dtype(device, amp_enabled):
    if not amp_enabled or device.type != "cuda":
        return None

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def autocast_context(device, amp_dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_dtype is not None
    )


def evaluate(
    model,
    loss_fn,
    val_loader,
    device,
    eval_batches=EVAL_BATCHES,
    amp_dtype=None
):
    model.eval()
    losses = []

    with torch.inference_mode():
        for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
            if batch_idx >= eval_batches:
                break

            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            with autocast_context(device, amp_dtype):
                loss = compute_loss(model, loss_fn, input_ids, target_ids)

            losses.append(loss.item())

    model.train()

    if not losses:
        raise RuntimeError("Validation dataset does not contain any full batches.")

    return sum(losses) / len(losses)


def build_scheduler(
    optimizer,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    min_learning_rate=MIN_LEARNING_RATE
):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, (step + 1) / max(1, warmup_steps))

        progress = min(
            1.0,
            (step - warmup_steps) / max(1, max_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min_learning_rate / learning_rate

        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]


def infinite_batches(data_loader):
    while True:
        yield from data_loader


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(device, not args.no_amp)
    print(format_device(device))

    train_loader, val_loader, corpus_tokens = build_dataloaders(
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = VeloraTransformer(
        vocab_size=VOCAB_SIZE,
        block_size=args.seq_length
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate
    )

    if args.reset:
        start_step = 0
        best_loss = float("inf")
    else:
        start_step, best_loss = load_checkpoint(
            model,
            optimizer,
            device,
            scheduler=scheduler
        )

    loss_fn = nn.CrossEntropyLoss()
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    current_step = start_step
    current_loss = None
    batch_iterator = infinite_batches(train_loader)

    effective_batch_size = (
        args.batch_size * args.gradient_accumulation_steps
    )
    tokens_per_step = effective_batch_size * args.seq_length
    planned_tokens = args.max_steps * tokens_per_step
    planned_corpus_passes = planned_tokens / max(
        1,
        corpus_tokens * (1.0 - VAL_RATIO)
    )
    training_config = {
        "vocab_size": VOCAB_SIZE,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
        "eval_batches": args.eval_batches,
        "val_ratio": VAL_RATIO,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "corpus_tokens": corpus_tokens,
        "planned_tokens": planned_tokens,
        "planned_corpus_passes": planned_corpus_passes,
        "token_cache_path": str(TOKEN_CACHE_PATH),
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else None
    }

    print(
        f"Parametrow: {sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    print(
        f"Tokenow w corpusie: {corpus_tokens:,} | "
        f"tokenow na krok: {tokens_per_step:,} | "
        f"plan: {planned_tokens:,} tokenow "
        f"({planned_corpus_passes:.2f} przejscia przez dane)"
    )

    def persist_checkpoint(step, loss, is_best=False):
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            loss=loss,
            best_loss=best_loss,
            config=training_config,
            is_best=is_best
        )

    try:
        while current_step < args.max_steps:
            step_started_at = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            processed_tokens = 0

            for _ in range(args.gradient_accumulation_steps):
                input_ids, target_ids = next(batch_iterator)
                input_ids = input_ids.to(device, non_blocking=True)
                target_ids = target_ids.to(device, non_blocking=True)

                with autocast_context(device, amp_dtype):
                    loss = compute_loss(model, loss_fn, input_ids, target_ids)
                    scaled_loss = loss / args.gradient_accumulation_steps

                scaler.scale(scaled_loss).backward()
                accumulated_loss += loss.item()
                processed_tokens += input_ids.numel()

            if use_scaler:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP_NORM
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            current_step += 1
            current_loss = (
                accumulated_loss / args.gradient_accumulation_steps
            )
            elapsed = max(time.perf_counter() - step_started_at, 1e-8)
            tokens_per_second = processed_tokens / elapsed
            learning_rate = get_learning_rate(optimizer)

            if current_step % args.eval_every == 0:
                val_loss = evaluate(
                    model,
                    loss_fn,
                    val_loader,
                    device,
                    eval_batches=args.eval_batches,
                    amp_dtype=amp_dtype
                )
                is_best = val_loss < best_loss

                if is_best:
                    best_loss = val_loss

                persist_checkpoint(
                    current_step,
                    val_loss,
                    is_best=is_best
                )
                print(format_evaluation_step(
                    current_step,
                    current_loss,
                    val_loss,
                    best_loss
                ))
            elif current_step % args.save_every == 0:
                persist_checkpoint(current_step, current_loss)

            print(format_training_step(
                current_step,
                current_loss,
                learning_rate,
                tokens_per_second
            ))

        if (
            current_loss is not None
            and current_step % args.save_every != 0
            and current_step % args.eval_every != 0
        ):
            persist_checkpoint(current_step, current_loss)
    except KeyboardInterrupt:
        print(format_training_interrupted())

        if current_loss is not None:
            persist_checkpoint(current_step, current_loss)


if __name__ == "__main__":
    main()
