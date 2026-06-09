from pathlib import Path

import torch

from utils.console import (
    format_checkpoint_loaded,
    format_checkpoint_saved,
    format_checkpoint_skipped
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LATEST_CHECKPOINT = "velora_latest.pt"
BEST_CHECKPOINT = "velora_best.pt"

CHECKPOINT_DIR.mkdir(exist_ok=True)


def _checkpoint_payload(model, optimizer, scheduler, step, loss, best_loss, config):
    scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None

    return {
        "step": step,
        "loss": loss,
        "best_loss": best_loss,
        "config": config or {},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler_state_dict
    }


def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
    scheduler=None,
    best_loss=None,
    config=None,
    is_best=False
):
    best_loss = loss if best_loss is None else best_loss
    checkpoint = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        loss=loss,
        best_loss=best_loss,
        config=config
    )

    latest_path = CHECKPOINT_DIR / LATEST_CHECKPOINT
    torch.save(checkpoint, latest_path)
    print(format_checkpoint_saved(latest_path, step, loss))

    if is_best:
        best_path = CHECKPOINT_DIR / BEST_CHECKPOINT
        torch.save(checkpoint, best_path)
        print(format_checkpoint_saved(best_path, step, loss, is_best=True))


def load_checkpoint(
    model,
    optimizer,
    device,
    scheduler=None,
    checkpoint_name=LATEST_CHECKPOINT
):
    path = CHECKPOINT_DIR / checkpoint_name

    if not path.exists():
        return 0, float("inf")

    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    except (RuntimeError, KeyError, EOFError, OSError) as error:
        print(format_checkpoint_skipped(str(error)))
        return 0, float("inf")

    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", float("inf"))
    best_loss = checkpoint.get("best_loss", loss)

    print(format_checkpoint_loaded(path, step, loss))

    return step, best_loss
