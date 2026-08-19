"""Main training loop for the tiny transformer language model."""

import logging
import os
import random
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.loader import ShardedTokenDataset
from models.config import ModelConfig
from models.model import TinyTransformerLM
from training.config import TrainConfig
from training.lr_schedule import get_lr
from training.utils import get_device, set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, val_loader_iter, device, eval_iters, vocab_size):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(eval_iters):
        input_ids, target_ids = next(val_loader_iter)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, vocab_size), target_ids.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    model,
    optimizer,
    step,
    model_config,
    checkpoint_dir,
    filename="checkpoint_latest.pt",
):
    """Save a training checkpoint atomically, including RNG state for exact resume."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    final_path = Path(checkpoint_dir) / filename
    tmp_path = Path(checkpoint_dir) / f".{filename}.tmp"

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config.__dict__,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
    }

    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, final_path)  # atomic on POSIX systems
    logger.info("Saved checkpoint to %s (step %d)", final_path, step)
    return str(final_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    """Load a checkpoint in-place. Returns the step it was saved at."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Always load to CPU first, regardless of target device. torch.set_rng_state
    # requires a CPU ByteTensor specifically -- if we map_location the whole
    # checkpoint onto a GPU, the saved RNG state tensor gets relocated to GPU
    # too and set_rng_state() raises a TypeError. Model/optimizer weights are
    # moved to the target device explicitly below instead.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if device is not None:
            # optimizer.load_state_dict restores tensors on CPU (since we
            # loaded the checkpoint to CPU); move any per-parameter state
            # (e.g. Adam's momentum buffers) onto the target device too, or
            # the next optimizer.step() will hit a device-mismatch error.
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)

    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])  # always a CPU ByteTensor now
        if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])

    step = checkpoint.get("step", 0)
    logger.info(
        "Loaded checkpoint from %s (resuming at step %d)", checkpoint_path, step
    )
    return step


def train(model_config: ModelConfig, train_config: TrainConfig) -> nn.Module:
    set_seed(train_config.seed)
    device = get_device()

    logger.info("Loading datasets...")
    train_ds = ShardedTokenDataset(
        train_config.train_manifest, seq_len=train_config.seq_len
    )
    val_ds = ShardedTokenDataset(
        train_config.val_manifest, seq_len=train_config.seq_len
    )

    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size)

    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)

    logger.info("Building model...")
    model = TinyTransformerLM(model_config).to(device)
    logger.info("Model has %s parameters", f"{model.num_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    # Auto-resume: if a checkpoint already exists, pick up from there instead
    # of starting over. This is what lets a Colab session be stopped and
    # resumed later, or a crashed local run continue.
    start_step = 0
    latest_ckpt = Path(train_config.checkpoint_dir) / "checkpoint_latest.pt"
    if latest_ckpt.is_file():
        logger.info("Found existing checkpoint at %s — resuming.", latest_ckpt)
        start_step = load_checkpoint(str(latest_ckpt), model, optimizer, device)
    else:
        logger.info("No existing checkpoint found — starting fresh from step 0.")

    if start_step >= train_config.max_steps:
        logger.info(
            "Checkpoint step %d already >= max_steps %d — nothing to do.",
            start_step,
            train_config.max_steps,
        )
        return model

    model.train()
    t_start = time.time()

    for step in range(start_step, train_config.max_steps):
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids, target_ids = next(train_iter)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, model_config.vocab_size), target_ids.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        optimizer.step()

        if step % train_config.log_interval == 0:
            elapsed = time.time() - t_start
            logger.info(
                "step %d/%d | loss %.4f | lr %.2e | elapsed %.1fs",
                step,
                train_config.max_steps,
                loss.item(),
                lr,
                elapsed,
            )

        if step > 0 and step % train_config.eval_interval == 0:
            val_loss = evaluate(
                model,
                val_iter,
                device,
                train_config.eval_iters,
                model_config.vocab_size,
            )
            logger.info("step %d | VALIDATION loss %.4f", step, val_loss)

        if step > 0 and step % train_config.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, step, model_config, train_config.checkpoint_dir
            )

    final_val_loss = evaluate(
        model, val_iter, device, train_config.eval_iters, model_config.vocab_size
    )
    logger.info("Training complete. Final validation loss: %.4f", final_val_loss)

    save_checkpoint(
        model,
        optimizer,
        train_config.max_steps,
        model_config,
        train_config.checkpoint_dir,
        filename="checkpoint_final.pt",
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the tiny transformer LM.")
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--train-manifest", type=str, default="data/shards/train_shards.json"
    )
    parser.add_argument(
        "--val-manifest", type=str, default="data/shards/val_shards.json"
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    m_cfg = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    )
    t_cfg = TrainConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
    )
    train(m_cfg, t_cfg)
