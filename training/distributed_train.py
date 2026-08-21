"""Multi-process / multi-machine training using PyTorch Distributed (Gloo)."""

import logging
import os
import time
import json
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from dataset.loader import ShardedTokenDataset
from models.config import ModelConfig
from models.model import TinyTransformerLM
from training.config import TrainConfig
from training.lr_schedule import get_lr
from training.train import _format_duration, cycle, evaluate, load_checkpoint, save_checkpoint
from training.utils import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | rank=%(rank)s | %(levelname)s | %(message)s",
)


def _rank_filter(record):
    record.rank = os.environ.get("RANK", "?")
    return True


logger.addFilter(_rank_filter)


def setup_distributed():
    required = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    missing = [v for v in required if v not in os.environ]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s) {missing}. This script must be "
            f"launched with torchrun, e.g.:\n"
            f"    torchrun --nproc_per_node=2 training/distributed_train.py [args]"
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    is_main_process = rank == 0
    logger.info("Process group initialized: rank=%d, world_size=%d", rank, world_size)

    return rank, world_size, is_main_process


def partition_shards(manifest_path: str, rank: int, world_size: int) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        all_shards = json.load(f)

    if len(all_shards) < world_size:
        raise ValueError(
            f"Only {len(all_shards)} shard(s) available but world_size={world_size} "
            f"-- some workers would receive no data. Use fewer workers or more shards."
        )

    my_shards = all_shards[rank::world_size]
    logger.info("Rank %d: assigned %d/%d shards from %s", rank, len(my_shards), len(all_shards), manifest_path)
    return my_shards


def _write_partitioned_manifest(shards, rank, tmp_dir):
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(tmp_dir, f"rank_{rank}_shards.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(shards, f)
    return path


def distributed_train(model_config: ModelConfig, train_config: TrainConfig) -> None:
    rank, world_size, is_main = setup_distributed()
    set_seed(train_config.seed + rank)

    device = torch.device("cpu")

    my_train_shards = partition_shards(train_config.train_manifest, rank, world_size)
    tmp_manifest_dir = os.path.join(train_config.checkpoint_dir, "_rank_manifests")
    my_train_manifest = _write_partitioned_manifest(my_train_shards, rank, tmp_manifest_dir)

    train_ds = ShardedTokenDataset(my_train_manifest, seq_len=train_config.seq_len)
    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size)
    train_iter = cycle(train_loader)

    val_iter = None
    if is_main:
        val_ds = ShardedTokenDataset(train_config.val_manifest, seq_len=train_config.seq_len)
        val_loader = DataLoader(val_ds, batch_size=train_config.batch_size)
        val_iter = cycle(val_loader)

    model = TinyTransformerLM(model_config).to(device)
    ddp_model = DDP(model)

    if is_main:
        logger.info("Model has %s parameters (x%d workers)", f"{model.num_parameters():,}", world_size)

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    start_step = 0
    if is_main:
        latest_ckpt = Path(train_config.checkpoint_dir) / "checkpoint_latest.pt"
        if latest_ckpt.is_file():
            logger.info("Found existing checkpoint -- resuming.")
            start_step = load_checkpoint(str(latest_ckpt), model, optimizer, device)

    start_step_tensor = torch.tensor([start_step], dtype=torch.long)
    dist.broadcast(start_step_tensor, src=0)
    start_step = int(start_step_tensor.item())

    if start_step >= train_config.max_steps:
        if is_main:
            logger.info("Checkpoint already at/past max_steps -- nothing to do.")
        dist.destroy_process_group()
        return

    ddp_model.train()
    t_start = time.time()

    for step in range(start_step, train_config.max_steps):
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids, target_ids = next(train_iter)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)

        logits = ddp_model(input_ids)
        loss = loss_fn(logits.view(-1, model_config.vocab_size), target_ids.view(-1))

        if model_config.use_moe:
            loss = loss + model_config.moe_aux_loss_weight * model.last_aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), train_config.grad_clip)

        optimizer.step()

        if is_main and step % train_config.log_interval == 0:
            elapsed = time.time() - t_start
            logger.info(
                "step %d/%d | loss %.4f | lr %.2e | elapsed %s | world_size=%d",
                step, train_config.max_steps, loss.item(), lr,
                _format_duration(elapsed), world_size,
            )

        if is_main and step > 0 and step % train_config.eval_interval == 0:
            val_loss = evaluate(model, val_iter, device, train_config.eval_iters, model_config.vocab_size)
            logger.info("step %d | VALIDATION loss %.4f", step, val_loss)

        if is_main and step > 0 and step % train_config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, model_config, train_config.checkpoint_dir)

        dist.barrier()

    if is_main:
        final_val_loss = evaluate(model, val_iter, device, train_config.eval_iters, model_config.vocab_size)
        total_elapsed = time.time() - t_start
        logger.info(
            "Distributed training complete. Final val loss: %.4f | Total time: %s | world_size=%d",
            final_val_loss, _format_duration(total_elapsed), world_size,
        )
        save_checkpoint(model, optimizer, train_config.max_steps, model_config,
                         train_config.checkpoint_dir, filename="checkpoint_final.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distributed training via PyTorch Distributed (Gloo).")
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--train-manifest", type=str, default="data/shards/train_shards.json")
    parser.add_argument("--val-manifest", type=str, default="data/shards/val_shards.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/distributed_run")
    args = parser.parse_args()

    m_cfg = ModelConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff, max_seq_len=args.max_seq_len,
    )
    t_cfg = TrainConfig(
        train_manifest=args.train_manifest, val_manifest=args.val_manifest,
        seq_len=args.seq_len, batch_size=args.batch_size, max_steps=args.max_steps,
        learning_rate=args.learning_rate, checkpoint_dir=args.checkpoint_dir,
    )
    distributed_train(m_cfg, t_cfg)
