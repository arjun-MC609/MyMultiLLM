"""CUDA FSDP training for models too large to replicate with DDP.

Launch with torchrun. This intentionally requires CUDA and NCCL: CPU/Gloo is
useful for tests but is not a practical backend for GPT-2-class pretraining.
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    LocalStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

from dataset.loader import ShardedTokenDataset
from models.profiles import PROFILES, get_model_profile
from models.transformer_block import TransformerBlock
from training.config import TrainConfig
from training.distributed_train import partition_shards, _write_partitioned_manifest
from training.lr_schedule import get_lr
from training.train import cycle


def _save_local_checkpoint(model, optimizer, checkpoint_path: Path, step: int) -> None:
    """Save one sharded model state per rank without gathering a 4B model."""
    with FSDP.state_dict_type(
        model, StateDictType.LOCAL_STATE_DICT, LocalStateDictConfig(offload_to_cpu=True),
    ):
        model_state = model.state_dict()
    torch.save({"step": step, "model": model_state, "optimizer": optimizer.state_dict()}, checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="FSDP pretraining for GPT-2-class and larger models.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="gpt2-small")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--cpu-offload", action="store_true", help="Save GPU memory at a substantial speed cost.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP training requires CUDA GPUs.")
    if not {"RANK", "WORLD_SIZE", "LOCAL_RANK"}.issubset(os.environ):
        raise RuntimeError("Launch with torchrun, for example: torchrun --nproc_per_node=8 -m training.fsdp_train ...")

    rank, world_size, local_rank = (int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")

    model_config = get_model_profile(args.profile)
    if args.seq_len > model_config.max_seq_len:
        raise ValueError("seq_len cannot exceed the selected profile's max_seq_len")
    train_config = TrainConfig(
        train_manifest=args.train_manifest, checkpoint_dir=args.checkpoint_dir,
        seq_len=args.seq_len, batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, max_steps=args.max_steps,
        learning_rate=args.learning_rate, precision="bf16",
        checkpoint_interval=args.checkpoint_interval,
    )

    shard_list = partition_shards(args.train_manifest, rank, world_size)
    manifest = _write_partitioned_manifest(shard_list, rank, str(Path(args.checkpoint_dir) / "_rank_manifests"))
    loader = DataLoader(ShardedTokenDataset(manifest, args.seq_len), batch_size=args.batch_size)
    data_iter = cycle(loader)

    from models.model import TinyTransformerLM
    model = TinyTransformerLM(model_config)
    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload),
        device_id=device,
        use_orig_params=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for step in range(args.max_steps):
        for group in optimizer.param_groups:
            group["lr"] = get_lr(step, train_config)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro_step in range(args.grad_accum):
            input_ids, target_ids = next(data_iter)
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            # Synchronize gradients only for the last micro-batch.
            sync_context = model.no_sync() if micro_step < args.grad_accum - 1 else torch.enable_grad()
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                loss = loss_fn(logits.flatten(0, 1), target_ids.flatten()) / args.grad_accum
            loss.backward()
            total_loss += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if rank == 0 and step % 10 == 0:
            print(f"step {step}/{args.max_steps} | loss {total_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)
        # Each rank writes its local FSDP shard without gathering the full model.
        # A resume loader must use the same world size and these rank-local files.
        if step > 0 and step % args.checkpoint_interval == 0:
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            _save_local_checkpoint(model, optimizer, Path(args.checkpoint_dir) / f"rank_{rank}_step_{step}.pt", step)
        dist.barrier()

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    _save_local_checkpoint(model, optimizer, Path(args.checkpoint_dir) / f"rank_{rank}_final.pt", args.max_steps)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
