"""Configuration for the training run."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    train_manifest: str = "data/shards/train_shards.json"
    val_manifest: str = "data/shards/val_shards.json"
    seq_len: int = 32
    batch_size: int = 4
    max_steps: int = 200
    learning_rate: float = 3e-4
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_iters: int = 5
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 100
    seed: int = 1337
