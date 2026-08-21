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
    gradient_accumulation_steps: int = 1
    precision: str = "auto"  # auto, fp32, bf16, or fp16
    eval_interval: int = 50
    eval_iters: int = 5
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 100
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.precision not in {"auto", "fp32", "bf16", "fp16"}:
            raise ValueError("precision must be auto, fp32, bf16, or fp16")
