"""Colab-compatible training entry point."""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def print_gpu_info() -> None:
    if not torch.cuda.is_available():
        logger.warning(
            "No GPU detected! In Colab: Runtime -> Change runtime type -> "
            "Hardware accelerator -> GPU. Falling back to CPU (will be slow)."
        )
        return
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info("GPU detected: %s", gpu_name)
    logger.info("GPU memory: %.1f GB", gpu_mem_gb)
    logger.info("CUDA version: %s", torch.version.cuda)


def mount_drive_if_in_colab(mount_point: str = "/content/drive") -> bool:
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        logger.info("Not running in Colab (google.colab unavailable) — skipping Drive mount.")
        return False

    if os.path.isdir(mount_point) and os.listdir(mount_point):
        logger.info("Drive already mounted at %s", mount_point)
        return True

    logger.info("Mounting Google Drive at %s ...", mount_point)
    drive.mount(mount_point)
    return True


def setup_colab_paths(project_root_in_drive: str) -> dict:
    os.makedirs(project_root_in_drive, exist_ok=True)
    paths = {
        "train_manifest": os.path.join(project_root_in_drive, "data/shards/train_shards.json"),
        "val_manifest": os.path.join(project_root_in_drive, "data/shards/val_shards.json"),
        "tokenizer_dir": os.path.join(project_root_in_drive, "tokenizer/trained"),
        "checkpoint_dir": os.path.join(project_root_in_drive, "checkpoints/colab_run"),
    }
    for key, path in paths.items():
        logger.info("%s -> %s", key, path)
    return paths


def run_colab_training(
    project_root_in_drive: str = "/content/drive/MyDrive/MyMultiLLM",
    vocab_size: int = 8000,
    d_model: int = 384,
    n_layers: int = 6,
    n_heads: int = 6,
    d_ff: int = 1536,
    max_seq_len: int = 512,
    seq_len: int = 512,
    batch_size: int = 16,
    max_steps: int = 5000,
    learning_rate: float = 3e-4,
) -> None:
    sys.path.insert(0, "/content/MyMultiLLM")

    print_gpu_info()
    mount_drive_if_in_colab()
    paths = setup_colab_paths(project_root_in_drive)

    from models.config import ModelConfig
    from training.config import TrainConfig
    from training.train import train

    model_config = ModelConfig(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_ff=d_ff, max_seq_len=max_seq_len,
    )
    train_config = TrainConfig(
        train_manifest=paths["train_manifest"],
        val_manifest=paths["val_manifest"],
        seq_len=seq_len,
        batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        checkpoint_dir=paths["checkpoint_dir"],
        checkpoint_interval=200,
        eval_interval=200,
    )

    logger.info("Starting training. Checkpoints will auto-resume from %s if present.", paths["checkpoint_dir"])
    train(model_config, train_config)
    logger.info("Done. If your session disconnects later, just re-run this cell — it will resume automatically.")


if __name__ == "__main__":
    run_colab_training()
