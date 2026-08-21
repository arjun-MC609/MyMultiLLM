"""Launch a named model profile after sharded data has been prepared."""

import argparse

from models.profiles import PROFILES, get_model_profile
from training.config import TrainConfig
from training.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a named MyMultiLLM model profile.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="gpt2-small")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    model_config = get_model_profile(args.profile)
    if args.seq_len > model_config.max_seq_len:
        parser.error("--seq-len cannot exceed the selected profile's max_seq_len")
    train_config = TrainConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        checkpoint_dir=args.checkpoint_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        precision=args.precision,
    )
    train(model_config, train_config)


if __name__ == "__main__":
    main()
