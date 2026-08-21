"""Create fresh English-only data and train the System identity model."""

from pathlib import Path

from dataset.config import DatasetConfig
from dataset.prepare import prepare_dataset
from models.config import ModelConfig
from tokenizer.config import TokenizerConfig
from tokenizer.train_tokenizer import save_tokenizer, train_tokenizer
from training.config import TrainConfig
from training.train import train


def main() -> None:
    raw_dir = "data/system_english_raw"
    tokenizer_dir = "tokenizer/system_english_v2"
    shards_dir = "data/system_english_shards_v2"
    checkpoint_dir = "checkpoints/system_english_v2"

    # This directory contains only English text; no Tamil text is included.
    english_files = [str(path) for path in sorted(Path(raw_dir).glob("*.txt"))]
    tokenizer_config = TokenizerConfig(
        vocab_size=500,
        min_frequency=1,
        save_dir=tokenizer_dir,
    )
    tokenizer = train_tokenizer(english_files, tokenizer_config)
    save_tokenizer(tokenizer, tokenizer_config)

    dataset_config = DatasetConfig(
        tokenizer_dir=tokenizer_dir,
        raw_dir=raw_dir,
        shards_dir=shards_dir,
        state_file=f"{shards_dir}/_prepare_state.json",
        tokens_per_shard=64,
        val_fraction=0.2,
    )
    prepare_dataset(dataset_config)

    model_config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_seq_len=64,
    )
    train_config = TrainConfig(
        train_manifest=f"{shards_dir}/train_shards.json",
        val_manifest=f"{shards_dir}/val_shards.json",
        seq_len=32,
        batch_size=4,
        max_steps=800,
        learning_rate=3e-4,
        checkpoint_dir=checkpoint_dir,
        eval_interval=100,
        checkpoint_interval=200,
    )
    train(model_config, train_config)


if __name__ == "__main__":
    main()
