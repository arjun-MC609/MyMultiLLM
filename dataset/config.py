"""Configuration for the dataset preparation pipeline."""

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    seq_len: int = 64
    tokens_per_shard: int = 500_000
    val_fraction: float = 0.1
    min_line_length: int = 8
    tokenizer_dir: str = "tokenizer/trained_test"
    raw_dir: str = "data/raw"
    shards_dir: str = "data/shards"
    state_file: str = "data/shards/_prepare_state.json"
