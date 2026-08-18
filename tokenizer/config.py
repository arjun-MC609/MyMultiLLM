"""Configuration for tokenizer training."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TokenizerConfig:
    """Settings for training a byte-level BPE tokenizer.

    Attributes:
        vocab_size: Target vocabulary size (including special tokens).
        min_frequency: Minimum pair frequency to consider for a merge.
            Higher values -> fewer, more "confident" merges.
        special_tokens: Reserved tokens always present at fixed IDs.
        save_dir: Where to write the trained tokenizer files.
    """

    vocab_size: int = 8000
    min_frequency: int = 2
    special_tokens: List[str] = field(
        default_factory=lambda: ["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    save_dir: str = "tokenizer/trained"
