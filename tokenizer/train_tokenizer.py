"""Train and load a byte-level BPE tokenizer from scratch.

We use the `tokenizers` library (Hugging Face) purely as the fast, correct
implementation of the BPE *algorithm* — no pretrained vocabulary or merges
are loaded. Training starts from an empty vocabulary built entirely from
your own corpus.

Why byte-level BPE specifically:
- The base alphabet is the 256 possible byte values, so ANY UTF-8 text
  (English, Tamil, code, emoji, whatever) can be represented with zero
  <unk> tokens. This matters a lot once we add the Tamil expert model.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from tokenizers import ByteLevelBPETokenizer

from tokenizer.config import TokenizerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def train_tokenizer(
    files: List[str],
    config: TokenizerConfig,
) -> ByteLevelBPETokenizer:
    """Train a byte-level BPE tokenizer from raw text files.

    Args:
        files: List of paths to plain-text training files.
        config: TokenizerConfig with vocab size, special tokens, etc.

    Returns:
        The trained ByteLevelBPETokenizer instance (already fit on data).

    Raises:
        FileNotFoundError: if any input file does not exist.
        ValueError: if `files` is empty.
    """
    if not files:
        raise ValueError("No training files provided.")

    for f in files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Training file not found: {f}")

    logger.info("Initializing empty byte-level BPE tokenizer (no pretrained data).")
    tokenizer = ByteLevelBPETokenizer()

    logger.info(
        "Training on %d file(s), target vocab_size=%d, min_frequency=%d",
        len(files),
        config.vocab_size,
        config.min_frequency,
    )
    tokenizer.train(
        files=files,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
    )

    logger.info("Training complete. Final vocab size: %d", tokenizer.get_vocab_size())
    return tokenizer


def save_tokenizer(tokenizer: ByteLevelBPETokenizer, config: TokenizerConfig) -> None:
    """Save trained tokenizer files (vocab.json + merges.txt) to disk."""
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(save_path))

    with open(save_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(config.__dict__, f, indent=2)

    logger.info("Saved tokenizer to %s", save_path)


def load_tokenizer(save_dir: str) -> ByteLevelBPETokenizer:
    """Load a previously trained tokenizer from disk.

    Raises:
        FileNotFoundError: if vocab.json or merges.txt is missing.
    """
    vocab_path = os.path.join(save_dir, "vocab.json")
    merges_path = os.path.join(save_dir, "merges.txt")

    if not os.path.isfile(vocab_path) or not os.path.isfile(merges_path):
        raise FileNotFoundError(
            f"Could not find vocab.json/merges.txt in {save_dir}. "
            "Did you train and save a tokenizer first?"
        )

    tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
    logger.info("Loaded tokenizer from %s (vocab_size=%d)", save_dir, tokenizer.get_vocab_size())
    return tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer from scratch.")
    parser.add_argument("--files", nargs="+", required=True, help="Path(s) to raw text file(s).")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--save-dir", type=str, default="tokenizer/trained")
    args = parser.parse_args()

    cfg = TokenizerConfig(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_dir=args.save_dir,
    )
    tok = train_tokenizer(args.files, cfg)
    save_tokenizer(tok, cfg)
