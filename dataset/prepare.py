"""Prepare raw text into tokenized, packed, sharded training data."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Set

import numpy as np

from dataset.config import DatasetConfig
from tokenizer.train_tokenizer import load_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _load_state(state_file: str) -> dict:
    if os.path.isfile(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_files": [], "next_shard_id": 0}


def _save_state(state_file: str, state: dict) -> None:
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _is_valid_line(line: str, min_length: int) -> bool:
    stripped = line.strip()
    return len(stripped) >= min_length


def _line_hash(line: str) -> str:
    return hashlib.md5(line.strip().encode("utf-8")).hexdigest()


def prepare_dataset(config: DatasetConfig) -> None:
    raw_files = sorted(Path(config.raw_dir).glob("*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"No .txt files found in {config.raw_dir}")

    tokenizer = load_tokenizer(config.tokenizer_dir)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer has no <eos> special token — was it trained correctly?")

    state = _load_state(config.state_file)
    processed: Set[str] = set(state["processed_files"])
    shard_id = state["next_shard_id"]

    Path(config.shards_dir).mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    token_buffer: List[int] = []
    total_tokens_written = 0
    shard_paths: List[str] = []

    def flush_shard(buffer: List[int]) -> None:
        nonlocal shard_id, total_tokens_written
        if not buffer:
            return
        arr = np.array(buffer, dtype=np.uint16)
        shard_path = os.path.join(config.shards_dir, f"shard_{shard_id:05d}.bin")
        arr.tofile(shard_path)
        shard_paths.append(shard_path)
        logger.info("Wrote %s (%d tokens)", shard_path, len(buffer))
        total_tokens_written += len(buffer)
        shard_id += 1

    for raw_file in raw_files:
        fname = str(raw_file)
        if fname in processed:
            logger.info("Skipping already-processed file: %s", fname)
            continue

        logger.info("Processing %s", fname)
        n_lines_kept, n_lines_dropped_dup, n_lines_dropped_quality = 0, 0, 0

        with open(raw_file, "r", encoding="utf-8") as f:
            for line in f:
                if not _is_valid_line(line, config.min_line_length):
                    n_lines_dropped_quality += 1
                    continue

                h = _line_hash(line)
                if h in seen_hashes:
                    n_lines_dropped_dup += 1
                    continue
                seen_hashes.add(h)

                ids = tokenizer.encode(line.strip()).ids
                token_buffer.extend(ids)
                token_buffer.append(eos_id)
                n_lines_kept += 1

                while len(token_buffer) >= config.tokens_per_shard:
                    flush_shard(token_buffer[: config.tokens_per_shard])
                    token_buffer = token_buffer[config.tokens_per_shard :]

        logger.info(
            "%s: kept=%d dropped_duplicate=%d dropped_quality=%d",
            fname, n_lines_kept, n_lines_dropped_dup, n_lines_dropped_quality,
        )

        processed.add(fname)
        state["processed_files"] = list(processed)
        state["next_shard_id"] = shard_id
        _save_state(config.state_file, state)

    flush_shard(token_buffer)
    state["next_shard_id"] = shard_id
    _save_state(config.state_file, state)

    logger.info("Done. Total tokens written this run: %d across %d shard(s).",
                total_tokens_written, len(shard_paths))
    _write_train_val_split(config)


def _write_train_val_split(config: DatasetConfig) -> None:
    all_shards = sorted(Path(config.shards_dir).glob("shard_*.bin"))
    if not all_shards:
        logger.warning("No shards found to split.")
        return

    n_val = max(1, int(len(all_shards) * config.val_fraction)) if len(all_shards) > 1 else 0
    val_shards = all_shards[-n_val:] if n_val > 0 else []
    train_shards = all_shards[: len(all_shards) - n_val] if n_val > 0 else all_shards

    manifest_dir = Path(config.shards_dir)
    with open(manifest_dir / "train_shards.json", "w", encoding="utf-8") as f:
        json.dump([str(p) for p in train_shards], f, indent=2)
    with open(manifest_dir / "val_shards.json", "w", encoding="utf-8") as f:
        json.dump([str(p) for p in val_shards], f, indent=2)

    logger.info("Split: %d train shard(s), %d val shard(s)", len(train_shards), len(val_shards))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare raw text into tokenized shards.")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--shards-dir", type=str, default="data/shards")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer/trained_test")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--tokens-per-shard", type=int, default=500_000)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    cfg = DatasetConfig(
        raw_dir=args.raw_dir,
        shards_dir=args.shards_dir,
        tokenizer_dir=args.tokenizer_dir,
        seq_len=args.seq_len,
        tokens_per_shard=args.tokens_per_shard,
        val_fraction=args.val_fraction,
        state_file=str(Path(args.shards_dir) / "_prepare_state.json"),
    )
    prepare_dataset(cfg)
