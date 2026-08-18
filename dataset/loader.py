"""Streaming dataset loader for tokenized shards."""

import json
import logging
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class ShardedTokenDataset(IterableDataset):
    def __init__(self, manifest_path: str, seq_len: int) -> None:
        if not Path(manifest_path).is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.shard_paths: List[str] = json.load(f)

        if not self.shard_paths:
            raise ValueError(f"Manifest {manifest_path} lists no shards.")

        self.seq_len = seq_len

    def __iter__(self) -> Iterator[tuple]:
        window = self.seq_len + 1

        for shard_path in self.shard_paths:
            tokens = np.memmap(shard_path, dtype=np.uint16, mode="r")

            n_windows = len(tokens) // window
            if n_windows == 0:
                logger.warning(
                    "Shard %s has only %d tokens, smaller than window size %d — skipping.",
                    shard_path, len(tokens), window,
                )
                continue

            for i in range(n_windows):
                chunk = tokens[i * window : (i + 1) * window]
                chunk = np.array(chunk, dtype=np.int64)
                input_ids = torch.from_numpy(chunk[:-1])
                target_ids = torch.from_numpy(chunk[1:])
                yield input_ids, target_ids

    def estimated_length(self) -> int:
        total = 0
        window = self.seq_len + 1
        for shard_path in self.shard_paths:
            n_tokens = Path(shard_path).stat().st_size // 2
            total += n_tokens // window
        return total
