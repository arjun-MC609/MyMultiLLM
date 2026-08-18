"""Token + positional embeddings."""

import torch
import torch.nn as nn

from models.config import ModelConfig


class Embeddings(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.max_seq_len = config.max_seq_len

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        tok_vecs = self.token_emb(token_ids)
        pos_vecs = self.pos_emb(positions)
        return self.dropout(tok_vecs + pos_vecs)
