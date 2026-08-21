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

    def forward(self, token_ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        if position_offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"position_offset ({position_offset}) + seq_len ({seq_len}) "
                f"exceeds max_seq_len ({self.max_seq_len})"
            )

        positions = torch.arange(
            position_offset, position_offset + seq_len, device=token_ids.device
        ).unsqueeze(0)

        tok_vecs = self.token_emb(token_ids)
        pos_vecs = self.pos_emb(positions)

        return self.dropout(tok_vecs + pos_vecs)
