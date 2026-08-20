"""A single transformer block: attention + feed-forward, Pre-LN."""

import torch
import torch.nn as nn

from models.attention import MultiHeadSelfAttention
from models.config import ModelConfig
from models.feedforward import FeedForward
from models.moe import MoEFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.use_moe = config.use_moe
        self.ffn = MoEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
