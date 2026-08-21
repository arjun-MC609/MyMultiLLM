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

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        normed = self.ln1(x)
        if use_cache:
            attn_out, present_kv = self.attn(normed, past_kv=past_kv, use_cache=True)
        else:
            attn_out = self.attn(normed, past_kv=past_kv, use_cache=False)
            present_kv = None

        x = x + attn_out
        x = x + self.ffn(self.ln2(x))

        if use_cache:
            return x, present_kv
        return x
