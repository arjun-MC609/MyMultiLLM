"""Multi-head causal self-attention with optional KV-caching."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", causal_mask.bool())

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        batch_size, new_seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, new_seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, new_seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, new_seq_len, self.n_heads, self.d_head).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.shape[2]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        total_len = past_len + new_seq_len

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        query_positions = torch.arange(past_len, total_len, device=x.device).unsqueeze(1)
        key_positions = torch.arange(0, total_len, device=x.device).unsqueeze(0)
        mask = key_positions <= query_positions

        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, new_seq_len, d_model)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        if use_cache:
            return out, present_kv
        return out
