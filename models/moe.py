"""Mixture-of-Experts feed-forward layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig
from models.feedforward import FeedForward


class MoEFeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.moe_top_k > config.num_experts:
            raise ValueError(
                f"moe_top_k ({config.moe_top_k}) cannot exceed num_experts ({config.num_experts})"
            )

        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.num_experts)])
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)

        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_expert_usage: torch.Tensor = torch.zeros(config.num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        n_tokens = x_flat.shape[0]

        router_logits = self.gate(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)

        topk_weights, topk_idx = routing_probs.topk(self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            slot_mask = topk_idx == expert_id
            if not slot_mask.any():
                continue

            token_rows, slot_cols = slot_mask.nonzero(as_tuple=True)
            expert_input = x_flat[token_rows]
            expert_output = self.experts[expert_id](expert_input)

            weights = topk_weights[token_rows, slot_cols].unsqueeze(-1)
            output.index_add_(0, token_rows, expert_output * weights)

        top1_idx = topk_idx[:, 0]
        f = torch.zeros(self.num_experts, device=x.device)
        for expert_id in range(self.num_experts):
            f[expert_id] = (top1_idx == expert_id).float().mean()
        p = routing_probs.mean(dim=0)

        self.last_aux_loss = self.num_experts * torch.sum(f * p)
        self.last_expert_usage = f.detach()

        return output.view(batch_size, seq_len, d_model)
