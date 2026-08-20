"""The full tiny transformer language model."""

import torch
import torch.nn as nn

from models.config import ModelConfig
from models.embeddings import Embeddings
from models.transformer_block import TransformerBlock


class TinyTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.embeddings.token_emb.weight

        self.apply(self._init_weights)
        self.last_aux_loss: torch.Tensor = torch.tensor(0.0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(token_ids)

        total_aux_loss = torch.tensor(0.0, device=token_ids.device)
        for block in self.blocks:
            x = block(x)
            if self.config.use_moe:
                total_aux_loss = total_aux_loss + block.ffn.last_aux_loss
        self.last_aux_loss = total_aux_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def num_parameters(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embeddings.pos_emb.weight.numel()
        return n_params
