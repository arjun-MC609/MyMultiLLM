"""Configuration for the transformer model architecture."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 500
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1

    use_moe: bool = False
    num_experts: int = 4
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.use_moe and self.moe_top_k > self.num_experts:
            raise ValueError(
                f"moe_top_k ({self.moe_top_k}) cannot exceed num_experts ({self.num_experts})"
            )
