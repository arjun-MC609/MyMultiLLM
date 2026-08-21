"""Named model sizes for reproducible training launches."""

from models.config import ModelConfig


def gpt2_small() -> ModelConfig:
    """The original GPT-2 small geometry: approximately 124M parameters."""
    return ModelConfig(
        vocab_size=50_257, d_model=768, n_layers=12, n_heads=12,
        d_ff=3072, max_seq_len=1024, dropout=0.1,
    )


def dense_4b() -> ModelConfig:
    """A dense ~4B-parameter model; use FSDP/ZeRO, not a single GPU."""
    return ModelConfig(
        vocab_size=50_000, d_model=4096, n_layers=28, n_heads=32,
        d_ff=8192, max_seq_len=2048, dropout=0.0,
        gradient_checkpointing=True,
    )


PROFILES = {"gpt2-small": gpt2_small, "dense-4b": dense_4b}


def get_model_profile(name: str) -> ModelConfig:
    try:
        return PROFILES[name]()
    except KeyError as error:
        raise ValueError(f"Unknown model profile {name!r}. Choose one of: {', '.join(PROFILES)}") from error
