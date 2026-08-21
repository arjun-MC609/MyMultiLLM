"""Autoregressive text generation from a trained TinyTransformerLM, with KV-caching."""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

from models.config import ModelConfig
from models.model import TinyTransformerLM
from training.train import load_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_model_for_inference(checkpoint_path: str, device: Optional[torch.device] = None) -> TinyTransformerLM:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**raw_checkpoint["model_config"])

    model = TinyTransformerLM(model_config).to(device)
    load_checkpoint(checkpoint_path, model, optimizer=None, device=device)

    model.eval()
    return model


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    top_values, _ = torch.topk(logits, k)
    min_kept_value = top_values[..., -1, None]
    return torch.where(logits < min_kept_value, torch.full_like(logits, float("-inf")), logits)


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs - sorted_probs > p
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    out = torch.full_like(logits, float("-inf"))
    out.scatter_(-1, sorted_indices, sorted_logits)
    return out


def _sample(next_token_logits, temperature, top_k, top_p, greedy) -> int:
    if greedy:
        return int(torch.argmax(next_token_logits).item())

    scaled_logits = next_token_logits / max(temperature, 1e-6)
    scaled_logits = _apply_top_k(scaled_logits, top_k)
    scaled_logits = _apply_top_p(scaled_logits, top_p)
    probs = F.softmax(scaled_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: TinyTransformerLM,
    prompt_ids: List[int],
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    greedy: bool = False,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
) -> List[int]:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty.")

    if device is None:
        device = next(model.parameters()).device

    max_seq_len = model.config.max_seq_len
    generated = list(prompt_ids)

    if not use_cache:
        for _ in range(max_new_tokens):
            context = generated[-max_seq_len:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            next_token = _sample(next_token_logits, temperature, top_k, top_p, greedy)
            generated.append(next_token)
            if eos_id is not None and next_token == eos_id:
                break
        return generated

    prompt_tensor = torch.tensor([generated[-max_seq_len:]], dtype=torch.long, device=device)
    logits, kv_cache = model(prompt_tensor, use_cache=True)
    next_token_logits = logits[0, -1, :]

    for _ in range(max_new_tokens):
        next_token = _sample(next_token_logits, temperature, top_k, top_p, greedy)
        generated.append(next_token)

        if eos_id is not None and next_token == eos_id:
            break

        if len(generated) - 1 >= max_seq_len:
            break

        next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, kv_cache = model(next_input, past_kv_list=kv_cache, use_cache=True)
        next_token_logits = logits[0, -1, :]

    return generated
