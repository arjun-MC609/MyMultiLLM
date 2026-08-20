"""End-to-end multi-model entry point: route, load, generate."""

import logging
from typing import Optional

import torch

from router.router import Router
from specialists.loader import load_specialist
from specialists.registry import SpecialistRegistry
from inference.generate import generate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def ask(query, registry, router=None, max_new_tokens=50, temperature=0.8, top_k=40, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if router is None:
        router = Router(registry)

    specialist_name = router.route(query)
    model, tokenizer = load_specialist(registry, specialist_name, device=device)

    prompt_ids = tokenizer.encode(query).ids
    eos_id = tokenizer.token_to_id("<eos>")

    output_ids = generate(
        model=model, prompt_ids=prompt_ids, max_new_tokens=max_new_tokens,
        eos_id=eos_id, temperature=temperature, top_k=top_k, device=device,
    )

    response = tokenizer.decode(output_ids)
    logger.info("[%s specialist] %r -> %r", specialist_name, query, response)
    return response
