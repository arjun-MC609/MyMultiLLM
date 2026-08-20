"""Convenience loader: registry entry -> ready-to-use model + tokenizer."""

import logging
from typing import Optional, Tuple

import torch

from inference.generate import load_model_for_inference
from models.model import TinyTransformerLM
from specialists.registry import SpecialistRegistry
from tokenizer.train_tokenizer import load_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_specialist(registry: SpecialistRegistry, name: str, device: Optional[torch.device] = None):
    entry = registry.get(name)
    logger.info("Loading specialist '%s': %s", name, entry.description)
    tokenizer = load_tokenizer(entry.tokenizer_dir)
    model = load_model_for_inference(entry.checkpoint_path, device=device)
    return model, tokenizer
