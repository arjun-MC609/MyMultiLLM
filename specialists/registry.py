"""Registry of specialist models."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

from models.config import ModelConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass
class SpecialistEntry:
    name: str
    description: str
    model_config: dict
    tokenizer_dir: str
    checkpoint_path: str
    trained_steps: int = 0
    notes: str = ""


class SpecialistRegistry:
    def __init__(self, registry_path: str) -> None:
        self.registry_path = registry_path
        self._entries: Dict[str, SpecialistEntry] = {}
        self._load()

    def _load(self) -> None:
        if Path(self.registry_path).is_file():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._entries = {name: SpecialistEntry(**entry) for name, entry in raw.items()}
            logger.info("Loaded registry from %s (%d specialists)", self.registry_path, len(self._entries))
        else:
            logger.info("No existing registry at %s -- starting empty.", self.registry_path)

    def _save(self) -> None:
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump({name: asdict(entry) for name, entry in self._entries.items()}, f, indent=2)
        logger.info("Saved registry to %s", self.registry_path)

    def register(self, name, description, model_config, tokenizer_dir, checkpoint_path, trained_steps=0, notes=""):
        entry = SpecialistEntry(
            name=name, description=description, model_config=model_config.__dict__,
            tokenizer_dir=tokenizer_dir, checkpoint_path=checkpoint_path,
            trained_steps=trained_steps, notes=notes,
        )
        self._entries[name] = entry
        self._save()
        logger.info("Registered specialist '%s' -> %s", name, checkpoint_path)

    def get(self, name: str) -> SpecialistEntry:
        if name not in self._entries:
            available = list(self._entries.keys())
            raise KeyError(f"No specialist named '{name}'. Available: {available}")
        return self._entries[name]

    def list_specialists(self) -> Dict[str, SpecialistEntry]:
        return dict(self._entries)

    def get_model_config(self, name: str) -> ModelConfig:
        entry = self.get(name)
        return ModelConfig(**entry.model_config)
