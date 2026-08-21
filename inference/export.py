"""Quantize and export a trained model for local release."""

import json
import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from models.config import ModelConfig
from models.model import TinyTransformerLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def quantize_model(model: TinyTransformerLM) -> nn.Module:
    model = model.to("cpu")
    model.eval()
    quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    return quantized


def export_model(checkpoint_path, tokenizer_dir, output_dir, quantize=True):
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not Path(tokenizer_dir).is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])

    model = TinyTransformerLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    original_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    if quantize:
        logger.info("Applying dynamic int8 quantization...")
        model = quantize_model(model)

    model_path = output_path / "model.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "model_config": model_config.__dict__, "quantized": quantize},
        model_path,
    )

    exported_size_mb = model_path.stat().st_size / (1024 ** 2)

    tokenizer_out = output_path / "tokenizer"
    tokenizer_out.mkdir(exist_ok=True)
    for fname in ("vocab.json", "merges.txt"):
        src = Path(tokenizer_dir) / fname
        if src.is_file():
            shutil.copy(src, tokenizer_out / fname)

    manifest = {
        "model_config": model_config.__dict__,
        "quantized": quantize,
        "original_size_mb": round(original_size_mb, 2),
        "exported_size_mb": round(exported_size_mb, 2),
    }
    with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Exported to %s (%.1f MB -> %.1f MB, quantized=%s)",
        output_path, original_size_mb, exported_size_mb, quantize,
    )
    return str(output_path)


def load_exported_model(export_dir):
    model_path = Path(export_dir) / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"No model.pt found in {export_dir}")

    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**bundle["model_config"])

    model = TinyTransformerLM(model_config)

    if bundle.get("quantized", False):
        model = quantize_model(model)

    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    return model, model_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a trained model for local release.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    export_model(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.output_dir,
        quantize=not args.no_quantize,
    )
