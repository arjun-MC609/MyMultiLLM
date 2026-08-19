"""Command-line text generation: prompt in, generated text out."""

import argparse
import logging

import torch

from inference.generate import generate, load_model_for_inference
from tokenizer.train_tokenizer import load_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer = load_tokenizer(args.tokenizer_dir)
    model = load_model_for_inference(args.checkpoint, device=device)

    eos_id = tokenizer.token_to_id("<eos>")
    prompt_ids = tokenizer.encode(args.prompt).ids

    logger.info("Prompt: %r (%d tokens)", args.prompt, len(prompt_ids))

    output_ids = generate(
        model=model, prompt_ids=prompt_ids, max_new_tokens=args.max_new_tokens,
        eos_id=eos_id, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, greedy=args.greedy, device=device,
    )

    generated_text = tokenizer.decode(output_ids)
    print("\n=== GENERATED TEXT ===")
    print(generated_text)
    print("======================\n")


if __name__ == "__main__":
    main()
