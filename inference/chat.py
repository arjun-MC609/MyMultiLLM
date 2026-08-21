"""Interactive local chat: the final release interface."""

import argparse
import logging

import torch

from inference.generate import generate, load_model_for_inference
from tokenizer.train_tokenizer import load_tokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def _print_banner(mode_desc):
    print("=" * 60)
    print("  MyMultiLLM -- Local Chat")
    print(f"  {mode_desc}")
    print("  Type your message and press Enter. Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print()


def chat_with_model(model, tokenizer, device, label="model"):
    eos_id = tokenizer.token_to_id("<eos>")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if not user_input:
            continue

        prompt_ids = tokenizer.encode(user_input).ids
        output_ids = generate(
            model=model, prompt_ids=prompt_ids, max_new_tokens=60,
            eos_id=eos_id, temperature=0.8, top_k=40, device=device, use_cache=True,
        )
        response_ids = output_ids[len(prompt_ids):]
        response = tokenizer.decode(response_ids)
        print(f"{label}: {response}\n")


def chat_with_router(registry_path, device):
    from router.router import Router
    from router.pipeline import ask
    from specialists.registry import SpecialistRegistry

    registry = SpecialistRegistry(registry_path)
    if not registry.list_specialists():
        raise RuntimeError(
            f"No specialists registered in {registry_path}. Register at least one first."
        )

    router = Router(registry)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if not user_input:
            continue

        specialist_name = router.route(user_input)
        response = ask(user_input, registry, router=router, max_new_tokens=60, device=device)
        print(f"[{specialist_name}]: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive local chat with a trained model or specialist.")
    parser.add_argument("--registry", type=str, default=None)
    parser.add_argument("--specialist", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tokenizer-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        if not args.tokenizer_dir:
            parser.error("--tokenizer-dir is required when using --checkpoint")
        tokenizer = load_tokenizer(args.tokenizer_dir)
        model = load_model_for_inference(args.checkpoint, device=device)
        _print_banner(f"Direct checkpoint: {args.checkpoint}")
        chat_with_model(model, tokenizer, device, label="model")

    elif args.registry and args.specialist:
        from specialists.loader import load_specialist
        from specialists.registry import SpecialistRegistry

        registry = SpecialistRegistry(args.registry)
        model, tokenizer = load_specialist(registry, args.specialist, device=device)
        _print_banner(f"Specialist: {args.specialist}")
        chat_with_model(model, tokenizer, device, label=args.specialist)

    elif args.registry:
        _print_banner("Auto-routing across all registered specialists")
        chat_with_router(args.registry, device)

    else:
        parser.error("Provide either --checkpoint + --tokenizer-dir, or --registry (optionally with --specialist).")


if __name__ == "__main__":
    main()
