from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.decoding import decode
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM checkpoint.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to a model checkpoint.")
    parser.add_argument("--vocab-path", type=Path, required=True, help="Path to tokenizer vocab JSON.")
    parser.add_argument("--merges-path", type=Path, required=True, help="Path to tokenizer merges TXT.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text to continue.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--eos-token", type=str, default="<|endoftext|>")

    # Model hyperparameters must match the checkpoint architecture.
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> TransformerLM:
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
    )
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def main() -> None:
    args = parse_args()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.vocab_path),
        merges_filepath=str(args.merges_path),
        special_tokens=[args.eos_token],
    )
    model = load_model(args)

    prompt_ids = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.special_token_to_id.get(args.eos_token)
    generated_ids = decode(
        model=model,
        prompt=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )

    generated_text = tokenizer.decode(generated_ids)
    print(generated_text)


if __name__ == "__main__":
    main()
