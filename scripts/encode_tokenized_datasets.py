#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def _count_tokens_for_file(tokenizer: Tokenizer, input_path: Path) -> int:
    count = 0
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in tokenizer.encode_iterable(f):
            count += 1
    return count


def _encode_file_to_uint16_npy(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> dict[str, int]:
    """Two-pass encoding:
    1) count tokens to know array size
    2) write token ids into a uint16 .npy memmap
    """
    token_count = _count_tokens_for_file(tokenizer, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.lib.format.open_memmap(
        str(output_path),
        mode="w+",
        dtype=np.uint16,
        shape=(token_count,),
    )

    i = 0
    max_id = -1
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for token_id in tokenizer.encode_iterable(f):
            if token_id > np.iinfo(np.uint16).max:
                raise ValueError(
                    f"Token id {token_id} exceeds uint16 max; uint16 only supports up to 65535."
                )
            arr[i] = token_id
            if token_id > max_id:
                max_id = token_id
            i += 1

    # Ensure data is flushed to disk.
    del arr

    return {
        "token_count": token_count,
        "max_token_id": max_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode TinyStories/OWT train+dev into uint16 token arrays.")

    parser.add_argument("--special-token", default="<|endoftext|>")

    parser.add_argument("--tiny-vocab", type=Path, default=Path("experiments/tinystories-bpe-10k-vocab.json"))
    parser.add_argument("--tiny-merges", type=Path, default=Path("experiments/tinystories-bpe-10k-merges.txt"))
    parser.add_argument("--owt-vocab", type=Path, default=Path("experiments/owt-bpe-32k-vocab.json"))
    parser.add_argument("--owt-merges", type=Path, default=Path("experiments/owt-bpe-32k-merges.txt"))

    parser.add_argument("--tiny-train", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--tiny-dev", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--owt-train", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--owt-dev", type=Path, default=Path("data/owt_valid.txt"))

    parser.add_argument("--out-dir", type=Path, default=Path("experiments/tokenized_datasets"))
    parser.add_argument("--metadata-out", type=Path, default=Path("experiments/tokenized_datasets/metadata.json"))
    args = parser.parse_args()

    tiny_tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.tiny_vocab),
        merges_filepath=str(args.tiny_merges),
        special_tokens=[args.special_token],
    )
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.owt_vocab),
        merges_filepath=str(args.owt_merges),
        special_tokens=[args.special_token],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plan = [
        ("tinystories_train", tiny_tokenizer, args.tiny_train, args.out_dir / "tinystories-train.uint16.npy"),
        ("tinystories_dev", tiny_tokenizer, args.tiny_dev, args.out_dir / "tinystories-dev.uint16.npy"),
        ("owt_train", owt_tokenizer, args.owt_train, args.out_dir / "owt-train.uint16.npy"),
        ("owt_dev", owt_tokenizer, args.owt_dev, args.out_dir / "owt-dev.uint16.npy"),
    ]

    metadata: dict[str, dict[str, int | str]] = {}
    for name, tokenizer, src, dst in plan:
        stats = _encode_file_to_uint16_npy(tokenizer=tokenizer, input_path=src, output_path=dst)
        metadata[name] = {
            "input_path": str(src),
            "output_path": str(dst),
            "dtype": "uint16",
            **stats,
        }
        print(f"{name}: tokens={stats['token_count']} max_id={stats['max_token_id']} -> {dst}")

    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote_metadata={args.metadata_out}")


if __name__ == "__main__":
    main()
