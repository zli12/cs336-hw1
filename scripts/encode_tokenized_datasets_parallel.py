"""Parallel BPE tokenization of TinyStories/OWT into uint16 .npy arrays.

Strategy:
  1) Find chunk boundaries snapped to the next ``<|endoftext|>`` so no chunk
     ever splits a special token.
  2) Spawn ``--num-workers`` processes; each tokenizes its byte slice with the
     standard ``cs336_basics.tokenizer.Tokenizer`` and writes a partial
     uint16 array to ``<out>.parts/chunk_NNN.npy``.
  3) Concatenate all parts into the final ``<out>.uint16.npy`` (memmap copy)
     and remove the parts directory.

Each (vocab, merges, special-token) bundle is loaded once per worker.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Snap each evenly-spaced boundary forward to the next sentinel-token start."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1:
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size

    mini = 4096
    for i in range(1, len(boundaries) - 1):
        pos = boundaries[i]
        file.seek(pos)
        while True:
            buf = file.read(mini)
            if buf == b"":
                boundaries[i] = file_size
                break
            found = buf.find(split_special_token)
            if found != -1:
                boundaries[i] = pos + found
                break
            pos += mini

    return sorted(set(boundaries))


# --- worker ------------------------------------------------------------------

# Loaded once per worker process via the initializer; reused across all chunks
# assigned to that worker.
_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_INPUT_PATH: str | None = None
_WORKER_PARTS_DIR: str | None = None


def _worker_init(vocab_path: str, merges_path: str, special_token: str, input_path: str, parts_dir: str) -> None:
    global _WORKER_TOKENIZER, _WORKER_INPUT_PATH, _WORKER_PARTS_DIR
    _WORKER_TOKENIZER = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=[special_token],
    )
    _WORKER_INPUT_PATH = input_path
    _WORKER_PARTS_DIR = parts_dir


def _worker_encode_chunk(args: tuple[int, int, int]) -> tuple[int, int, str]:
    """Encode ``input_path[start:end]`` and dump tokens to ``parts_dir/chunk_NNN.npy``."""
    chunk_idx, start, end = args
    assert _WORKER_TOKENIZER is not None
    assert _WORKER_INPUT_PATH is not None
    assert _WORKER_PARTS_DIR is not None

    with open(_WORKER_INPUT_PATH, "rb") as f:
        f.seek(start)
        buf = f.read(end - start)

    text = buf.decode("utf-8", errors="ignore")
    ids = _WORKER_TOKENIZER.encode(text)

    arr = np.asarray(ids, dtype=np.uint16)
    out_path = Path(_WORKER_PARTS_DIR) / f"chunk_{chunk_idx:06d}.npy"
    np.save(out_path, arr, allow_pickle=False)
    return chunk_idx, len(arr), str(out_path)


def _encode_file_parallel(
    *,
    name: str,
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_token: str,
    num_workers: int,
    chunk_factor: int = 4,
) -> dict[str, int]:
    """Tokenize ``input_path`` in parallel and write the concatenated uint16 array to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts_dir = output_path.with_suffix(".parts")
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=False)

    sentinel_bytes = special_token.encode("utf-8")
    desired_chunks = max(1, num_workers * chunk_factor)
    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(
            file=f,
            desired_num_chunks=desired_chunks,
            split_special_token=sentinel_bytes,
        )

    tasks: list[tuple[int, int, int]] = []
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if start >= end:
            continue
        tasks.append((i, start, end))

    print(
        f"[{name}] {input_path.name}: file_bytes={boundaries[-1]:,} "
        f"chunks={len(tasks)} workers={num_workers}"
    )

    t0 = time.time()
    chunk_lens: dict[int, int] = {}
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(str(vocab_path), str(merges_path), special_token, str(input_path), str(parts_dir)),
    ) as exe:
        futures = [exe.submit(_worker_encode_chunk, t) for t in tasks]
        completed = 0
        for fut in as_completed(futures):
            chunk_idx, n_tokens, _ = fut.result()
            chunk_lens[chunk_idx] = n_tokens
            completed += 1
            if completed % max(1, len(tasks) // 20) == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                done_tokens = sum(chunk_lens.values())
                rate = done_tokens / max(1e-9, elapsed)
                print(
                    f"[{name}] progress {completed}/{len(tasks)} "
                    f"tokens={done_tokens:,} rate={rate / 1e6:.2f}M tok/s "
                    f"elapsed={elapsed:.1f}s"
                )

    total_tokens = sum(chunk_lens.values())

    print(f"[{name}] concatenating {len(chunk_lens)} parts into {output_path} ({total_tokens:,} tokens)...")
    out_arr = np.lib.format.open_memmap(str(output_path), mode="w+", dtype=np.uint16, shape=(total_tokens,))

    cursor = 0
    max_id = -1
    for chunk_idx in sorted(chunk_lens):
        part_path = parts_dir / f"chunk_{chunk_idx:06d}.npy"
        part = np.load(part_path, mmap_mode="r")
        n = part.shape[0]
        out_arr[cursor : cursor + n] = part
        local_max = int(part.max()) if n > 0 else -1
        if local_max > max_id:
            max_id = local_max
        cursor += n
        del part

    del out_arr
    shutil.rmtree(parts_dir)

    elapsed = time.time() - t0
    print(
        f"[{name}] DONE tokens={total_tokens:,} max_id={max_id} "
        f"avg_rate={total_tokens / max(1e-9, elapsed) / 1e6:.2f}M tok/s "
        f"total={elapsed:.1f}s -> {output_path}"
    )

    return {
        "token_count": total_tokens,
        "max_token_id": max_id,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel BPE-encode TinyStories/OWT to uint16 npy.")
    parser.add_argument("--special-token", default="<|endoftext|>")

    parser.add_argument("--tiny-vocab", type=Path, default=Path("experiments/tinystories-bpe-10k-vocab.json"))
    parser.add_argument("--tiny-merges", type=Path, default=Path("experiments/tinystories-bpe-10k-merges.txt"))
    parser.add_argument("--owt-vocab", type=Path, default=Path("experiments/owt-bpe-32k-vocab.json"))
    parser.add_argument("--owt-merges", type=Path, default=Path("experiments/owt-bpe-32k-merges.txt"))

    parser.add_argument("--tiny-train", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--tiny-dev", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--owt-train", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--owt-dev", type=Path, default=Path("data/owt_valid.txt"))

    parser.add_argument("--out-dir", type=Path, default=Path("data/tokenized_datasets"))
    parser.add_argument("--metadata-out", type=Path, default=Path("data/tokenized_datasets/metadata.json"))

    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Worker processes (default ~half of cpu_count).",
    )
    parser.add_argument(
        "--chunk-factor",
        type=int,
        default=4,
        help="Chunks per worker; larger = finer-grained progress and load balance.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=[None, "tinystories_train", "tinystories_dev", "owt_train", "owt_dev"],
        nargs="?",
        help="Only encode the given split (default: encode all four).",
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plan = [
        ("tinystories_train", args.tiny_vocab, args.tiny_merges, args.tiny_train, args.out_dir / "tinystories-train.uint16.npy"),
        ("tinystories_dev", args.tiny_vocab, args.tiny_merges, args.tiny_dev, args.out_dir / "tinystories-dev.uint16.npy"),
        ("owt_train", args.owt_vocab, args.owt_merges, args.owt_train, args.out_dir / "owt-train.uint16.npy"),
        ("owt_dev", args.owt_vocab, args.owt_merges, args.owt_dev, args.out_dir / "owt-dev.uint16.npy"),
    ]

    if args.only is not None:
        plan = [row for row in plan if row[0] == args.only]
        if not plan:
            print(f"--only={args.only} matched no plan entry", file=sys.stderr)
            sys.exit(1)

    metadata: dict[str, dict[str, object]] = {}
    for name, vocab_path, merges_path, src, dst in plan:
        if not src.exists():
            print(f"[{name}] SKIP: input {src} does not exist", file=sys.stderr)
            continue
        stats = _encode_file_parallel(
            name=name,
            input_path=src,
            output_path=dst,
            vocab_path=vocab_path,
            merges_path=merges_path,
            special_token=args.special_token,
            num_workers=args.num_workers,
            chunk_factor=args.chunk_factor,
        )
        metadata[name] = {
            "input_path": str(src),
            "output_path": str(dst),
            "dtype": "uint16",
            **stats,
        }

    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata_out.exists():
        try:
            existing = json.loads(args.metadata_out.read_text(encoding="utf-8"))
            existing.update(metadata)
            metadata = existing
        except Exception:
            pass
    args.metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote_metadata={args.metadata_out}")


if __name__ == "__main__":
    main()
