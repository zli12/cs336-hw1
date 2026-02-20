# Run TinyStories BPE Training

This file captures reproducible steps for **Problem `train_bpe_tinystories`**.

## 1) Train a 10k byte-level BPE tokenizer and save artifacts

Runs multiprocessing pretokenization via `BPETrainerMulti`, then writes:
- `data/tinystories-bpe-10k-vocab.json`
- `data/tinystories-bpe-10k-merges.txt`

```bash
uv run python - <<'PY'
import json
import os
import threading
import time
from pathlib import Path

import psutil
from cs336_basics.bpe_train_multi import BPETrainerMulti

train_path = Path("data/TinyStoriesV2-GPT4-train.txt")
out_vocab = Path("data/tinystories-bpe-10k-vocab.json")
out_merges = Path("data/tinystories-bpe-10k-merges.txt")

process = psutil.Process(os.getpid())
peak_rss_sum = 0
peak_uss_sum = 0
running = True

def sample_memory():
    global peak_rss_sum, peak_uss_sum
    while running:
        rss_sum = 0
        uss_sum = 0
        procs = [process] + process.children(recursive=True)
        for p in procs:
            try:
                rss_sum += p.memory_info().rss
                full = p.memory_full_info()
                uss_sum += getattr(full, "uss", 0)
            except psutil.Error:
                pass
        peak_rss_sum = max(peak_rss_sum, rss_sum)
        peak_uss_sum = max(peak_uss_sum, uss_sum)
        time.sleep(0.05)

sampler = threading.Thread(target=sample_memory, daemon=True)
sampler.start()

num_processes = min(8, os.cpu_count() or 1)
start = time.perf_counter()
vocab, merges = BPETrainerMulti.train(
    input_path=train_path,
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
    num_processes=num_processes,
)
elapsed = time.perf_counter() - start

running = False
sampler.join(timeout=1)

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))

encoder = bytes_to_unicode()
def encode_token_bytes(token_bytes: bytes) -> str:
    return "".join(encoder[b] for b in token_bytes)

vocab_json = {encode_token_bytes(token): idx for idx, token in vocab.items()}
out_vocab.write_text(json.dumps(vocab_json, ensure_ascii=False, indent=2), encoding="utf-8")
out_merges.write_text(
    "\n".join(f"{encode_token_bytes(a)} {encode_token_bytes(b)}" for a, b in merges) + "\n",
    encoding="utf-8",
)

longest_token = max(vocab.values(), key=len)
print(f"num_processes={num_processes}")
print(f"elapsed_seconds={elapsed:.3f}")
print(f"elapsed_hours={elapsed/3600:.6f}")
print(f"peak_rss_sum_gib={peak_rss_sum/(1024**3):.3f}")  # includes shared memory (overestimates)
print(f"peak_uss_sum_gib={peak_uss_sum/(1024**3):.3f}")  # unique memory (better estimate)
print(f"vocab_size={len(vocab)} merges={len(merges)}")
print(f"longest_token_len_bytes={len(longest_token)}")
print(f"longest_token_preview={longest_token!r}")
print(f"wrote_vocab={out_vocab}")
print(f"wrote_merges={out_merges}")
PY
```

## 2) Optional quick benchmark by process count

```bash
uv run python - <<'PY'
import time
from pathlib import Path
from cs336_basics.bpe_train_multi import BPETrainerMulti

input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
for p in (1, 2, 4, 8):
    t0 = time.perf_counter()
    vocab, merges = BPETrainerMulti.train(
        input_path=input_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        num_processes=p,
    )
    dt = time.perf_counter() - t0
    print(f"num_processes={p}: {dt:.3f}s, vocab={len(vocab)}, merges={len(merges)}")
PY
```

## 3) One-line checks for serialized artifacts

```bash
uv run python - <<'PY'
import json
from pathlib import Path

vocab = json.loads(Path("data/tinystories-bpe-10k-vocab.json").read_text(encoding="utf-8"))
merges = Path("data/tinystories-bpe-10k-merges.txt").read_text(encoding="utf-8").splitlines()
print("vocab_entries:", len(vocab))
print("merge_lines:", len([m for m in merges if m.strip()]))
PY
```

