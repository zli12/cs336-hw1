# TinyStories BPE Run Log

## Run Summary

- **Task:** Train byte-level BPE on TinyStories
- **Script:** `scripts/run_tinystories_bpe.py`
- **Input:** `data/TinyStoriesV2-GPT4-train.txt`
- **Special token:** `<|endoftext|>`
- **Target vocab size:** `10000`
- **Process count:** `32` (max workers from `nproc`)
- **Command:**
  - `uv run python scripts/run_tinystories_bpe.py --vocab-size 10000 --num-processes 32 --out-vocab experiments/tinystories-bpe-10k-vocab.json --out-merges experiments/tinystories-bpe-10k-merges.txt`

## Completion Status

- **Exit code:** `0` (success)
- **Elapsed time:** `62.644 s` (`0.017401 h`)
- **Peak memory (RSS sum):** `1.711 GiB`
- **Peak memory (USS sum):** `1.275 GiB`
- **Result sizes:** `vocab_size=10000`, `merges=9743`

## Longest Token

- **Longest token length:** `15` bytes
- **Longest token preview:**
  - `b' accomplishment'`
- **Interpretation:** reasonable; frequent leading-space wordpiece from natural English text.

## Output Artifacts

- `experiments/tinystories-bpe-10k-vocab.json`
- `experiments/tinystories-bpe-10k-merges.txt`
