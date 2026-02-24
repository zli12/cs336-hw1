# OpenWebText BPE Run Log

## Run Summary

- **Task:** Train byte-level BPE on OpenWebText
- **Script:** `scripts/run_owt_bpe.py`
- **Input:** `data/owt_train.txt`
- **Special token:** `<|endoftext|>`
- **Target vocab size:** `32000`
- **Process count:** `12`
- **Command:**
  - `uv run python scripts/run_owt_bpe.py --vocab-size 32000 --num-processes 12 --out-vocab data/owt-bpe-32k-vocab.json --out-merges data/owt-bpe-32k-merges.txt`

## Completion Status

- **Exit code:** `0` (success)
- **Elapsed time:** `8736.027 s` (`2.426674 h`)
- **Peak memory (RSS sum):** `25.648 GiB`
- **Peak memory (USS sum):** `25.646 GiB`
- **Result sizes:** `vocab_size=32000`, `merges=31743`

## Longest Token

- **Longest token length:** `64` bytes
- **Preview (raw bytes):**
  - `b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'`
- **Interpretation:** appears to be repeated mojibake-like bytes, which is plausible for noisy web text.

## Output Artifacts

- `scripts/owt-bpe-32k-vocab.json`
- `scripts/owt-bpe-32k-merges.txt`

## Notes

- Runs with higher worker counts (`32`, `12`, `8`) previously failed in this environment due to `BrokenProcessPool`; this successful run completed with the updated `bpe_train_multi.py` implementation at `12` workers.
