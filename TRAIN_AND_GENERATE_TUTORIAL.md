# Train and Generate Tutorial

This guide shows how to:

1. Train a language model with `scripts/train_lm.py`
2. Generate text with `scripts/generate_lm.py`

It includes copy/paste commands and a step-by-step workflow.

## Prerequisites

- Repo dependencies installed through `uv`
- TinyStories data files present in `data/`:
  - `data/TinyStoriesV2-GPT4-train.txt`
  - `data/TinyStoriesV2-GPT4-valid.txt`
- A tokenizer vocab/merges pair (either already generated or created in Step 1 below)

Run all commands from repo root:

```bash
cd /home/coder/cs336-hw1
```

---

## Quick Start (Use Existing Checkpoint)

If you already have:

- checkpoint: `experiments/checkpoints/tinystories-base-a10g.pt`
- tokenizer files:
  - `experiments/tinystories-bpe-10k-vocab.json`
  - `experiments/tinystories-bpe-10k-merges.txt`

you can jump directly to generation:

```bash
uv run python scripts/generate_lm.py \
  --checkpoint-path experiments/checkpoints/tinystories-base-a10g.pt \
  --vocab-path experiments/tinystories-bpe-10k-vocab.json \
  --merges-path experiments/tinystories-bpe-10k-merges.txt \
  --prompt "Once upon a time there was a curious rabbit who" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95 \
  --eos-token "<|endoftext|>" \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --rope-theta 10000 \
  --device cuda
```

Important: model hyperparameters passed to `generate_lm.py` must match the architecture used during training.

---

## Full Workflow: Train Then Generate

## Step 1: Train TinyStories BPE tokenizer (if needed)

If your tokenizer files do not exist yet, generate them:

```bash
uv run python scripts/run_tinystories_bpe.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-token "<|endoftext|>" \
  --num-processes 32 \
  --out-vocab experiments/tinystories-bpe-10k-vocab.json \
  --out-merges experiments/tinystories-bpe-10k-merges.txt
```

Adjust `--num-processes` to your machine.

## Step 2: Encode TinyStories text into token arrays

Create tokenized `.npy` arrays for train/validation:

```bash
uv run python scripts/encode_tokenized_datasets.py \
  --tiny-vocab experiments/tinystories-bpe-10k-vocab.json \
  --tiny-merges experiments/tinystories-bpe-10k-merges.txt \
  --out-dir data/tokenized_datasets
```

You should get:

- `data/tokenized_datasets/tinystories-train.uint16.npy`
- `data/tokenized_datasets/tinystories-dev.uint16.npy`

## Step 3: Train a model

Example baseline training command:

```bash
uv run python scripts/train_lm.py \
  --train-data data/tokenized_datasets/tinystories-train.uint16.npy \
  --val-data data/tokenized_datasets/tinystories-dev.uint16.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --batch-size 128 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --rope-theta 10000 \
  --max-steps 10000 \
  --learning-rate 0.002 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --eps 1e-8 \
  --max-grad-norm 1.0 \
  --device cuda \
  --log-every 20 \
  --val-every 200 \
  --val-batches 20 \
  --checkpoint-path experiments/checkpoints/tinystories-run.pt \
  --checkpoint-every 200 \
  --metrics-csv experiments/logs/tinystories-run.csv
```

Artifacts created:

- checkpoint: `experiments/checkpoints/tinystories-run.pt`
- metrics: `experiments/logs/tinystories-run.csv`

Optional resume:

```bash
uv run python scripts/train_lm.py \
  ... \
  --checkpoint-path experiments/checkpoints/tinystories-run.pt \
  --resume-from experiments/checkpoints/tinystories-run.pt
```

## Step 4: Generate text from your trained checkpoint

```bash
uv run python scripts/generate_lm.py \
  --checkpoint-path experiments/checkpoints/tinystories-run.pt \
  --vocab-path experiments/tinystories-bpe-10k-vocab.json \
  --merges-path experiments/tinystories-bpe-10k-merges.txt \
  --prompt "Lily found a map in her attic and decided to" \
  --max-new-tokens 180 \
  --temperature 0.8 \
  --top-p 0.95 \
  --eos-token "<|endoftext|>" \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --rope-theta 10000 \
  --device cuda
```

---

## Prompt and Decoding Tips

- Start with `temperature=0.7-0.9` and `top-p=0.9-0.95`
- Lower temperature (`0.5-0.7`) for more deterministic outputs
- Increase `max-new-tokens` for longer stories
- If GPU is unavailable, switch to `--device cpu` (slower but works)

## Troubleshooting

- `RuntimeError` or OOM on GPU:
  - reduce `--batch-size` during training
  - use `--device cpu` for generation if needed
- Gibberish outputs:
  - verify tokenizer files match the checkpoint training run
  - ensure architecture flags passed to generation match training exactly
- No checkpoint found:
  - make sure `--checkpoint-path` in train matches the path used in generation

