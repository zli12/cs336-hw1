#!/usr/bin/env bash
# Generate text samples from the Phase-5 leaderboard final checkpoint.
#
# Mirrors the 7.4 OWT samples driver (scripts/run_owt_samples.sh) so the
# 7.5 report can do a direct fluency comparison: same temperature/top-p
# sweep, same prompt set, plus one EOS-disabled long-form sample.
set -u

cd "$(dirname "$0")/.."

CKPT=experiments/checkpoints/leaderboard-final.pt

# Architecture flags must match the Phase-5 training run exactly so the
# checkpoint loads cleanly.
COMMON=(
  --checkpoint-path "$CKPT"
  --vocab-path experiments/owt-bpe-32k-vocab.json
  --merges-path experiments/owt-bpe-32k-merges.txt
  --vocab-size 32000 --context-length 1024
  --d-model 1024 --num-layers 8 --num-heads 16 --d-ff 2752 --rope-theta 10000
  --use-sdpa --qk-norm --tie-embeddings --logit-soft-cap 30
  --device cuda
)

OUT_DIR=experiments/generations/leaderboard
mkdir -p "$OUT_DIR"

echo "[1] temp=0.8 top_p=0.95 EOS-terminated (matches the 7.4 OWT headline sample style)"
uv run python scripts/generate_lm.py "${COMMON[@]}" \
  --prompt "Once upon a time" \
  --max-new-tokens 320 --temperature 0.8 --top-p 0.95 \
  --eos-token "<|endoftext|>" \
  > "$OUT_DIR/sample-temp0.8-topp0.95.txt" 2>&1
echo "[1] DONE"

echo "[2] temp=0.7 top_p=0.9 EOS-terminated"
uv run python scripts/generate_lm.py "${COMMON[@]}" \
  --prompt "Once upon a time" \
  --max-new-tokens 320 --temperature 0.7 --top-p 0.9 \
  --eos-token "<|endoftext|>" \
  > "$OUT_DIR/sample-temp0.7-topp0.9.txt" 2>&1
echo "[2] DONE"

echo "[3] temp=1.0 top_p=0.9 EOS-terminated"
uv run python scripts/generate_lm.py "${COMMON[@]}" \
  --prompt "Once upon a time" \
  --max-new-tokens 320 --temperature 1.0 --top-p 0.9 \
  --eos-token "<|endoftext|>" \
  > "$OUT_DIR/sample-temp1.0-topp0.9.txt" 2>&1
echo "[3] DONE"

echo "[4] longer prompt (web-text-flavored, matches 7.4 OWT 'policy' sample)"
uv run python scripts/generate_lm.py "${COMMON[@]}" \
  --prompt "The president said today that the country must focus on" \
  --max-new-tokens 320 --temperature 0.8 --top-p 0.95 \
  --eos-token "<|endoftext|>" \
  > "$OUT_DIR/sample-policy-temp0.8-topp0.95.txt" 2>&1
echo "[4] DONE"

echo "[5] EOS-disabled 320-token sample (full 320 tokens of model output)"
uv run python scripts/generate_lm.py "${COMMON[@]}" \
  --prompt "Once upon a time" \
  --max-new-tokens 320 --temperature 0.8 --top-p 0.95 \
  --eos-token "<|never|>" \
  > "$OUT_DIR/sample-noeos-temp0.8-topp0.95.txt" 2>&1
echo "[5] DONE"

echo "ALL DONE - samples in $OUT_DIR/"
