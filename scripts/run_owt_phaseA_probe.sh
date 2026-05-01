#!/usr/bin/env bash
# Phase-A probe for 7.5 leaderboard: bf16 + torch.compile + Flash SDPA.
# Same model + bs + LR as the 7.4 baseline; 1000 steps; fixed schedule for cleanest A/B.
# Compares against baseline 7.4 step-1000 reading: train_loss=4.956, tok/s=30905.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Each torch.compile worker process grabs its own CUDA context (~1.5-2 GiB on A10G);
# limit to 1 worker to fit alongside the bs=96 forward/backward graph.
export TORCHINDUCTOR_COMPILE_THREADS=1

cd "$(dirname "$0")/.."

RUN_NAME="owt-phaseA-probe-bf16-compile-sdpa-bs96-lr2.45e-3"
CSV="experiments/logs/${RUN_NAME}.csv"
LOG="experiments/logs/${RUN_NAME}.console.log"

mkdir -p experiments/logs

echo "[START] ${RUN_NAME} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
uv run python scripts/train_lm.py \
  --train-data data/tokenized_datasets/owt-train.uint16.npy \
  --val-data data/tokenized_datasets/owt-dev.uint16.npy \
  --vocab-size 32000 --context-length 256 \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
  --batch-size 96 --max-steps 1000 \
  --lr-schedule fixed \
  --learning-rate 2.45e-3 \
  --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0 \
  --device cuda --dtype bf16 --compile --attn-kernel torch \
  --log-every 50 --val-every 200 --val-batches 20 \
  --metrics-csv "${CSV}" \
  > "${LOG}" 2>&1
EC=$?
echo "[END] ${RUN_NAME} exit=${EC} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
