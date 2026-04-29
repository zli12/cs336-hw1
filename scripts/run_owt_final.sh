#!/usr/bin/env bash
# OWT main_experiment full-budget run (Phase 2 of the 7.4 plan).
#
# Config rationale (see experiments/problem_responses/7.1_experiment_log.md
# §7.4 for full evidence):
#   - bs=96 was the largest batch that fit on A10G with vocab=32k logits;
#     bs=128 OOMed. bs=96 beat bs=64 anchor by -0.142 nats at step 3000.
#   - lr_max=2.45e-3 is the sqrt(2)-scaled LR from the bs=64 LR-probe winner
#     (lr=2e-3). Probe at this LR converged smoothly through 3000 steps.
#   - Cosine schedule with warmup=500 and decay to lr_min=lr_max/10 mirrors
#     the TinyStories §7.2 final run that gained ~0.10 nats over fixed-LR.
#   - 10000 steps matches the TinyStories iter count (assignment requirement).
#
# Wallclock estimate: ~2.2 h on A10G (bs=96 takes ~0.79 s/step).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

RUN_NAME="owt-final-cosine-bs96-lr2.45e-3"
CSV="experiments/logs/${RUN_NAME}.csv"
LOG="experiments/logs/${RUN_NAME}.console.log"
CKPT="experiments/checkpoints/owt-final-cosine-bs96.pt"

mkdir -p experiments/logs experiments/checkpoints

echo "[START] ${RUN_NAME} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
uv run python scripts/train_lm.py \
  --train-data data/tokenized_datasets/owt-train.uint16.npy \
  --val-data data/tokenized_datasets/owt-dev.uint16.npy \
  --vocab-size 32000 --context-length 256 \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
  --batch-size 96 --max-steps 10000 \
  --lr-schedule cosine \
  --learning-rate 2.45e-3 --min-learning-rate 2.45e-4 \
  --warmup-iters 500 --cosine-cycle-iters 10000 \
  --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0 \
  --device cuda \
  --log-every 50 --val-every 200 --val-batches 20 \
  --checkpoint-path "${CKPT}" \
  --checkpoint-every 1000 \
  --metrics-csv "${CSV}" \
  > "${LOG}" 2>&1
EC=$?
echo "[END] ${RUN_NAME} exit=${EC} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
