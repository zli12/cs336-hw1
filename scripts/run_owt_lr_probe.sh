#!/usr/bin/env bash
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

LRS=("1e-3" "2e-3" "3e-3")

for LR in "${LRS[@]}"; do
  RUN_NAME="owt-lr-probe-fixed-lr${LR}"
  CSV="experiments/logs/${RUN_NAME}.csv"
  LOG="experiments/logs/${RUN_NAME}.console.log"
  echo "[START] ${RUN_NAME} lr=${LR} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uv run python scripts/train_lm.py \
    --train-data data/tokenized_datasets/owt-train.uint16.npy \
    --val-data data/tokenized_datasets/owt-dev.uint16.npy \
    --vocab-size 32000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
    --batch-size 64 --max-steps 3000 --learning-rate "${LR}" \
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0 \
    --lr-schedule fixed --device cuda \
    --log-every 50 --val-every 200 --val-batches 20 \
    --metrics-csv "${CSV}" \
    > "${LOG}" 2>&1
  EC=$?
  echo "[END] ${RUN_NAME} exit=${EC} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done
echo "[ALL DONE] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
