#!/usr/bin/env bash
# Phase-B probe driver for 7.5: 3K-step OWT run on top of Phase-A speedups.
# Usage: scripts/run_owt_phaseB_probe.sh <suffix> [extra train_lm.py flags...]
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

cd "$(dirname "$0")/.."

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <suffix> [extra train_lm.py flags...]" >&2
    exit 2
fi

SUFFIX="$1"
shift

RUN_NAME="owt-phaseB-probe-${SUFFIX}"
CSV="experiments/logs/${RUN_NAME}.csv"
LOG="experiments/logs/${RUN_NAME}.console.log"

mkdir -p experiments/logs

echo "[START] ${RUN_NAME} $(date -u +%Y-%m-%dT%H:%M:%SZ) extra-flags='$*'"
uv run python scripts/train_lm.py \
  --train-data data/tokenized_datasets/owt-train.uint16.npy \
  --val-data data/tokenized_datasets/owt-dev.uint16.npy \
  --vocab-size 32000 --context-length 256 \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
  --batch-size 96 --max-steps 3000 \
  --lr-schedule fixed \
  --learning-rate 2.45e-3 \
  --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0 \
  --device cuda --dtype bf16 --compile --attn-kernel torch \
  --log-every 50 --val-every 200 --val-batches 20 \
  --metrics-csv "${CSV}" \
  "$@" \
  > "${LOG}" 2>&1
EC=$?
echo "[END] ${RUN_NAME} exit=${EC} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
