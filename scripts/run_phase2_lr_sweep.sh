#!/usr/bin/env bash
# Phase 2c — 3-LR mini-sweep at the Phase 2b winner (d=1024 L=8, bs=96).
#
# d=1024 L=8 is the only Phase-2b config inside the plan's 30-60K step target
# band on the 11700s budget (61K steps at 124K tok/s). It also wins val@2000
# by 0.009 nats over d=768 L=8.
#
# Memory-wise this config is at the edge of the A100-40GB headroom (the
# (B, T, V) bf16 logits + their fp32 grad in backward + soft-cap intermediate
# add up to ~38-39 GiB), so cleanups across consecutive runs in the same
# shell session matter: `train_lm.py`'s exit path now does an explicit
# `_dynamo.reset()` + `torch.cuda.synchronize()` + `torch.cuda.empty_cache()`
# before `os._exit(0)`, and we set `TORCHINDUCTOR_COMPILE_THREADS=1` to keep
# parallel compile workers from spawning their own CUDA contexts.
#
# Plan-prescribed LR ladder centered on the muP-style sqrt-scaled anchor:
#   lr_mid = 2e-3 * sqrt(512 / 1024) = 1.414e-3  (the Phase 2b winner LR)
#   lr_low = lr_mid / sqrt(2)        ~= 1.000e-3
#   lr_hi  = lr_mid * sqrt(2)        ~= 2.000e-3
# 1.000e-3 also matches the Phase 0.5 boundary; 2.000e-3 matches the Phase
# 1.5 winner LR. So the ladder is a clean factor-of-2 around the sqrt-scaled
# point, anchored on points we already have d=512 data for.
#
# 2000 steps each, fixed LR, full Tier-1 stack. Pick the LR with the lowest
# val@2000 as the joint (size, LR) winner for Phase 2.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Single-threaded compile workers to avoid extra CUDA contexts under tight
# memory budgets.
export TORCHINDUCTOR_COMPILE_THREADS=1

cd "$(dirname "$0")/.."

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"

LRS=("1.0e-3" "1.414e-3" "2.0e-3")

COMMON=(
    --train-data data/tokenized_datasets/owt-train.uint16.npy
    --val-data   data/tokenized_datasets/owt-dev.uint16.npy
    --vocab-size 32000 --context-length 256
    --d-model 1024 --num-layers 8 --num-heads 16 --d-ff 2752 --rope-theta 10000
    --batch-size 96 --max-steps 2000
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --lr-schedule fixed --device cuda
    --log-every 50 --val-every 400 --val-batches 20
    --seed 42
    --use-sdpa --torch-compile --compile-mode default --bf16
    --qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4
)

MAX_STEPS=2000

for LR in "${LRS[@]}"; do
    NAME="phase2-lr-${LR}"
    CSV="${LOG_DIR}/${NAME}.csv"
    LOG="${LOG_DIR}/${NAME}.console.log"
    rm -f "${CSV}"
    echo "[START] ${NAME} d=1024 L=8 bs=96 lr=${LR} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_lm.py \
        "${COMMON[@]}" \
        --learning-rate "${LR}" \
        --metrics-csv "${CSV}" \
        > "${LOG}" 2>&1
    EC=$?
    HAS_FINAL_VAL=$(awk -F, -v step="${MAX_STEPS}" '$1==step && $3=="val" {print "yes"; exit}' "${CSV}" 2>/dev/null)
    if [[ "${HAS_FINAL_VAL}" == "yes" ]]; then
        echo "[END]   ${NAME} exit=${EC} (csv has step=${MAX_STEPS} val row -> success) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
        echo "[END]   ${NAME} exit=${EC} (no step=${MAX_STEPS} val row in csv -> FAILED) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "  see ${LOG} for failure details"
        echo "  last 20 lines:"
        tail -20 "${LOG}"
        exit ${EC:-1}
    fi
done

echo
echo "=== Phase 2c LR mini-sweep summary ==="
echo "(see /tmp/phase2c_summary.py for the analysis script)"
