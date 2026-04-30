#!/usr/bin/env bash
# Phase 4.5 — final LR sanity A/B at +/- sqrt(2) around the Phase-4 winning
# AdamW LR (1.0e-3), holding everything else at the cumulative Phase-4 best
# config: d=1024 L=8 bs=24 ctx=1024, full Tier-1 stack, Muon-mixed
# muon-lr=0.02, WSD schedule warmup=100 decay=20%.
#
# We already have val@2000 = 3.9365 at lr=1.0e-3 from Phase 4d
# (experiments/logs/phase4d_wsd_ctx1024_bs24.csv); we only need to add the
# two flanks.
#
# Decision rule: pick the LR with the lowest val@2000. If 0.71e-3 wins, we
# bias the final-run peak LR slightly down; if 1.41e-3 wins, slightly up.
# Margins likely small at 2000 steps; the bigger 51K-step final run will
# tolerate a wider range due to WSD's late decay.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

cd "$(dirname "$0")/.."

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"

LRS=("0.707e-3" "1.414e-3")

COMMON=(
    --train-data data/tokenized_datasets/owt-train.uint16.npy
    --val-data   data/tokenized_datasets/owt-dev.uint16.npy
    --vocab-size 32000 --context-length 1024
    --d-model 1024 --num-layers 8 --num-heads 16 --d-ff 2752 --rope-theta 10000
    --batch-size 24 --max-steps 2000
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --device cuda
    --log-every 50 --val-every 400 --val-batches 20
    --seed 42
    --use-sdpa --torch-compile --compile-mode default --bf16
    --qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4
    --optimizer muon_mixed --muon-lr 0.02
    --lr-schedule wsd --warmup-iters 100 --decay-frac 0.2
)

MAX_STEPS=2000

for LR in "${LRS[@]}"; do
    NAME="phase45-lr${LR}"
    CSV="${LOG_DIR}/${NAME}.csv"
    LOG="${LOG_DIR}/${NAME}.console.log"
    rm -f "${CSV}"
    echo "[START] ${NAME} lr=${LR} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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
        echo "  last 10 lines:"
        tail -10 "${LOG}"
        exit ${EC:-1}
    fi
done
