#!/usr/bin/env bash
# Phase 3 — Muon vs AdamW A/B at the Phase 2 winner config.
#
# Config: d=1024 L=8 bs=96 ctx=256, full Tier-1 stack
# (--qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4),
# bf16 + SDPA + compile=default. Same seed (42) for both arms so the per-step
# input batches are byte-identical and any difference in val loss is from the
# optimizer alone.
#
# Arm A (control): --optimizer adamw --learning-rate 1.0e-3 (Phase 2c winner).
# Arm B (Muon-mixed): --optimizer muon_mixed --learning-rate 1.0e-3 (still
#   the AdamW LR, applied to embeddings/tied LM head/biases/norms) plus
#   --muon-lr 0.02 (NanoGPT-speedrun default for the matmul matrices).
#
# Decision rule: if Muon's val@2000 beats AdamW's by more than the noise
# envelope (~0.005 nats), keep Muon and run a 2-LR mini-sweep for muon-lr
# in {0.02, 0.05} (Phase 3b). If Muon loses or ties, drop it.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

cd "$(dirname "$0")/.."

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"

COMMON=(
    --train-data data/tokenized_datasets/owt-train.uint16.npy
    --val-data   data/tokenized_datasets/owt-dev.uint16.npy
    --vocab-size 32000 --context-length 256
    --d-model 1024 --num-layers 8 --num-heads 16 --d-ff 2752 --rope-theta 10000
    --batch-size 96 --max-steps 2000 --learning-rate 1.0e-3
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --lr-schedule fixed --device cuda
    --log-every 50 --val-every 400 --val-batches 20
    --seed 42
    --use-sdpa --torch-compile --compile-mode default --bf16
    --qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4
)

CONFIGS=(
    "adamw_baseline:--optimizer adamw"
    "muon_mixed_lr0.02:--optimizer muon_mixed --muon-lr 0.02"
)

MAX_STEPS=2000

for ROW in "${CONFIGS[@]}"; do
    NAME="phase3-${ROW%%:*}"
    EXTRA="${ROW#*:}"
    CSV="${LOG_DIR}/${NAME}.csv"
    LOG="${LOG_DIR}/${NAME}.console.log"
    rm -f "${CSV}"
    echo "[START] ${NAME} d=1024 L=8 bs=96 lr=1.0e-3 extra='${EXTRA}' $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_lm.py \
        "${COMMON[@]}" \
        ${EXTRA} \
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
echo "=== Phase 3 Muon vs AdamW summary ==="
uv run python /tmp/phase3_summary.py 2>/dev/null || echo "(write a summary script at /tmp/phase3_summary.py to compute the comparison)"
