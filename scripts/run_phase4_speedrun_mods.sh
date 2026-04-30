#!/usr/bin/env bash
# Phase 4 — cumulative greedy ablation of speedrun mods, starting from the Phase 3
# winning stack (d=1024 L=8 bs=96 ctx=256, full Tier-1 stack, Muon-mixed
# muon-lr=0.02 + AdamW lr=1.0e-3, fixed schedule, SwiGLU FFN).
#
# Cumulative-add order (each iteration starts from the previous winner config):
#   4a — switch fixed schedule -> WSD (warmup 100, decay 20%)
#   4b — change SwiGLU FFN -> ReLU^2 FFN
#   4c — value embeddings / U-net skip at layer L/2 (= layer 4 of 8)
#   4d — context length 256 -> 1024 (with bs reduction if needed to fit)
#
# Decision rule per mod: keep if val@2000 better by >0.005 nats, drop otherwise.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

cd "$(dirname "$0")/.."

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"

# Common base: Phase 3 winner config (Muon-mixed, full Tier-1 stack, d=1024 L=8 bs=96).
COMMON_BASE=(
    --train-data data/tokenized_datasets/owt-train.uint16.npy
    --val-data   data/tokenized_datasets/owt-dev.uint16.npy
    --vocab-size 32000
    --d-model 1024 --num-layers 8 --num-heads 16 --d-ff 2752 --rope-theta 10000
    --batch-size 96 --max-steps 2000 --learning-rate 1.0e-3
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --device cuda
    --log-every 50 --val-every 400 --val-batches 20
    --seed 42
    --use-sdpa --torch-compile --compile-mode default --bf16
    --qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4
    --optimizer muon_mixed --muon-lr 0.02
)

# Each row: NAME : EXTRA_FLAGS (the cumulative additions added to COMMON_BASE).
# 4a is the only ablation that always uses ctx=256; the rest may need adjustment
# (handled per-cell below).
ABLATIONS=(
    "phase4a_wsd:--context-length 256 --lr-schedule wsd --warmup-iters 100 --decay-frac 0.2"
    "phase4b_wsd_relu2:--context-length 256 --lr-schedule wsd --warmup-iters 100 --decay-frac 0.2 --ffn-type relu2"
    # Phase 4b decision: ReLU2 lost to SwiGLU at this scale (val@2000 4.0149 vs 3.9951; +0.020 nats).
    # 4c onwards build on the Phase-4a winner (WSD + SwiGLU + Muon-mixed).
    "phase4c_wsd_valueembed_l4:--context-length 256 --lr-schedule wsd --warmup-iters 100 --decay-frac 0.2 --value-embed-layers 4"
)

MAX_STEPS=2000

run_cell() {
    local NAME="$1"
    local EXTRA="$2"
    local CSV="${LOG_DIR}/${NAME}.csv"
    local LOG="${LOG_DIR}/${NAME}.console.log"
    rm -f "${CSV}"
    echo "[START] ${NAME} extra='${EXTRA}' $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_lm.py \
        "${COMMON_BASE[@]}" \
        ${EXTRA} \
        --metrics-csv "${CSV}" \
        > "${LOG}" 2>&1
    local EC=$?
    local HAS_FINAL_VAL
    HAS_FINAL_VAL=$(awk -F, -v step="${MAX_STEPS}" '$1==step && $3=="val" {print "yes"; exit}' "${CSV}" 2>/dev/null)
    if [[ "${HAS_FINAL_VAL}" == "yes" ]]; then
        echo "[END]   ${NAME} exit=${EC} (csv has step=${MAX_STEPS} val row -> success) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        return 0
    fi
    echo "[END]   ${NAME} exit=${EC} (no step=${MAX_STEPS} val row in csv -> FAILED) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "  see ${LOG} for failure details"
    echo "  last 10 lines:"
    tail -10 "${LOG}"
    return 1
}

for ROW in "${ABLATIONS[@]}"; do
    NAME="${ROW%%:*}"
    EXTRA="${ROW#*:}"
    run_cell "${NAME}" "${EXTRA}" || exit $?
done

echo
echo "=== Phase 4 partial summary ==="
echo "(see /tmp/phase4_summary.py for the cumulative-greedy comparison)"
