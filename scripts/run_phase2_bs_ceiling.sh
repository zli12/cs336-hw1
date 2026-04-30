#!/usr/bin/env bash
# Phase 2a — model-size × batch-size ceiling probe.
#
# Goal: figure out the largest batch size that fits at bf16 + SDPA + compile
# (mode=default) for each candidate (d_model, num_layers) shape on A100-40GB,
# *before* committing to a full Phase-2 size sweep. Each cell runs 50 steps;
# anything that doesn't OOM in 50 steps will not OOM at 1500 either, since
# memory peaks within the first compile graph.
#
# Configs (informed by the 7.5_leaderboard_plan.md Phase 2 grid):
#   (d=512,  L=4)   <- current 7.4 / Phase 0.5 baseline; sanity row
#   (d=768,  L=6)   <- moderate scale-up
#   (d=512,  L=8)   <- "deeper not wider"
#   (d=768,  L=8)   <- moderate scale-up + deeper
#   (d=1024, L=8)   <- aggressive scale-up
# bs sweep per config: {96, 192, 384}
#
# Heads / d_ff conventions: head_dim=64 (so num_heads = d_model / 64), and
# d_ff = round(d_model * 8/3 / 64) * 64 (SwiGLU 8/3 ratio rounded to mult-of-64
# so the FFN matmul is well-aligned). For d=512 -> 1344 (matches 7.4 baseline).
# For d=768 -> 2048. For d=1024 -> 2752.
#
# Same Phase-1 winning Tier-1 stack: --qk-norm --tie-embeddings
# --logit-soft-cap 30 --z-loss-weight 1e-4. Compile mode=default to keep
# memory usable (reduce-overhead doubles the (B,T,V) buffer at capture time).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"

# Grid: tuples of "name d_model num_layers d_ff num_heads". We compute these
# offline rather than in the loop to keep the table easy to read.
CONFIGS=(
    "d512_l4    512  4   1344  16"
    "d768_l6    768  6   2048  12"
    "d512_l8    512  8   1344  16"
    "d768_l8    768  8   2048  12"
    "d1024_l8  1024  8   2752  16"
)

BATCH_SIZES=(96 192 384)

COMMON=(
    --train-data data/tokenized_datasets/owt-train.uint16.npy
    --val-data   data/tokenized_datasets/owt-dev.uint16.npy
    --vocab-size 32000 --context-length 256
    --rope-theta 10000
    --max-steps 50 --learning-rate 2e-3
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --lr-schedule fixed --device cuda
    --log-every 25 --val-every 1000 --val-batches 1
    --seed 42
    --use-sdpa --torch-compile --compile-mode default --bf16
    --qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4
)

echo "=== Phase 2a bs-ceiling probe ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

# Per-cell results: record OK / OOM along with last printed step + tokens/s.
RESULT_FILE="${LOG_DIR}/phase2-bs-ceiling.summary.txt"
: > "${RESULT_FILE}"
echo "config,batch_size,outcome,last_step,tokens_per_sec,wallclock_s,note" >> "${RESULT_FILE}"

for ROW in "${CONFIGS[@]}"; do
    # shellcheck disable=SC2086
    set -- ${ROW}
    NAME="$1"; DMODEL="$2"; LAYERS="$3"; DFF="$4"; HEADS="$5"
    for BS in "${BATCH_SIZES[@]}"; do
        CELL="phase2-${NAME}_bs${BS}"
        CSV="${LOG_DIR}/${CELL}.csv"
        LOG="${LOG_DIR}/${CELL}.console.log"
        rm -f "${CSV}"
        START_TS=$(date -u +%s)
        echo "[START] ${CELL} d=${DMODEL} L=${LAYERS} d_ff=${DFF} heads=${HEADS} bs=${BS} $(date -u +%H:%M:%S)"
        CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_lm.py \
            "${COMMON[@]}" \
            --d-model "${DMODEL}" --num-layers "${LAYERS}" --num-heads "${HEADS}" --d-ff "${DFF}" \
            --batch-size "${BS}" \
            --metrics-csv "${CSV}" \
            > "${LOG}" 2>&1
        EC=$?
        END_TS=$(date -u +%s)
        DUR=$((END_TS - START_TS))

        if [[ -f "${CSV}" ]] && awk -F, 'NR>1 && $3=="train"' "${CSV}" | tail -1 | grep -q '^50,'; then
            # Reached step 50; "OK" — read the last train row's tok/s.
            LAST_LINE=$(awk -F, '$3=="train"' "${CSV}" | tail -1)
            TPS=$(echo "${LAST_LINE}" | cut -d, -f5)
            STEP=$(echo "${LAST_LINE}" | cut -d, -f1)
            echo "[ OK   ] ${CELL} step=${STEP} tok/s=${TPS} dur=${DUR}s"
            echo "${NAME},${BS},OK,${STEP},${TPS},${DUR}," >> "${RESULT_FILE}"
        elif grep -q "OutOfMemoryError" "${LOG}"; then
            echo "[ OOM  ] ${CELL} (CUDA OOM) dur=${DUR}s"
            echo "${NAME},${BS},OOM,,,${DUR}," >> "${RESULT_FILE}"
        else
            # Some other failure (compile error, NaN, etc.).
            LAST=$(tail -3 "${LOG}" | tr '\n' ' | ')
            echo "[ FAIL ] ${CELL} ec=${EC} dur=${DUR}s last='${LAST}'"
            echo "${NAME},${BS},FAIL,,,${DUR},ec=${EC}" >> "${RESULT_FILE}"
        fi
    done
done

echo
echo "=== Phase 2a summary ==="
column -t -s ',' "${RESULT_FILE}"
echo
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
