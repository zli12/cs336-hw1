#!/usr/bin/env bash
# Phase 5 - final compute-budgeted A100 leaderboard run.
#
# Locked-in config (from Phase 4.5 cumulative best, val@2000 = 3.9209):
#   - d=1024 L=8, num-heads=16, d-ff=2752, ctx=1024, bs=24
#   - Optimizer: Muon-mixed (muon-lr=0.02 for 2D matmul weights)
#                + AdamW (lr=1.414e-3 for embeddings, biases, RMSNorm gains)
#   - LR schedule: WSD with warmup=100 and decay_frac=0.2
#   - Tier-1 stack: --qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4
#   - Infra: --bf16 --use-sdpa --torch-compile --compile-mode default
#
# Budget:
#   - Compute envelope: 11700s on A100-40GB (== 1.5h H100 bf16-dense per
#     Phase 0 conversion factor of ~2.0-2.5x).
#   - Wallclock cap: 11400s (5-min safety margin under 11700s; the
#     --max-wallclock-sec guard inside train_lm.py will force a final val +
#     checkpoint write before exiting if we cross this).
#   - Measured throughput from a previous Phase-5 launch attempt that ran
#     ~1600 steps before being aborted (see leaderboard-final-aborted.* if
#     archived): 102K tok/s training-only steady state, plus ~8s overhead
#     per val pass (val_every=1000, val_batches=50). That gives an
#     effective cycle of 248.94s per 1000 steps, i.e. ~4.018 steps/s.
#   - Step budget at 4.018 steps/s × 11400s = 45,800 steps. We round down
#     to 45000 so the WSD schedule reaches LR=0 at step 45000 and we
#     finish naturally in ~11,200s, leaving ~200s of margin under the
#     11400s wallclock cap. (The earlier sanity probe at 105.6K tok/s was
#     run-by-run noise on a 200-step window; the longer run reveals 102K
#     as the true training-only steady state.)
#   - The --max-wallclock-sec 11400 guard is the second line of defense:
#     if throughput drops below ~102K tok/s mid-run we'd otherwise overrun
#     11400s; the guard cuts us off cleanly with a final val + checkpoint.
#
# Outputs:
#   - experiments/logs/leaderboard-final.csv        (step+wallclock metrics)
#   - experiments/logs/leaderboard-final.console.log (full stdout/stderr)
#   - experiments/checkpoints/leaderboard-final.pt   (final model + opt state)
#
# Exit handling: at end of run, verify the CSV has at least one val row past
# step 36000 (= 0.8 * max_steps, i.e. the start of the WSD decay phase) so a
# hung run or wallclock-cut-off-too-early run is flagged loudly.

set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=1

cd "$(dirname "$0")/.."

LOG_DIR="experiments/logs"
CKPT_DIR="experiments/checkpoints"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

NAME="leaderboard-final"
CSV="${LOG_DIR}/${NAME}.csv"
LOG="${LOG_DIR}/${NAME}.console.log"
CKPT="${CKPT_DIR}/${NAME}.pt"

# Pre-flight: refuse to clobber an existing run unless explicitly forced.
if [[ -f "${CSV}" || -f "${CKPT}" ]]; then
    if [[ "${FORCE:-0}" != "1" ]]; then
        echo "Refusing to overwrite existing ${NAME} artifacts:"
        [[ -f "${CSV}" ]] && echo "  ${CSV}"
        [[ -f "${CKPT}" ]] && echo "  ${CKPT}"
        echo "Re-run with FORCE=1 to overwrite."
        exit 1
    fi
    rm -f "${CSV}" "${LOG}" "${CKPT}"
fi

ARGS=(
    --train-data data/tokenized_datasets/owt-train.uint16.npy
    --val-data   data/tokenized_datasets/owt-dev.uint16.npy
    --vocab-size 32000 --context-length 1024
    --d-model 1024 --num-layers 8 --num-heads 16 --d-ff 2752 --rope-theta 10000
    --batch-size 24
    --max-steps 45000
    --max-wallclock-sec 11400
    --learning-rate 1.414e-3
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --device cuda
    --log-every 100 --val-every 1000 --val-batches 50
    --seed 42
    --use-sdpa --torch-compile --compile-mode default --bf16
    --qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4
    --optimizer muon_mixed --muon-lr 0.02
    --lr-schedule wsd --warmup-iters 100 --decay-frac 0.2
    --metrics-csv "${CSV}"
    --checkpoint-path "${CKPT}" --checkpoint-every 5000
)

echo "[START] ${NAME} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  csv=${CSV}"
echo "  log=${LOG}"
echo "  ckpt=${CKPT}"
echo "  budget: max-steps=45000  wallclock-cap=11400s  expected-finish=~11,200s"
echo "  config: d=1024 L=8 bs=24 ctx=1024 / muon-lr=0.02 + adamw lr=1.414e-3 / WSD warmup=100 decay=0.2"

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_lm.py "${ARGS[@]}" \
    > "${LOG}" 2>&1
EC=$?

# Validate run finished cleanly:
#   - process exited with EC=0 (the os._exit(0) at the end of train_lm.py
#     guarantees this for any successful inner run)
#   - CSV has at least one val row past step 38400 (= 0.8 * max_steps), i.e.
#     the WSD decay phase actually started
DEEP_VAL_STEP=$(awk -F, '$3=="val" && $1+0 > 36000 {print $1; exit}' "${CSV}" 2>/dev/null)
LAST_VAL_STEP=$(awk -F, '$3=="val" {last=$1} END {print last}' "${CSV}" 2>/dev/null)
LAST_VAL_LOSS=$(awk -F, '$3=="val" {last=$4} END {print last}' "${CSV}" 2>/dev/null)
LAST_WALLCLOCK=$(awk -F, '$3=="val" {last=$2} END {print last}' "${CSV}" 2>/dev/null)

if [[ "${EC}" == "0" && -n "${DEEP_VAL_STEP}" ]]; then
    echo "[END]   ${NAME} exit=0 last_val_step=${LAST_VAL_STEP} val_loss=${LAST_VAL_LOSS} wallclock=${LAST_WALLCLOCK}s $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "  decay phase verified: first val past step 36000 was at step ${DEEP_VAL_STEP}"
else
    echo "[END]   ${NAME} exit=${EC} (FAILED or terminated before decay phase) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "  last_val_step=${LAST_VAL_STEP} (expected past 36000) val_loss=${LAST_VAL_LOSS}"
    echo "  see ${LOG} for failure details. Last 20 lines:"
    tail -20 "${LOG}"
    exit "${EC:-1}"
fi
