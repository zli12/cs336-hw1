#!/usr/bin/env bash
# Phase 2b — model-size sweep at the bs=96 ceiling.
#
# Phase 2a established that bs=96 is the only batch size that fits with the
# full Tier-1 stack (--qk-norm --tie-embeddings --logit-soft-cap 30
# --z-loss-weight 1e-4) at bf16 + SDPA + compile (default mode) on A100-40GB
# at ctx=256 / vocab=32000. The (B, T, V) fp32 logits grad in backward
# (~6.3 GiB at bs=192) is what tips us over the 40 GiB headroom.
#
# So we sweep size at the natural ceiling and use the throughput numbers from
# Phase 2a to pick a config that lands in the plan's target step band
# (30-60K steps on the 11700s budget).
#
# Configs selected:
#   (d=768,  L=6)  ~89K steps in 11700s -> a bit fast but well-balanced
#   (d=768,  L=8)  ~78K steps              -> deeper at the same width
#   (d=1024, L=8) ~61K steps              -> the only one inside 30-60K
# We keep d=512 L=4 as the implicit control via Phase 1.5's 4.5924 result.
#
# Each run: 2000 steps, fixed LR at the muP-style sqrt-scaled anchor:
#    lr_new = 2e-3 * sqrt(512 / d_new)
# i.e. d=768 -> 1.633e-3, d=1024 -> 1.414e-3. We keep the same fixed schedule
# (no warmup/decay) for clean cross-config comparison; the actual final-run
# schedule will be cosine/WSD with proper warmup.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"

CONFIGS=(
    # name           d   L   d_ff   heads   lr
    "d768_l6        768  6   2048   12      1.633e-3"
    "d768_l8        768  8   2048   12      1.633e-3"
    "d1024_l8      1024  8   2752   16      1.414e-3"
)

COMMON=(
    --train-data data/tokenized_datasets/owt-train.uint16.npy
    --val-data   data/tokenized_datasets/owt-dev.uint16.npy
    --vocab-size 32000 --context-length 256
    --rope-theta 10000
    --batch-size 96 --max-steps 2000
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --lr-schedule fixed --device cuda
    --log-every 50 --val-every 400 --val-batches 20
    --seed 42
    --use-sdpa --torch-compile --compile-mode default --bf16
    --qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4
)

MAX_STEPS=2000

for ROW in "${CONFIGS[@]}"; do
    # shellcheck disable=SC2086
    set -- ${ROW}
    NAME="$1"; DMODEL="$2"; LAYERS="$3"; DFF="$4"; HEADS="$5"; LR="$6"
    CELL="phase2-size-${NAME}"
    CSV="${LOG_DIR}/${CELL}.csv"
    LOG="${LOG_DIR}/${CELL}.console.log"
    rm -f "${CSV}"
    echo "[START] ${CELL} d=${DMODEL} L=${LAYERS} d_ff=${DFF} heads=${HEADS} lr=${LR} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_lm.py \
        "${COMMON[@]}" \
        --d-model "${DMODEL}" --num-layers "${LAYERS}" --num-heads "${HEADS}" --d-ff "${DFF}" \
        --learning-rate "${LR}" \
        --metrics-csv "${CSV}" \
        > "${LOG}" 2>&1
    EC=$?
    HAS_FINAL_VAL=$(awk -F, -v step="${MAX_STEPS}" '$1==step && $3=="val" {print "yes"; exit}' "${CSV}" 2>/dev/null)
    if [[ "${HAS_FINAL_VAL}" == "yes" ]]; then
        echo "[END]   ${CELL} exit=${EC} (csv has step=${MAX_STEPS} val row -> success) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
        echo "[END]   ${CELL} exit=${EC} (no step=${MAX_STEPS} val row in csv -> FAILED) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "  see ${LOG} for failure details"
        echo "  last 20 lines:"
        tail -20 "${LOG}"
        exit ${EC:-1}
    fi
done

echo
echo "=== Phase 2b size-sweep summary ==="
uv run python - <<'PY'
import csv
from pathlib import Path

CONFIGS = [("d768_l6", "1.633e-3"), ("d768_l8", "1.633e-3"), ("d1024_l8", "1.414e-3")]
LOG_DIR = Path("experiments/logs")

PHASE15_BEST = 4.5924   # d=512 L=4 + full Tier-1 stack @ lr=2e-3 / 1500 steps (Phase 1.5 winner)

print(f"{'config':<12s} {'lr':<10s} {'tok/s (avg)':>14s} {'val@400':>10s} {'val@800':>10s} "
      f"{'val@1200':>10s} {'val@1600':>10s} {'val@2000':>10s} {'\u0394 vs Ph1.5':>11s}")
print("-" * 102)
best_cfg = None
best_val = float("inf")
for cfg, lr in CONFIGS:
    path = LOG_DIR / f"phase2-size-{cfg}.csv"
    if not path.exists():
        print(f"{cfg:<12s} (missing csv)")
        continue
    rows = list(csv.DictReader(path.open()))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    tok_rates = [float(r["tokens_per_sec"]) for r in train_rows
                 if r["tokens_per_sec"] and int(r["step"]) > 100]
    avg_tok = sum(tok_rates) / max(1, len(tok_rates))
    val_at = {int(r["step"]): float(r["loss"]) for r in val_rows}
    final_val = val_at.get(2000, float("nan"))
    if final_val < best_val:
        best_val = final_val
        best_cfg = (cfg, lr)
    cells = " ".join(f"{val_at.get(s, float('nan')):>10.4f}" for s in (400, 800, 1200, 1600, 2000))
    delta = final_val - PHASE15_BEST
    print(f"{cfg:<12s} {lr:<10s} {avg_tok:>14,.0f} {cells} {delta:>+11.4f}")

print()
if best_cfg is not None:
    print(f"Phase 2b size winner -> {best_cfg[0]} (lr={best_cfg[1]}, val@2000={best_val:.4f})")
PY
