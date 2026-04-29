#!/usr/bin/env bash
# Phase 1.5 — QK-Norm-aware LR re-anchor.
#
# Why: QK-Norm reliably lifts the LR optimum (typically 2-4x) by stabilizing
# the QK product. The Phase-0.5 anchor of 2e-3 was tuned on the OLD architecture
# without QK-Norm. Before scaling up the model in Phase 2, we need to find the
# new LR optimum with the full Phase-1 winning stack enabled, so any Phase-2
# sqrt-scaling guess starts from the right anchor.
#
# Setup: full Phase-1 default stack (qk-norm + tie + softcap + zloss) on the
# OLD model shape (4-layer / 512d / 16-head / d_ff=1344 / ctx=256 / vocab=32k),
# Phase-0 winning infra (--bf16 --use-sdpa --torch-compile), bs=96, fixed LR.
# Sweep: lr ∈ {2e-3 (anchor), 4e-3 (2x), 8e-3 (4x)}, 1500 steps each.
# Seed: 42 across all 3 runs so per-step batches are byte-identical.
#
# Decision rule: pick the LR with the lowest val@1500. That becomes the
# "QK-Norm-aware anchor" used by Phase 2's sqrt-scaling guess.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."

LOG_DIR="experiments/logs"
mkdir -p "${LOG_DIR}"

COMMON=(
    --train-data data/tokenized_datasets/owt-train.uint16.npy
    --val-data   data/tokenized_datasets/owt-dev.uint16.npy
    --vocab-size 32000 --context-length 256
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000
    --batch-size 96 --max-steps 1500
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --lr-schedule fixed --device cuda
    --log-every 50 --val-every 300 --val-batches 20
    --seed 42
    --use-sdpa --torch-compile --compile-mode default --bf16
    --qk-norm --tie-embeddings --logit-soft-cap 30 --z-loss-weight 1e-4
)
# NOTE on --compile-mode default (vs the reduce-overhead used in earlier
# phases): with the full Tier-1 stack, the (96, 256, 32000) bf16 logits flow
# through (lm_head -> tanh soft-cap -> cross_entropy_with_z_loss). The
# reduce-overhead path additionally allocates a CUDA-graph memory pool that
# double-counts those large activations and pushes peak GPU memory past 40 GB
# at bs=96. The default mode still fuses kernels (and turned out to be ~10%
# faster on this stack at this scale during a 200-step probe), without the
# CUDA-graph pool overhead.

LRS=("2e-3" "4e-3" "8e-3")

# Robustness: trailing OOMs in torch.compile cleanup at process exit can
# return a non-zero exit code even when the run actually completed. We treat
# the run as successful if the CSV contains a ``val`` row at the requested
# ``--max-steps``. Otherwise we report the failure and exit.
MAX_STEPS=1500

for LR in "${LRS[@]}"; do
    NAME="phase15-lr${LR}"
    CSV="${LOG_DIR}/${NAME}.csv"
    LOG="${LOG_DIR}/${NAME}.console.log"
    rm -f "${CSV}"
    echo "[START] ${NAME} lr=${LR} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    uv run python scripts/train_lm.py \
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
echo "=== Phase 1.5 QK-Norm-aware LR sweep summary ==="
uv run python - <<'PY'
import csv
from pathlib import Path

LRS = ["2e-3", "4e-3", "8e-3"]
LOG_DIR = Path("experiments/logs")

PHASE05_BASE = 4.6744   # OLD arch, lr=2e-3 (Phase 0.5 winner)
PHASE1_QKNORM = 4.6375  # OLD arch + qk-norm only, lr=2e-3 (Phase 1 best single mod)

print(f"Phase 0.5 baseline (no Tier-1 mods, lr=2e-3): val@1500={PHASE05_BASE:.4f}")
print(f"Phase 1   qk-norm-only          (lr=2e-3): val@1500={PHASE1_QKNORM:.4f}")
print()
print(f"{'lr':<8s} {'tok/s (avg)':>14s} {'train@1500':>12s} {'val@300':>10s} {'val@600':>10s} {'val@900':>10s} {'val@1200':>10s} {'val@1500':>10s} {'\u0394 vs Ph0.5':>11s} {'\u0394 vs Ph1-qk':>12s}")
print("-" * 122)
best_lr = None
best_val = float("inf")
for lr in LRS:
    path = LOG_DIR / f"phase15-lr{lr}.csv"
    if not path.exists():
        print(f"{lr:<8s} (missing csv)")
        continue
    rows = list(csv.DictReader(path.open()))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    tok_rates = [float(r["tokens_per_sec"]) for r in train_rows if r["tokens_per_sec"] and int(r["step"]) > 50]
    avg_tok = sum(tok_rates) / max(1, len(tok_rates))
    train_at_end = float(train_rows[-1]["loss"]) if train_rows else float("nan")
    val_at = {int(r["step"]): float(r["loss"]) for r in val_rows}
    final_val = val_at.get(1500, float("nan"))
    if final_val < best_val:
        best_val = final_val
        best_lr = lr
    cells = " ".join(f"{val_at.get(s, float('nan')):>10.4f}" for s in (300, 600, 900, 1200, 1500))
    d05 = final_val - PHASE05_BASE
    d1 = final_val - PHASE1_QKNORM
    print(f"{lr:<8s} {avg_tok:>14,.0f} {train_at_end:>12.4f} {cells} {d05:>+11.4f} {d1:>+12.4f}")

print()
print(f"Phase 1.5 QK-Norm-aware anchor LR -> {best_lr} (val@1500={best_val:.4f})")
PY
