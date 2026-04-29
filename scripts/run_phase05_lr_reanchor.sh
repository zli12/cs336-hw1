#!/usr/bin/env bash
# Phase 0.5 — LR re-anchor on new infra, OLD architecture.
#
# Goal: figure out whether the A10G `2e-3` LR anchor still applies on
# A100 + bf16 + SDPA + torch.compile, *before* layering any Tier-1 arch
# changes. Keeping the architecture unchanged isolates the infra effect on
# the LR landscape so Phase-1 mods have a clean baseline.
#
# Sweep: lr ∈ {1e-3, 2e-3, 4e-3}, 1500 steps each (~10 min total).
# Architecture: 4-layer / 512d / 16-head / d_ff=1344 / ctx=256 / vocab=32k (matches 7.4 baseline).
# Stack: --bf16 --use-sdpa --torch-compile (Phase-0 winning stack).
# Schedule: fixed LR (no warmup/decay) for clean LR comparison, mirrors 7.2/7.4 LR-probe protocol.
# Seed: 42 across all 3 runs so the per-step batches are byte-identical.
#
# Decision rule: pick the LR whose val loss at step 1500 is lowest.
# Expected: 2e-3 still wins, or 4e-3 marginally beats it. If 1e-3 wins, something is wrong with
# bf16/SDPA/compile and we should investigate before going further.
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
    --use-sdpa --torch-compile --bf16
)

LRS=("1e-3" "2e-3" "4e-3")

for LR in "${LRS[@]}"; do
    NAME="phase05-lr${LR}"
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
    echo "[END]   ${NAME} exit=${EC} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ ${EC} -ne 0 ]]; then
        echo "  see ${LOG} for failure details"
        echo "  last 20 lines:"
        tail -20 "${LOG}"
        exit ${EC}
    fi
done

echo
echo "=== Phase 0.5 LR re-anchor summary ==="
uv run python - <<'PY'
import csv
from pathlib import Path

LRS = ["1e-3", "2e-3", "4e-3"]
LOG_DIR = Path("experiments/logs")

print(f"{'lr':<8s} {'tok/s (avg)':>14s} {'train@1500':>12s} {'val@300':>10s} {'val@600':>10s} {'val@900':>10s} {'val@1200':>10s} {'val@1500':>10s}")
print("-" * 96)
best_lr = None
best_val = float("inf")
for lr in LRS:
    path = LOG_DIR / f"phase05-lr{lr}.csv"
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
    print(f"{lr:<8s} {avg_tok:>14,.0f} {train_at_end:>12.4f} {cells}")

print()
print(f"Phase 0.5 anchor LR -> {best_lr} (val@1500={best_val:.4f})")
PY
