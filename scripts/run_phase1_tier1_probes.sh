#!/usr/bin/env bash
# Phase 1 — Tier-1 quality-win A/B probes.
#
# Each run uses the Phase-0.5 winning anchor LR (2e-3) and the same OWT config
# + Phase-0 winning stack (--bf16 --use-sdpa --torch-compile). Seed=42 across
# all runs. The Phase-0.5 baseline run (val@1500=4.6744) serves as the no-mod
# reference; here we A/B each Tier-1 mod individually so we can attribute
# gains/regressions to each mod cleanly.
#
# Mods tested:
#   1. tie         — weight-tie input embedding with LM head; embed_init_std auto-set to 1/sqrt(d_model).
#   2. qknorm      — per-head RMSNorm on Q/K post-RoPE.
#   3. softcap     — tanh logit soft-cap with cap=30 (Gemma-2 style).
#   4. zloss       — auxiliary z-loss with weight=1e-4 (PaLM-style).
#   5. embedinit   — embed_init_std=0.0442 (~1/sqrt(d_model)) WITHOUT tying, to isolate the init-only effect.
#
# Decision rule per mod: keep it if val@1500 is no worse than +0.01 nats above the Phase-0.5 baseline (4.6744).
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
    --batch-size 96 --max-steps 1500 --learning-rate 2e-3
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --lr-schedule fixed --device cuda
    --log-every 50 --val-every 300 --val-batches 20
    --seed 42
    --use-sdpa --torch-compile --bf16
)

CONFIGS=(
    "tie:--tie-embeddings"
    "qknorm:--qk-norm"
    "softcap:--logit-soft-cap 30"
    "zloss:--z-loss-weight 1e-4"
    "embedinit:--embed-init-std 0.0442"
)

for CFG_PAIR in "${CONFIGS[@]}"; do
    NAME="phase1-${CFG_PAIR%%:*}"
    EXTRA="${CFG_PAIR#*:}"
    CSV="${LOG_DIR}/${NAME}.csv"
    LOG="${LOG_DIR}/${NAME}.console.log"
    rm -f "${CSV}"
    echo "[START] ${NAME} extra='${EXTRA}' $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # shellcheck disable=SC2086
    uv run python scripts/train_lm.py \
        "${COMMON[@]}" \
        --metrics-csv "${CSV}" \
        ${EXTRA} \
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
echo "=== Phase 1 Tier-1 A/B summary ==="
uv run python - <<'PY'
import csv
from pathlib import Path

CONFIGS = ["tie", "qknorm", "softcap", "zloss", "embedinit"]
LOG_DIR = Path("experiments/logs")
BASELINE = 4.6744  # val@1500 from phase05-lr2e-3

print(f"{'config':<14s} {'tok/s (avg)':>14s} {'val@300':>10s} {'val@600':>10s} {'val@900':>10s} {'val@1200':>10s} {'val@1500':>10s} {'\u0394 vs base':>10s}")
print("-" * 96)
print(f"{'baseline':<14s} {'~234,500':>14s} {'5.5690':>10s} {'5.1557':>10s} {'4.9284':>10s} {'4.7873':>10s} {'4.6744':>10s} {'0.0000':>10s}")
print("-" * 96)
keep = []
drop = []
for cfg in CONFIGS:
    path = LOG_DIR / f"phase1-{cfg}.csv"
    if not path.exists():
        print(f"{cfg:<14s} (missing csv)")
        continue
    rows = list(csv.DictReader(path.open()))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    tok_rates = [float(r["tokens_per_sec"]) for r in train_rows if r["tokens_per_sec"] and int(r["step"]) > 50]
    avg_tok = sum(tok_rates) / max(1, len(tok_rates))
    val_at = {int(r["step"]): float(r["loss"]) for r in val_rows}
    final = val_at.get(1500, float("nan"))
    delta = final - BASELINE
    cells = " ".join(f"{val_at.get(s, float('nan')):>10.4f}" for s in (300, 600, 900, 1200, 1500))
    print(f"{cfg:<14s} {avg_tok:>14,.0f} {cells} {delta:>+10.4f}")
    if delta < 0.01:  # within +0.01 nats of baseline (or better) -> keep
        keep.append((cfg, delta))
    else:
        drop.append((cfg, delta))

print()
print(f"KEEP (delta < +0.01 nats): {[(c, f'{d:+.4f}') for c, d in keep]}")
print(f"DROP (regressed):          {[(c, f'{d:+.4f}') for c, d in drop]}")
PY
