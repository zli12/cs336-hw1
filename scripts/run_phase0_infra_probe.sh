#!/usr/bin/env bash
# Phase 0 — leaderboard infra speedup probe (see experiments/problem_responses/7.5_leaderboard_plan.md).
#
# Runs the same 4-layer / 512d / bs=96 / ctx=256 OWT config that 7.4 used, but
# layers Phase-0 infra options on top to measure the throughput uplift and
# verify val loss is unchanged within +/-0.005 nats:
#
#   1. fp32_baseline       --device cuda                                        (TF32 on, einsum attention)
#   2. sdpa                --use-sdpa
#   3. sdpa_compile        --use-sdpa --torch-compile
#   4. sdpa_compile_bf16   --use-sdpa --torch-compile --bf16                   (full Phase-0 stack)
#
# All 4 use seed=42 so batches are byte-identical; any val-loss drift comes
# from the infra layer being toggled, not from sampling.
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
    --batch-size 96 --max-steps 200 --learning-rate 2e-3
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0
    --lr-schedule fixed --device cuda
    --log-every 20 --val-every 100 --val-batches 5
    --seed 42
)

CONFIGS=(
    "fp32_baseline:"
    "sdpa:--use-sdpa"
    "sdpa_compile:--use-sdpa --torch-compile"
    "sdpa_compile_bf16:--use-sdpa --torch-compile --bf16"
)

for CFG_PAIR in "${CONFIGS[@]}"; do
    NAME="${CFG_PAIR%%:*}"
    EXTRA="${CFG_PAIR#*:}"
    CSV="${LOG_DIR}/phase0-${NAME}.csv"
    LOG="${LOG_DIR}/phase0-${NAME}.console.log"
    rm -f "${CSV}"
    echo "[START] phase0-${NAME} extra='${EXTRA}' $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # shellcheck disable=SC2086
    uv run python scripts/train_lm.py \
        "${COMMON[@]}" \
        --metrics-csv "${CSV}" \
        ${EXTRA} \
        > "${LOG}" 2>&1
    EC=$?
    echo "[END]   phase0-${NAME} exit=${EC} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ ${EC} -ne 0 ]]; then
        echo "  see ${LOG} for failure details"
        echo "  last 20 lines:"
        tail -20 "${LOG}"
        exit ${EC}
    fi
done

echo
echo "=== Phase 0 summary ==="
uv run python - <<'PY'
import csv
from pathlib import Path

CONFIGS = ["fp32_baseline", "sdpa", "sdpa_compile", "sdpa_compile_bf16"]
LOG_DIR = Path("experiments/logs")

print(f"{'config':<22s} {'tok/s (final 100)':>20s} {'val@100':>10s} {'val@200':>10s}")
print("-" * 65)
for cfg in CONFIGS:
    path = LOG_DIR / f"phase0-{cfg}.csv"
    if not path.exists():
        print(f"{cfg:<22s} (missing csv)")
        continue
    rows = list(csv.DictReader(path.open()))
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    # Average tokens/sec excluding the very first step (cold-start outlier).
    tok_rates = [float(r["tokens_per_sec"]) for r in train_rows if r["tokens_per_sec"] and int(r["step"]) > 20]
    avg_tok = sum(tok_rates) / max(1, len(tok_rates))
    val_at = {int(r["step"]): float(r["loss"]) for r in val_rows}
    print(f"{cfg:<22s} {avg_tok:>20,.0f} {val_at.get(100, float('nan')):>10.4f} {val_at.get(200, float('nan')):>10.4f}")
PY
