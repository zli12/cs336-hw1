#!/usr/bin/env bash
# Post-training tasks for the OWT main_experiment (Phases 3 + 4 of the 7.4 plan):
#   Phase 3: generate text samples from the final checkpoint
#   Phase 4 (plots): OWT learning curve + OWT-vs-TinyStories side-by-side
#
# Run AFTER scripts/run_owt_final.sh completes. Output goes to
# experiments/_owt_post_run_report.txt for review.
set -u

cd "$(dirname "$0")/.."

LOG="$(pwd)/experiments/_owt_post_run_report.txt"
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo "=== 1) Generate OWT samples ==="
bash scripts/run_owt_samples.sh

echo "=== 2) OWT learning curve plot ==="
uv run python scripts/plot_runs.py --kind lr-sweep \
  --inputs experiments/logs/owt-final-cosine-bs96-lr2.45e-3.csv \
  --labels "OWT cosine 10K (bs=96, lr_max=2.45e-3)" \
  --out experiments/problem_responses/7.4_owt_learning_curve.svg \
  --title "OWT main_experiment learning curve (10K steps, cosine schedule)"

echo "=== 3) OWT vs TinyStories comparison plot ==="
uv run python scripts/plot_runs.py --kind batch-sweep \
  --inputs \
    experiments/logs/owt-final-cosine-bs96-lr2.45e-3.csv \
    experiments/logs/lr-final-cosine-lr2e-3.csv \
  --labels \
    "OWT (vocab=32k, bs=96, 245M tokens)" \
    "TinyStories (vocab=10k, bs=128, 327M tokens)" \
  --out experiments/problem_responses/7.4_owt_vs_tinystories.svg \
  --title "OWT vs TinyStories: same architecture, same 10K iter count, cosine schedule"

echo "=== 4) Verify outputs ==="
ls -la experiments/generations/owt/
ls -la experiments/problem_responses/7.4_owt_learning_curve.svg
ls -la experiments/problem_responses/7.4_owt_vs_tinystories.svg

echo "=== 5) Sample file contents ==="
for f in experiments/generations/owt/*.txt; do
  echo "===== $f ====="
  cat "$f"
  echo
done

echo "=== DONE; full log: $LOG ==="
