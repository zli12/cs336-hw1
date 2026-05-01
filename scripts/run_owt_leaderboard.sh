#!/usr/bin/env bash
# 7.5 leaderboard final run: stack all winning Phase-A and Phase-B mods.
#
# Mods enabled:
#   Phase A (compute speedups, ~2x wallclock vs 7.4 baseline):
#     - bf16 autocast               (--dtype bf16)
#     - torch SDPA / FlashAttention (--attn-kernel torch)
#   Phase B (architecture/loss mods):
#     - Weight tying       (B1, -0.057 nats @3K steps probe; --tie-embeddings)
#     - QK-norm            (B2, -0.045 nats @3K steps probe; --qk-norm)
#     - PaLM-style z-loss  (B3, -0.017 nats @3K steps probe; --z-loss-coef 1e-4)
#   B4 (param-group WD) was negative at this scale and is intentionally OFF.
#
# Wallclock budget: 1.5h on A10G = 5400 s. Throughput with the full stack
# (bf16 + compile + SDPA) is ~88k tok/s at bs=96 ctx=256 -> ~0.28 s/step.
# Target 17000 steps -> ~4760 s, leaving ~640 s headroom for compile warmup,
# val passes, and checkpoint saves.
#
# Schedule: cosine with linear warmup of 500 steps, peak lr=2.45e-3 (proven
# optimum on OWT in 7.4), decay to lr_max/10 at step 17000. Same shape as 7.4
# but 1.7x more steps thanks to the Phase-A wallclock wins.
set -u
# expandable_segments avoids OOM-by-fragmentation as the model's allocation
# pattern shifts during training (warmup vs steady, fwd vs bwd vs val); the
# garbage-collection threshold makes the allocator return whole pages back
# to CUDA when reserved-but-unallocated memory exceeds 60% of allocated.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
# Single inductor compile worker (otherwise each spawns a CUDA context and
# OOMs alongside the bs=96 forward graph on the 22 GiB A10G).
export TORCHINDUCTOR_COMPILE_THREADS=1

cd "$(dirname "$0")/.."

RUN_NAME="owt-leaderboard-7.5-final"
CSV="experiments/logs/${RUN_NAME}.csv"
LOG="experiments/logs/${RUN_NAME}.console.log"
CKPT="experiments/checkpoints/${RUN_NAME}.pt"

mkdir -p experiments/logs experiments/checkpoints

echo "[START] ${RUN_NAME} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
uv run python scripts/train_lm.py \
  --train-data data/tokenized_datasets/owt-train.uint16.npy \
  --val-data data/tokenized_datasets/owt-dev.uint16.npy \
  --vocab-size 32000 --context-length 256 \
  --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 --rope-theta 10000 \
  --batch-size 96 --max-steps 17000 \
  --lr-schedule cosine \
  --learning-rate 2.45e-3 --min-learning-rate 2.45e-4 \
  --warmup-iters 500 --cosine-cycle-iters 17000 \
  --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --eps 1e-8 --max-grad-norm 1.0 \
  --device cuda --dtype bf16 --compile --attn-kernel torch \
  --tie-embeddings --qk-norm --z-loss-coef 1e-4 \
  --log-every 50 --val-every 500 --val-batches 20 \
  --checkpoint-path "${CKPT}" \
  --checkpoint-every 1000 \
  --metrics-csv "${CSV}" \
  > "${LOG}" 2>&1
EC=$?
echo "[END] ${RUN_NAME} exit=${EC} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
