# §7 Learnings — what worked, what didn't, and why

A distilled "lessons learned" from §7.1–§7.5 (TinyStories baseline + LR/bs
sweeps + architecture ablations + OWT main experiment + leaderboard
speedrun). The full numbers and run logs live in
[`7.1_to_7.5_consolidated_report.md`](../problem_responses/7.1_to_7.5_consolidated_report.md);
this doc focuses on *the principles that actually transferred*, not the
specific runs that produced them.

---

## The big ideas (read these first)

A handful of themes recurred across every phase. Each one is grounded in a
concrete A/B; the citations point to the §X.Y where it was first observed.

### 1. Late-stage LR decay is the high-value ingredient in every schedule

Three independent phases all show the same thing: **most of the gain from a
non-trivial schedule comes from the decay tail**, not the warmup.

- §7.2 cosine 10K vs fixed 10K (TinyStories): cosine *cost* −0.27 nats at
  step 200 (warmup tax), reached parity at step 2000, and the entire
  −0.10-nat end-of-run win came from the decay phase.
- §7.4 cosine 10K (OWT): val moved 4.458 → 4.017 over the last 7K decay
  steps — `−0.44` nats from decay alone.
- §7.5 WSD 45K (OWT leaderboard): val 3.397 → 3.190 over the last 9K decay
  steps. **`−0.207` nats — bigger than the entire Tier-1 quality-mod stack.**

*Why:* annealing the LR shrinks the bouncing radius around a local minimum,
letting parameters walk into a sharper basin within the same neighborhood.
At fixed LR, gradient noise keeps the parameters bouncing around that basin
indefinitely. **Warmup is conditional**: it's worth its early-step cost only
if peak LR is near the edge of stability (large models, fp16, post-norm,
QK-Norm with aggressive LR). At our scale with pre-norm + RMSNorm + bf16 +
`lr=2e-3`, warmup pays a small early tax for not-much benefit; the decay is
where the win is.

*Practical takeaway:* if you have a fixed compute budget, the schedule
should spend most of training at peak LR and reserve the last 15–25% for a
linear (or cosine) ramp-down to 0. WSD's `decay_frac=0.2` makes this an
explicit dial.

### 2. Architectural defaults compound; each one removes a different failure mode

The §7.3 ablations + §7.5's revisits showed that RMSNorm, pre-norm, RoPE,
and SwiGLU are not interchangeable — each removes a different failure mode:

- **RMSNorm**: forward-pass scale control + effective-LR scaling. Without
  it, residuals stack across blocks unrescaled and SwiGLU bilinearly
  amplifies the growth → activations overflow in a few hundred steps.
  Recovering with `lr=5e-4` *can* train but loses 0.10 nats vs the
  RMSNorm + `lr=2e-3` combination.
- **Pre-norm vs post-norm**: pre-norm gives gradients an identity path
  through the residual stream regardless of depth. Post-norm renormalizes
  after the residual addition, so gradients from later layers compound a
  `1/RMS(input)` Jacobian factor through every block → vanishing gradients
  in early layers. At our LR, post-norm gets stuck at unigram entropy
  (val ~5.81) with no progress.
- **RoPE**: explicit relative-position inductive bias baked into Q·Kᵀ.
  Without it (NoPE), the model can still infer position from causal-mask
  receptive-field structure, but it pays a *fixed* `+0.07–0.08`-nat
  efficiency cost across the entire training trajectory. The gap is
  expected to grow at longer contexts.
- **SwiGLU**: gated FFN that converges faster early in training. SwiGLU
  has a `+0.21`-nat lead at step 200 vs a param-matched ungated SiLU FFN;
  the gap collapses to ~0 by step 2000 at TinyStories scale, but at
  `(d=1024, L=8)` (§7.5) ReLU² actually regressed by `+0.020` nats — the
  gated FFN earns its keep at scale too.

*Why this matters as a general principle:* "modern Transformer defaults"
are not decorative. They're each fixing a specific instability or
inefficiency, and removing any one of them will cost you nats *or* require
expensive recovery (lower LR, warmup, more steps). The cheapest move is to
keep the consensus stack and spend your tuning budget elsewhere.

### 3. Quality-win mods stack super-linearly (when each removes a different failure)

§7.5 Phase 1 vs Phase 1.5 made this concrete. Four Tier-1 mods (QK-Norm,
weight tying, logit soft-cap, z-loss) measured *individually* summed to
`−0.065` nats at val@1500. Stacked, the same four hit **`−0.082` nats** —
**super-linear**, only visible by stacking.

*Why:* each mod targets a different failure mode, so removing any one of
them lets a different bottleneck dominate.
- QK-Norm bounds Q·Kᵀ dot products → softmax doesn't sharpen onto a single
  key.
- Weight tying frees ~16 M params (at vocab=32k / d=512) → bigger model
  fits in the same memory.
- Logit soft-cap bounds *individual* logit outliers → bf16 softmax doesn't
  lose precision.
- z-loss bounds the *collective* logsumexp → keeps the partition function
  numerically stable.

When you fix only one of these, the others still bottleneck the run. When
you fix all four, the system works in a regime where none of them is the
constraint, and the marginal value of each mod ends up bigger than its
single-mod measurement suggested.

*Practical takeaway:* don't decide which mods to keep based on single-mod
A/Bs alone. If a set of mods are individually small but each fixes a
different failure mode, the right test is to ablate them *out* of the
stacked combination, not *into* a baseline.

### 4. Speedrun mods stratify by what they touch — not all of them transfer

§7.5 ablated several NanoGPT-speedrun favorites at our `(d=1024, L=8, OWT
BPE-32k)` scale. The pattern that emerged:

- **Infrastructure-level mods transferred cleanly.** bf16 autocast, SDPA /
  FlashAttention-2, `torch.compile`, TF32 — all were drop-in wins (`7.32×`
  throughput vs the §7.4 A10G fp32 setup, val drift ≤ +0.025 nats).
- **Optimization-level mods transferred cleanly.** Muon optimizer
  (Newton-Schulz orthogonalization on hidden 2D weights): biggest single
  win in the entire program, `−0.192` nats. WSD schedule: `−0.121` at 2K
  steps and `−0.207` at 45K steps. Longer context (ctx=1024 at bs=24, same
  tokens/step): `−0.059` nats.
- **FFN-shape and skip-topology mods did not transfer.** ReLU² FFN
  regressed by `+0.020` nats vs SwiGLU. Value-embedding skip @ L4 was
  neutral within ±0.005 nats. The speedrun wins these at small scale; we
  lost or tied at our scale.

*Why this stratification:* infra mods don't change the model — only how
fast it runs. Optimization mods change which directions in parameter space
the optimizer spends gradient on, and these benefits seem to scale with
model size. FFN-shape and skip-topology mods are interacting with the
specific *capacity allocation* of the model — what you give up by removing
the gate (ReLU²) or what you gain from a fresh shortcut (value-embed skip)
depends very specifically on `d_model`, `L`, and the mix of tasks in the
data.

*Practical takeaway:* take infrastructure and optimizer wins from any
trustworthy source on faith; ablate every architectural / FFN / skip mod at
*your* scale before keeping it. The assignment caveat about "speedrun mods
may not transfer" landed exactly here — and the ablation took 30 minutes,
which is much cheaper than committing to a wrong final-run choice.

### 5. The LR optimum moves whenever the loss landscape changes — re-anchor cheaply, not exhaustively

§7.5 re-anchored LR four times: after the new infra (Phase 0.5), after the
new arch with QK-Norm (Phase 1.5), after scaling `d` from 512 → 1024
(Phase 2c), and after stacking the WSD + ctx=1024 + Muon mods (Phase 4.5).
Each re-anchor was 3 LRs × 1500–2000 steps ≈ 30 minutes. Two of the four
moved the optimum (Phase 2c shifted it *down* relative to the muP guess;
Phase 4.5 shifted it *back up* by sqrt(2)). Total re-anchor budget: ~3 h,
on a 3-h final run — paid back many times over by not committing to a
wrong LR.

*Why:* changes that move the gradient-noise scale, the per-parameter
update magnitude, or the schedule shape all move the LR optimum. The
optimum is broad (we usually saw `±sqrt(2)` LR multipliers within ~0.02
nats of each other), so the re-anchor sweep can be very cheap, but it
cannot be skipped entirely — *occasionally* the optimum drifts by 2× or
more.

*Practical takeaway:* whenever you change something that could plausibly
shift the LR optimum, schedule a 3-LR mini-sweep before the next
expensive run. The math says it should be cheap: 3 runs at 1/N the steps
of the next big run is 3/N of the budget, where N is usually ≥ 10.

### 6. Memory ceiling sets the upper batch / context / model size; throughput plateaus before optimization quality does

Throughout §7.4 and §7.5, the binding constraint kept shifting:

- §7.4 OWT bs probe: `bs=128` OOM'd at vocab=32k (logits + grad ≈ 8 GiB
  on 22 GiB A10G). `bs=96` was the largest that fit.
- §7.5 Phase 2a bs ceiling probe: `bs=192` OOM'd with the full Tier-1
  stack at `(d=1024, L=8)` because the `(B, T, V)` fp32 logits gradient
  alone is ~6 GiB. `bs=96` was again the only batch that fit across all
  candidate `(d, L)` shapes.
- §7.5 Phase 1.5: `cross_entropy` had to be rewritten on top of
  `F.cross_entropy` + `torch.logsumexp` (instead of an explicit `.float()`
  cast) to free the 3 GiB fp32 intermediate and let bs=96 fit at all.
- §7.5 Phase 5: `torch.compile(mode="reduce-overhead")` triggered CUDA-graph
  double-buffering of the (B, T, V) logits → OOM at our shape. `mode="default"`
  was both safer and ~10% faster.

Meanwhile, throughput saturates much earlier than memory does. GPU
utilization plateaued by `bs ≈ 32–64` on TinyStories (~63 k tok/s on A10G
fp32). On A100 with the full Tier-0 stack: ~220 k tok/s at the §7.4 shape,
~125 k tok/s at the §7.5 final `(d=1024, L=8)` shape. Past the saturation
point, larger bs doesn't speed things up — but it *does* keep improving val
loss (smaller gradient-noise scale, better generalization).

*Why:* the memory budget is dominated by the `(B, T, V)` logits and their
gradient at vocab=32k, which scales linearly in B and T but is a fixed
~`1.6 GB` per slice in fp32. Whatever you save in autocast, intermediate
materialization, or graph compilation buys directly back into batch /
context. Throughput is dominated by GPU occupancy, which saturates as
soon as the matmul tensor cores are full, and you can fill them with
modest batch sizes once the model is wide enough.

*Practical takeaway:* design for memory before throughput. Use the largest
batch (or longest context) that fits with your full mod stack; let the LR
schedule and the optimizer do the rest. If you're throughput-bound, you've
probably under-stacked your infra mods.

### 7. The same loss number means different things on different data

A 2.7-nat gap separated the §7.2 TinyStories cosine 10K (val 1.35) and the
§7.4 OWT cosine 10K (val 4.02) at the same architecture, optimizer, schedule,
and step count. The gap is not a training problem — it's three forces
multiplying together:

1. **Vocabulary size.** Larger vocab spreads probability mass over more
   classes; `log(32000) − log(10000) ≈ 1.16` nats of the gap is
   unavoidable headroom.
2. **Distribution complexity.** TinyStories is curated children's stories
   with templated narrative; OWT is news + forum + recipes + code +
   marketing copy + fiction. A 22 M-param model can near-memorize the
   former and only sketch the latter.
3. **Effective coverage.** TinyStories sees ~60% of its 540 M-token train
   set in 10K steps; OWT sees ~9% of its 2.73 B-token train set. The OWT
   model effectively never re-sees a batch.

*Why this matters:* the same val-loss number describes very different
qualities of model. TS val 1.35 is a fluent child-story generator; OWT val
4.02 is a per-paragraph stylist with topic drift; OWT val 3.19 (§7.5) is a
topic-stable generator with consistent named entities. The metric is
comparable only within a *single* (data, vocab, model-class) pair. When
comparing across datasets, decompose: vocab adds a baseline, complexity
sets the floor, coverage sets the achievable ceiling.

*Practical takeaway:* never compare losses across vocab or data without
explicit vocab-size and tokens-trained context. When designing a benchmark
suite, fix everything except the variable you're studying.

### 8. Track step *and* wallclock from day one

§7.2's batch-size sweep made the principle visible: at equal steps `bs=192`
wins (val 1.520); at equal wallclock `bs=64` is way ahead for the first
~600 s. §7.5's Phase 4d (ctx=1024 vs ctx=256, same tokens/step) decision
needed wallclock to evaluate — the per-step win wasn't enough info on its
own.

You can't reconstruct the wallclock axis after the fact, and you can't
reconstruct step count from wallclock either, so both must be logged
prospectively. CSV with `step,wallclock_sec,split,loss,tokens_per_sec,lr`
costs nothing and is what every ablation in this report reads.

---

## What worked (the keepers)

Concise list of every choice that survived all relevant ablations and
ended up in either the §7.2 / §7.4 / §7.5 final config. Grouped by what
domain they're a choice in.

### Architecture (kept across all 3 finals)

- **Pre-norm RMSNorm** in every Transformer block. Stability + effective-LR
  scaling.
- **RoPE** with `theta=10000`. Sample-efficient relative-position bias.
- **SwiGLU FFN.** Gating earns its keep — small early-training advantage
  on TinyStories, +0.020-nat margin over ReLU² at `(d=1024, L=8)`.
- **Standard `(d_model, num_heads, d_ff = ~2.7 × d_model)`** ratios. We
  didn't ablate head count or `d_ff` ratio; the assignment-baseline
  ratios held up.

### Architecture extensions (§7.5 only — kept after Phase 1 + Phase 1.5)

- **QK-Norm** (per-head RMSNorm of Q and K post-RoPE).
- **Weight tying** between input embedding and LM head, with
  `embed_init_std = 1/√d_model` to compensate for the doubled effective
  scale.
- **Gemma-2 logit soft-cap** with `cap = 30`.
- **PaLM z-loss** with weight `1e-4` on `mean(logsumexp(logits)²)`.

These four stack super-linearly (sum-of-singles `−0.065` < stacked
`−0.082`).

### Optimizer

- **AdamW** for everything in §7.2–§7.4 (`betas=(0.9, 0.95), eps=1e-8,
  weight_decay=0.1`).
- **Muon-mixed** in §7.5: Muon (`muon_lr=0.02`, `momentum=0.95`,
  `ns_steps=5`) on hidden 2D matmul weights; AdamW (`lr=1.414e-3`) on
  embeddings, biases, RMSNorm gains. Biggest single win in the program
  at `−0.192` nats.
- **Gradient clipping at norm 1.0.** Insurance against the rare
  high-magnitude gradient.

### Schedule

- **Cosine warmup-then-decay** for §7.2 / §7.4 (`warmup=500`, `lr_min=lr_max/10`,
  cycle = full step count).
- **WSD** for §7.5 (`warmup=100`, `decay_frac=0.2`, decay-to-0). Cleaner
  to tune than cosine; the explicit `decay_frac` is the single most
  important schedule knob.

### Hyperparameters (transferred broadly)

- **LR sqrt-scaling rule** as a starting point: `lr(bs) = lr_anchor ×
  sqrt(bs / bs_anchor)`. Within `0.003` nats of the actual optimum at
  large bs; needs adjustment at very small bs.
- **μP-style `lr / sqrt(d_new / d_old)`** as a *seed* for size-scaling
  re-anchors. Not exact; the actual optimum depends on schedule and step
  count, so a 3-LR mini-sweep around the seed is essential.

### Infrastructure (§7.5 Tier 0 — free wins, kept by default forever)

- **bf16 autocast** with fp32 master params and optimizer states.
- **`F.scaled_dot_product_attention`** (FA2 backend on A100/H100). Drops
  attention memory from O(T²) to O(T); essential for ctx=1024 at bs=24.
- **`torch.compile(mode="default")`** (not `reduce-overhead` — see "what
  didn't work" below). Inductor kernel fusion + autotune.
- **TF32 + `cudnn.benchmark`.** Free wins as long as input shapes are
  stable.

### Methodology (the meta-stuff that compounds across runs)

- **Cumulative greedy ablation.** Try one mod at a time on top of the
  previous winner; keep what helps, drop what doesn't. *Cumulatively* —
  not at the original baseline — because mods interact.
- **LR re-anchoring after every loss-landscape change.** 30 min of
  3-point sweep, almost always pays for itself.
- **Logging both step and wallclock to CSV from the first run.** Cheap
  insurance against needing to re-run for a different x-axis later.
- **Ratio-banded hardware substitution.** Use the public 2.0–2.5×
  A100→H100 ratio band's three points; report all three. The budget claim
  is robust to which point of the band you pick.

---

## What didn't work (the drops and cautionary tales)

Each of these was tried and explicitly rejected. Some are "doesn't transfer
to our scale" (worth retrying at different scale); some are "actively
counterproductive" (worth flagging for everyone).

### Architecture mods we ablated and dropped

- **No RMSNorm at the optimal LR.** Catastrophic divergence: loss goes
  ~10⁵× in the first 100 steps, NaN by step ~550 forever. *Why:* without
  rescaling, residuals stack across blocks and SwiGLU's bilinearity
  amplifies the growth multiplicatively. *Recoverable* by dropping LR
  4×, but at a 0.10-nat cost.
- **Post-norm at the same LR as pre-norm.** Stuck at val ≈ 5.81 (≈ unigram
  entropy) for the entire 3000-step probe. *Why:* gradients from late
  layers must propagate back through every RMSNorm, and the
  `1/RMS(input)` Jacobian factor compounds → vanishing gradients in early
  layers. Recoverable with smaller LR + long warmup but burns step budget.
- **NoPE (no positional embedding).** Trains stably, just `+0.07–0.08`
  nats above RoPE for the entire trajectory. *Why:* causal masking
  provides some position info for free, but the model has to *learn*
  what RoPE provides analytically — fixed efficiency tax.
- **ReLU² FFN at `(d=1024, L=8)` (§7.5 Phase 4b).** `+0.020` nats vs
  SwiGLU. *Why:* the speedrun wins ReLU² at small scale where the gating
  isn't earning its way; at our scale, the gate's expressivity matters
  enough that removing it costs loss. **Speedrun-mod transfer caveat
  landed exactly here.**
- **Value-embedding skip @ L4 (§7.5 Phase 4c).** Neutral within ±0.005
  nats. *Why (suspected):* at `d=1024 / L=8` the LM head doesn't need
  the surface-feature shortcut that justified the mod at smaller speedrun
  scales. Pre-staged in code but not in the final config.

### Hyperparameter / training-tactic regrets

- **Anchor LR `2e-3` at very small batch (bs=8).** Cost `+0.18` nats vs
  the sqrt-scaled `5e-4`. *Why:* small-batch gradient noise demands a
  smaller LR; the anchor LR is too aggressive. The sqrt-LR rule is most
  important at the small-batch end.
- **`embed_init_std = 1/√d_model` *without* weight tying (§7.5 Phase 1).**
  `+0.041` nats vs the baseline. *Why:* with a small embed std but the LM
  head retaining the standard `Linear` init, initial hidden activations
  are tiny → tiny gradients → slow escape from the under-scaled regime.
  The `1/√d_model` init is the *correct* init *only when tying* (auto-applied
  with `--tie-embeddings`).
- **`compile-mode=reduce-overhead` at our scale (§7.5 Phase 1.5+).**
  CUDA-graph double-buffering of the (B, T, V) logits blew memory at bs=96
  with the full Tier-1 stack. *Why:* CUDA graphs require allocator-pool
  pre-allocation, which competes for the same memory budget already
  reserved for activations. `mode="default"` skips the graph capture but
  still gets kernel fusion — was both safer and ~10% faster on the
  200-step probe. Use `reduce-overhead` only when memory headroom is
  comfortable.
- **Single-mod LR sweeps without re-anchoring.** Tempting to skip the
  re-anchor when "the last sweep said 2e-3"; speedrun folk-wisdom about
  QK-Norm "lifting LR 2–4×" suggested 4–8e-3 should win. At our scale +
  fixed-LR / 1500 steps, it didn't (Phase 1.5 winner was still 2e-3). The
  cheap re-anchor caught the discrepancy in the *opposite* direction
  predicted by folk-wisdom — saved us from committing 4× too high a LR.

### Methodology / experimental-setup mistakes (lessons paid for)

- **Trusting a single sum-of-singles A/B to decide a mod stack.** Sum of
  Tier-1 singles was `−0.065`; stacked was `−0.082`. If we'd used
  single-mod magnitudes to decide which to keep, we'd have undersold the
  combined effect. The fix: stacked-baseline ablation, not single-mod
  baseline.
- **Letting `torch.compile` cache go cold between phases.** §7.5 Phase 5's
  first launch hit 25–40 K tok/s instead of the expected ~105 K because
  the inductor cache at `/tmp/torchinductor_*` had been GC'd between
  Phase 4 and Phase 5. A second launch (warm cache) was at-spec immediately.
  Lesson: warm the cache before any timed run. (We re-tuned Phase 5's
  step budget at launch from 50K to 45K to account for the actual
  measured throughput — a recovery rather than a regression.)
- **Not isolating the wallclock-axis early.** §7.2 batch sweep was tempted
  to report only step-axis numbers; the wallclock-axis story changed the
  conclusions about which batch sizes are useful for which compute budgets.
  Now CSV always logs both.

---

## Methodology — how we made decisions

The §7.5 plan's discipline is what kept the program tractable. A few
specific habits:

### Cheap probes between expensive runs

Every loss-landscape change got a 3-LR mini-sweep at 1/15–1/30 of the next
big run's compute. New infra → re-anchor. New arch → re-anchor. New size →
re-anchor. New mods stack → re-anchor. The total LR-tuning overhead in §7.5
was ~3 h on a 3-h final run; one wrong-LR commit would have cost 3 h.

### Cumulative greedy ablation, not full grid

For each Tier-2 candidate (ReLU², value-embed, WSD, longer ctx), we A/B'd
it on top of the cumulative-best previous config. Two were kept, two were
dropped. We did *not* do a `2⁴` full grid — the search would cost 4× more
and offer little additional info, because the order in which mods are
introduced rarely changes the final keep/drop decision when each one is
strongly positive or clearly neutral.

### Stacked-baseline measurements, not single-mod measurements

When several mods individually look small, decide based on a stacked
ablation (remove from the combined stack), not a single-mod baseline. The
super-linear stacking effect (+0.017 nats above the sum of singles in §7.5
Phase 1.5) is invisible to single-mod measurements.

### Wallclock-aware configuration, not step-count-aware

§7.5 Phase 4d's ctx=1024 / bs=24 swap kept tokens/step constant but added
~30% per-step time. Worth it because the −0.059 per-step nats translated to
−0.059 *per-wallclock* nats inside the fixed compute budget. The decision
needed wallclock data, not just step-axis curves.

### Hardware substitution should be explicit and ratio-banded

§7.5's A100 → H100 conversion uses three points of the public 2.0–2.5×
bf16-dense ratio band. The final wallclock fits inside the 1.5 h H100
envelope under all three points (1.176 / 1.354 / 1.470 h). Reporting the
ratio band makes the budget claim robust to which midpoint a reader
prefers; reporting only the canonical 2.17× would invite "what if it's
actually 2.5×?" pushback.

### Wallclock-budgeted training as a safety net

The `--max-wallclock-sec` flag in `scripts/train_lm.py` writes one final
val + checkpoint and breaks the loop if the cap is exceeded. Implemented
during §7.5 Phase 5; never actually triggered (the run finished naturally
at step 45K), but is a useful safety belt for any compute-capped run.

---

## TL;DR

The seven principles we'd put in any future LM-from-scratch project,
ranked by how much each one mattered to the §7 program:

1. **Late-stage LR decay finds the sharp minimum.** Reserve 15–25% of
   training for a clean ramp-down to 0; the win is bigger than most
   architectural mods.
2. **Don't strip default architectural choices.** RMSNorm + pre-norm +
   RoPE + SwiGLU each fix a different problem; removing any one is
   measurably worse at the same compute.
3. **Quality mods stack super-linearly when each fixes a different
   failure.** Decide based on stacked ablations, not single-mod A/Bs.
4. **Speedrun mods stratify.** Trust infra and optimizer mods; ablate
   FFN-shape and skip-topology mods at *your* scale before committing.
5. **Re-anchor LR after every loss-landscape change.** 30 min of cheap
   probe; almost always pays back.
6. **Memory ceiling is the upper bound; throughput plateaus first.** Use
   the largest batch / context that fits with your full mod stack.
7. **The same loss number means different things on different data.**
   Decompose by vocab + complexity + coverage when comparing across
   datasets.

If you only do *three* of those: do (1) — pick a schedule with explicit
late decay, (2) — keep the modern Transformer defaults, and (5) —
re-anchor LR cheaply between expensive runs.
