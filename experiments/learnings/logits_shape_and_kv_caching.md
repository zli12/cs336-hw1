# Transformer LM logits shape, per-position predictions, and KV caching

Context: came up while reviewing `cs336_basics/decoding.py:111-114`:

```python
# logits shape: (1, seq_len, vocab_size)
logits = model(x)
# Next-token distribution is taken from the final position only.
next_token_logits = logits[0, -1]
```

Three related questions:
1. What's the shape of `logits` and why take `[0, -1]`?
2. Why does the model predict at every position simultaneously?
3. Is KV caching only useful at inference, or also at training?

---

## 1. Shape of `logits` and why `[0, -1]`

`logits.shape == (1, seq_len, vocab_size)`.

- `1` is the batch dim — `decode` does `x = ...unsqueeze(0)` so we always have batch size 1.
- `seq_len` is the length of the current context we just fed in (capped at `model.context_length`).
- `vocab_size` is the model's vocabulary; each position carries a length-`vocab_size` vector of unnormalized scores.

`TransformerLM.forward` produces this shape because the final layer is a per-position linear projection:

```python
x = self.token_embeddings(in_indices)   # (1, seq_len, d_model)
for layer in self.layers:
    x = layer(x)                        # (1, seq_len, d_model)
x = self.ln_final(x)                    # (1, seq_len, d_model)
return self.lm_head(x)                  # (1, seq_len, vocab_size)
```

Indexing `logits[0, -1]`:
- `0` strips the batch dim → `(seq_len, vocab_size)`.
- `-1` selects the **final** time step → `(vocab_size,)`, which is exactly the 1-D shape that `sample_next_token` requires (it asserts `ndim == 1`).

The deeper reason for `-1`: with causal masking, position `t`'s output is conditioned on tokens `0..t`. We've already chosen every token in `context`; the only *new* prediction we need is the one conditioned on the **full** current context — which lives at the final position.

---

## 2. The model predicts at every position simultaneously

This is intrinsic to causal-LM training, not an inference-time choice.

### Worked example

Vocab of 6 tokens: `0=the, 1=cat, 2=sat, 3=on, 4=mat, 5=<eos>`. Feed in **"the cat sat on"**:

```python
context = [0, 1, 2, 3]
x = torch.tensor(context).unsqueeze(0)   # shape (1, 4)
logits = model(x)                         # shape (1, 4, 6)
```

Each output position `i` is interpreted as "predict the next token after the prefix ending at `i`":

| pos i | input token | hidden state h_i depends on | logits[0, i] predicts             |
|-------|-------------|-----------------------------|-----------------------------------|
| 0     | "the"       | "the"                       | P(· \| the)                       |
| 1     | "cat"       | "the", "cat"                | P(· \| the, cat)                  |
| 2     | "sat"       | "the", "cat", "sat"         | P(· \| the, cat, sat)             |
| 3     | "on"        | "the", "cat", "sat", "on"   | P(· \| the, cat, sat, on)         |

### Why this works: the causal mask

Attention at position `i` is masked so it can only attend to positions `0..i`:

```
          attends to →
           t0  t1  t2  t3
   t0  [   ✓   .   .   .  ]   ← position 0 sees only "the"
   t1  [   ✓   ✓   .   .  ]   ← position 1 sees "the cat"
   t2  [   ✓   ✓   ✓   .  ]   ← position 2 sees "the cat sat"
   t3  [   ✓   ✓   ✓   ✓  ]   ← position 3 sees full prefix
```

Without this mask, position 1 would see "sat" and "on" and its prediction `P(· | the, cat)` would be cheating. The mask is what makes all `seq_len` predictions legitimate, parallel, independent next-token predictions.

### Why this shape exists: training efficiency

Loss is computed in parallel across all positions against a left-shifted target:

```
input  = [0, 1, 2, 3]   = the, cat, sat, on
target = [1, 2, 3, 4]   = cat, sat, on,  mat
```

| pos | predicted distribution         | target id | target token |
|-----|--------------------------------|-----------|--------------|
| 0   | P(· \| the)                    | 1         | "cat"        |
| 1   | P(· \| the, cat)               | 2         | "sat"        |
| 2   | P(· \| the, cat, sat)          | 3         | "on"         |
| 3   | P(· \| the, cat, sat, on)      | 4         | "mat"        |

**One forward pass yields `seq_len` supervision signals.** Multiplied by batch size, that's `B × seq_len` next-token gradient signals per training step. This parallelism is the reason transformer pretraining is feasible at scale.

### Common confusion to flag

It's tempting to read `logits[0, i]` as "what the model thinks comes after just `t_i` viewed alone." That's wrong. It's "what comes after the prefix `t_0..t_i`," because attention has mixed information from all earlier positions. **The position index in the logits tensor refers to how much prefix the prediction is conditioned on, not which input token "produced" it.**

### What inference throws away

In `decode`, only the last position is new info:

```
step k:   input length k+1, get k+1 logits, use 1, discard k
step k+1: input length k+2, get k+2 logits, use 1, discard k+1
...
```

Total work is O(n²) in sequence length even though each step only needs one new prediction. KV caching exists to remove this redundancy.

---

## 3. KV caching: inference only, not training

Short answer: **KV caching is inference-only.** At training it provides no benefit and would actively be the wrong thing to do.

### Why training doesn't need it

KV caching avoids re-doing work across **autoregressive decoding steps**. That step structure does not exist at training:

```
training step:
  input:  [t0, t1, t2, t3]   # whole sequence at once
  forward → logits shape (1, 4, vocab_size)
  loss = CE against [t1, t2, t3, t4]
  backward, update weights
```

Inside that single forward pass, K and V for every position are already computed exactly once. The parallelism over positions (via the causal mask) does the same job a KV cache does at inference.

Compare inference, where without a cache each step recomputes K, V for the entire prefix:

```
inference:
  step 0: forward on [t0, t1]              → K,V for 0..1 computed
  step 1: forward on [t0, t1, t2]          → K,V for 0..1 RECOMPUTED + new K,V for 2
  step 2: forward on [t0, t1, t2, t3]      → K,V for 0..2 RECOMPUTED + new K,V for 3
  ...
```

That redundancy is what the cache eliminates. At training, that redundancy doesn't exist in the first place.

### Why caching would actively be wrong at training

1. **Weights are changing.** A cache is only valid while K, V projection weights are fixed. After every optimizer step those weights move, so any cached K, V from a prior step is stale. Nothing meaningful to cache *across* training steps.
2. **Backward needs K, V in the autograd graph.** Gradients must flow through attention back into the K, V projections (and through them into the input embeddings). A cache by definition is a frozen tensor not connected to the live graph — using it would silently drop gradients. Inference doesn't care because there's no backward pass.

### Adjacent ideas that are *not* "KV cache at training"

- **Transformer-XL / segment-level recurrence.** When training on long documents in chunks, you can feed in **detached** hidden states from the previous chunk as extra context. It reuses past activations, but they're explicitly `.detach()`'d and it's a specific architectural choice, not the standard KV cache.
- **RLHF / SFT rollouts.** RL pipelines *generate* training data via inference (which uses a KV cache normally). The cache helps the data-collection phase, but the gradient-update phase still does a single forward pass with no cache.
- **Activation checkpointing.** The *opposite* of caching — intentionally throw away activations during forward and recompute them during backward to save memory. Common in training; unrelated to KV caching despite the surface similarity.

---

## TL;DR

- `logits` is `(1, seq_len, vocab_size)`. Position `i` predicts the next token given prefix `t_0..t_i`.
- All `seq_len` predictions are produced in one forward pass; the causal mask makes that legitimate.
- This shape is dictated by **training**, where every position contributes a supervision signal in parallel.
- At decode time, only position `-1` is new info; the rest is redundant work.
- KV caching collapses that redundant work for inference. It plays no role at training — there's no cross-step redundancy to amortize, weights are moving, and gradients need to flow through K, V every step.
