# `BPETrainer.train()` Internal Walkthrough

This walkthrough focuses on the two core internals:
- how pair lookups are constructed (`pair_counts` and `pair_to_sequence_indices`)
- how one merge updates those lookups incrementally

It mirrors the implementation in `cs336_basics/bpe_trainer.py`.

## Key data structures

- `token_sequences`: list of mutable token-id sequences (one per unique pretoken)
- `sequence_frequencies`: aligned frequency for each sequence index
- `pair_counts[(a, b)]`: weighted global adjacent-pair count
- `pair_to_sequence_indices[(a, b)]`: set of sequence indices that currently contain pair `(a, b)`

`pair_counts` tells you "what to merge next", while `pair_to_sequence_indices` tells you "what to recompute when that pair is merged".

## Tiny concrete example

Assume pretokenization produced these unique sequences:

```text
s0: " the"      -> [32,116,104,101], freq=3
s1: " theater"  -> [32,116,104,101,97,116,101,114], freq=1
s2: " in"       -> [32,105,110], freq=2
```

Byte IDs shown above: `32=' '`, `116='t'`, `104='h'`, `101='e'`, `105='i'`, `110='n'`.

## 1) Initial pair lookup construction

For each sequence, the trainer builds local adjacent-pair counts:

```text
s0 pairs: (32,116) (116,104) (104,101)                 each x1, weighted by freq=3
s1 pairs: (32,116) (116,104) (104,101) (101,97) ...    each x1, weighted by freq=1
s2 pairs: (32,105) (105,110)                            each x1, weighted by freq=2
```

Global weighted counts:

```text
pair_counts:
(32,116): 4
(116,104): 4
(104,101): 4
(32,105): 2
(105,110): 2
(101,97): 1
(97,116): 1
(116,101): 1
(101,114): 1
```

Reverse lookup:

```text
pair_to_sequence_indices:
(116,104) -> {0,1}
(104,101) -> {0,1}
(32,105)  -> {2}
...
```

This reverse map is the speed-up: if best pair is `(116,104)`, only sequence indices `{0,1}` are touched.

## 2) One merge step in detail

Suppose best pair is `(116,104)` (`'t' + 'h'`), and a new token ID `256` is created.

Affected indices from lookup:

```text
affected_sequence_indices = pair_to_sequence_indices[(116,104)] = {0,1}
```

### Sequence `s0` update

Before/after:

```text
before: [32,116,104,101]
after:  [32,256,101]
```

Old local pairs:

```text
(32,116), (116,104), (104,101)
```

New local pairs:

```text
(32,256), (256,101)
```

Delta application (weighted by `freq=3`):

```text
(32,116): -3
(116,104): -3
(104,101): -3
(32,256): +3
(256,101): +3
```

`pair_to_sequence_indices` is updated in lockstep:
- remove `0` from pairs that disappeared
- add `0` to newly introduced pairs

### Sequence `s1` update

Before/after:

```text
before: [32,116,104,101,97,116,101,114]
after:  [32,256,101,97,116,101,114]
```

Same local transition pattern, weighted by `freq=1`.

## 3) Why this is incremental (and fast)

The trainer does not rebuild pair stats over the whole corpus each iteration. It:

1. pops max pair from heap
2. fetches affected sequence indices from `pair_to_sequence_indices`
3. recomputes local old/new pair counts for those sequences only
4. updates global counts by deltas
5. refreshes heap entries only for changed pairs
6. removes merged-away best pair from active maps

This keeps each merge step localized.

## 4) Non-overlapping replacement behavior

Merges are non-overlapping because when a match is found, pointer advances by 2.

Example for merging `(a,a)` in `[a,a,a]`:

```text
[a,a,a] -> [aa,a]
```

not `[aa]`.

## Optional: inspect live internals with the script

Run:

```bash
uv run python scripts/visualize_bpe_training.py --show-internals --num-merges 8
```

This prints:
- top `pair_counts` before each merge
- `affected_sequence_indices` from pair lookup
- top `pair_counts` after applying the merge
