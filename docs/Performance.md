# Performance at scale

Reproducible large-repo numbers from the synthetic harness:

```bash
python tokengraph_all.py benchmark --scale 20000        # index time + pack p50/p95
```

The harness generates N two-symbol modules, indexes them cold, then times pack
assembly over a spread of queries. Absolute milliseconds are machine-dependent —
the **shape** of the curves is the point.

## Measured (deterministic hash backend, exact vector cosine, 6000-token budget)

| Files | Symbols | Cold index | Index / file | Pack mean | Pack p95 |
|--:|--:|--:|--:|--:|--:|
| 500 | 1,500 | 1.77 s | 3.54 ms | 71 ms | 86 ms |
| 2,000 | 6,000 | 5.37 s | 2.68 ms | 257 ms | 270 ms |
| 8,000 | 24,000 | 22.7 s | 2.84 ms | 1,045 ms | 1,085 ms |
| 20,000 | 60,000 | 58.1 s | 2.91 ms | 2,529 ms | 2,553 ms |

## What the curves say

- **Indexing is linear** at ~2.9 ms/file and holds flat across two orders of
  magnitude. Extrapolated: ~100k symbols (≈34k files) cold-indexes in ≈100 s.
  This is a **cold** build; a warm re-index is an mtime+size `stat()` sweep
  (the freshen-on-query fast path), so steady-state cost is far lower.
- **Pack latency grows with the vector count** because these numbers use
  **exact** cosine over every symbol vector. That is the *pessimistic* bound:
  it is exactly the case the `auto` vector backend hands to the `hnswlib` ANN
  index above `TOKENGRAPH_ANN_THRESHOLD` (5000 vectors) when `[ann]` is
  installed — turning the linear pack-latency growth sub-linear. Install
  `pip install 'contextiq[all]'` (or `[ann]`) to get that path.

## Caveats

- Run on the reproducible hash embedding (what CI uses); a neural backend adds
  a one-time embed cost at index time but does not change the retrieval shape.
- Synthetic modules are uniform; a real repo's hub structure shifts absolute
  numbers but not the linear-index / vector-bound-pack behavior.
- Retrieval also benefits from the neighbour relevance floor
  (`TOKENGRAPH_NEIGHBOR_FLOOR`, default 0.06), which roughly halved pack
  latency on the fixture corpus — see the benchmark section in the README.
