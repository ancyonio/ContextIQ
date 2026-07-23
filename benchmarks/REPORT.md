# ContextIQ Benchmark Report

> Reproducible, self-contained benchmark of ContextIQ (`tokengraph`) — token-efficient code context retrieval, cross-language test discovery, and hallucination guarding. Every input is content-hashed (see `benchmarks/MANIFEST.json`) so a third party can verify they ran the same dataset.

- Generated: 2026-07-23 19:25 UTC
- Dataset hash: `0443130df021f6f2d1a36a65f07de28a260d9dc264d044996ef2319e31eedbb3` (56 files)
- Extractor: `3:bash,c,cpp,csharp,dart,erlang,go,haskell,java,javascript,julia,kotlin,lua,nim,ocaml,perl,php,powershell,r,ruby,rust,scala,solidity,swift,tsx,typescript` · Python 3.11.13

## 1. Retrieval quality

| Metric | Value |
|---|--:|
| Queries | 96 |
| Corpora | 4 |
| Recall@5 | 0.989 |
| Symbol recall | 0.745 |
| Answerable rate | 0.584 |
| Irrelevant-token ratio (waste) | 0.673 |

### Per-corpus

| Corpus | n | Recall@5 | Symbol recall | Answerable | Waste |
|---|--:|--:|--:|--:|--:|
| retrieval_tasks | 42 | 0.976 | 0.643 | 0.548 | 0.522 |
| gosvc | 18 | 1.0 | 0.778 | 0.611 | 0.836 |
| pyshop | 18 | 1.0 | 0.972 | 0.667 | 0.718 |
| tsapi | 18 | 1.0 | 0.722 | 0.556 | 0.817 |

## 2. Test discovery (implementation ↔ test mapping)

| Metric | Value |
|---|--:|
| Precision | 1.0 |
| Recall | 0.9 |
| **F1** | **0.9474** |
| hit@1 | 0.9 |
| Gold pairs | 10 |
| TP / FP / FN | 9 / 0 / 1 |

Measured on `benchmarks/testmap/` (Python/Go/TypeScript/Java, labeled in `pairs.json`). The naming heuristic scores perfect precision; the single miss is a deliberately name-divergent pair that only the call graph links (recovered at symbol granularity by `get_test_map`).

## Reproduce

```bash
tokengraph benchmark --all          # retrieval quality across all corpora
tokengraph test-map --benchmark     # test-discovery precision/recall/F1
tokengraph publish-benchmark        # regenerate this report + manifest
```

Verify the dataset is byte-identical by re-hashing `benchmarks/` and comparing `dataset_hash` in `benchmarks/MANIFEST.json`.
