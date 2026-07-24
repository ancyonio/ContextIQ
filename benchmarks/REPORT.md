# ContextIQ Benchmark Report

> Reproducible, self-contained benchmark of ContextIQ (`tokengraph`) — token-efficient code context retrieval, cross-language test discovery, and hallucination guarding. Every input is content-hashed (see `benchmarks/MANIFEST.json`) so a third party can verify they ran the same dataset.

- Generated: 2026-07-24 14:25 UTC
- Dataset hash: `654a8300befab41d4396c8fee5d318d233a2debc52212a17b22d8cceca1f08b0` (63 files)
- Extractor: `3:bash,c,cpp,csharp,dart,erlang,go,haskell,java,javascript,julia,kotlin,lua,nim,ocaml,perl,php,powershell,r,ruby,rust,scala,solidity,swift,tsx,typescript` · Python 3.11.13

## 1. Retrieval quality

| Metric | Value |
|---|--:|
| Queries | 96 |
| Corpora | 4 |
| Recall@5 | 0.989 |
| Symbol recall | 0.735 |
| Answerable rate | 0.604 |
| Irrelevant-token ratio (waste) | 0.708 |

### Per-corpus

| Corpus | n | Recall@5 | Symbol recall | Answerable | Waste |
|---|--:|--:|--:|--:|--:|
| retrieval_tasks | 42 | 0.976 | 0.667 | 0.595 | 0.554 |
| gosvc | 18 | 1.0 | 0.722 | 0.556 | 0.853 |
| pyshop | 18 | 1.0 | 0.917 | 0.722 | 0.774 |
| tsapi | 18 | 1.0 | 0.722 | 0.556 | 0.854 |

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
