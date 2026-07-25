---
title: Benchmark & evidence
description: "A reproducible, multi-repo hallucination-reduction benchmark plus deterministic, hash-grounded evidence packs for CI and audit."
head:
  - - meta
    - property: og:title
      content: "ContextIQ benchmark & evidence"
---

# Benchmark & evidence

ContextIQ's claims are **measurable and reproducible**, not marketing.

## Published results (v1.0.0)

The v1.0.0 benchmark run is archived with a permanent DOI:
[10.5281/zenodo.21535772](https://doi.org/10.5281/zenodo.21535772). Every input
is content-hashed so a third party can confirm they ran the same dataset.

- **Generated:** 2026-07-24 14:25 UTC
- **Dataset hash:** `654a8300befab41d4396c8fee5d318d233a2debc52212a17b22d8cceca1f08b0` (63 files)
- **Environment:** Python 3.11.13 · 26-language deep-parse extractor
- **Source of record:** [`benchmarks/REPORT.md`](https://github.com/ancyonio/ContextIQ/blob/main/benchmarks/REPORT.md) + [`benchmarks/MANIFEST.json`](https://github.com/ancyonio/ContextIQ/blob/main/benchmarks/MANIFEST.json)

### Retrieval quality

| Metric | Value |
|---|--:|
| Queries | 96 |
| Corpora | 4 |
| Recall@5 | 0.989 |
| Symbol recall | 0.735 |
| Answerable rate | 0.604 |
| Irrelevant-token ratio (waste) | 0.708 |

**Per-corpus** (Python / Go / TypeScript):

| Corpus | n | Recall@5 | Symbol recall | Answerable | Waste |
|---|--:|--:|--:|--:|--:|
| retrieval_tasks | 42 | 0.976 | 0.667 | 0.595 | 0.554 |
| gosvc | 18 | 1.0 | 0.722 | 0.556 | 0.853 |
| pyshop | 18 | 1.0 | 0.917 | 0.722 | 0.774 |
| tsapi | 18 | 1.0 | 0.722 | 0.556 | 0.854 |

### Test discovery (implementation ↔ test mapping)

| Metric | Value |
|---|--:|
| Precision | 1.0 |
| Recall | 0.9 |
| **F1** | **0.9474** |
| hit@1 | 0.9 |
| Gold pairs | 10 |
| TP / FP / FN | 9 / 0 / 1 |

Measured on `benchmarks/testmap/` (Python / Go / TypeScript / Java, labeled in
`pairs.json`). The naming heuristic scores perfect precision; the single miss is
a deliberately name-divergent pair that only the call graph links (recovered at
symbol granularity by `get_test_map`).

### Reproduce

```bash
tokengraph benchmark --all          # retrieval quality across all corpora
tokengraph test-map --benchmark     # test-discovery precision / recall / F1
tokengraph publish-benchmark        # regenerate REPORT.md + MANIFEST.json
```

Verify the dataset is byte-identical by re-hashing `benchmarks/` and comparing
`dataset_hash` in `benchmarks/MANIFEST.json`.

## Hallucination-reduction benchmark

```bash
tokengraph hallucination
```

A multi-repo, reproducible **codebase-fact grounding** benchmark: it measures how
much the guard reduces fabricated files / symbols / imports across real
repositories. The published run is archived with a DOI (see the project README
badge).

## Quantify the guard on your repo

```bash
tokengraph grounding
```

Reports fabrications **caught** vs. real references **flagged** — the precision /
recall of the verify layer on your own code.

## Test-map accuracy

```bash
tokengraph test-map --benchmark
```

Scores the implementation ↔ test mapping with precision / recall / F1 / hit@1 on
a labeled corpus.

## Evidence packs (audit / CI)

```bash
tokengraph evidence "add retry logic"
```

Produces a **deterministic, hash-grounded** JSON pack for a task — same inputs,
same bytes — so a reviewer or CI job can verify exactly what context an agent was
given. All output is secret-scanned.

## Next steps

- The guard it measures: [Verify](./verify.md) · [Judge](./judge.md)
- The savings side of the story: [Savings](./savings.md)
