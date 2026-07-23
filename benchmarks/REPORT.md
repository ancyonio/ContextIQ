# ContextIQ Benchmark Report

> Reproducible, self-contained benchmark of ContextIQ (`tokengraph`) — token-efficient code context retrieval, cross-language test discovery, and hallucination guarding. Every input is content-hashed (see `benchmarks/MANIFEST.json`) so a third party can verify they ran the same dataset.

- Generated: 2026-07-23 20:17 UTC
- Dataset hash: `0443130df021f6f2d1a36a65f07de28a260d9dc264d044996ef2319e31eedbb3` (56 files)
- Extractor: `3:bash,c,cpp,csharp,dart,erlang,go,haskell,java,javascript,julia,kotlin,lua,nim,ocaml,perl,php,powershell,r,ruby,rust,scala,solidity,swift,tsx,typescript` · Python 3.11.13

## 1. Retrieval quality

| Metric | Value |
|---|--:|
| Queries | 96 |
| Corpora | 4 |
| Recall@5 | 1.0 |
| Symbol recall | 0.745 |
| Answerable rate | 0.584 |
| Irrelevant-token ratio (waste) | 0.672 |

### Per-corpus

| Corpus | n | Recall@5 | Symbol recall | Answerable | Waste |
|---|--:|--:|--:|--:|--:|
| retrieval_tasks | 42 | 1.0 | 0.643 | 0.548 | 0.52 |
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

## 3. Hallucination guard

```json
{
  "deterministic": true,
  "facts_total": 218,
  "hallucination_reduction_pct": null,
  "mean_grounding_coverage_pct": 96.79,
  "mean_guard_catch_pct": 50.46,
  "mean_guard_specificity_pct": 95.87,
  "measured": true,
  "methodology": "deterministic structural measurement (no LLM). Measured per repo partition: grounding coverage, guard catch rate, guard specificity. Unguarded fact share = (1-coverage)*(1-catch).",
  "ok": true,
  "per_repo": [
    {
      "facts": 40,
      "grounding_coverage_pct": 100.0,
      "guard_catch_pct": 30.0,
      "guard_specificity_pct": 87.5,
      "repo": "(root)",
      "unguarded_fact_share_pct": 0.0
    },
    {
      "facts": 40,
      "grounding_coverage_pct": 97.5,
      "guard_catch_pct": 10.0,
      "guard_specificity_pct": 100.0,
      "repo": ".github",
      "unguarded_fact_share_pct": 2.25
    },
    {
      "facts": 8,
      "grounding_coverage_pct": 100.0,
      "guard_catch_pct": 50.0,
      "guard_specificity_pct": 100.0,
      "repo": ".prompts",
      "unguarded_fact_share_pct": 0.0
    },
    {
      "facts": 40,
      "grounding_coverage_pct": 97.5,
      "guard_catch_pct": 90.0,
      "guard_specificity_pct": 100.0,
      "repo": "benchmarks",
      "unguarded_fact_share_pct": 0.25
    },
    {
      "facts": 40,
      "grounding_coverage_pct": 100.0,
      "guard_catch_pct": 10.0,
      "guard_specificity_pct": 90.0,
      "repo": "docs",
      "unguarded_fact_share_pct": 0.0
    },
    {
      "facts": 40,
      "grounding_coverage_pct": 92.5,
      "guard_catch_pct": 100.0,
      "guard_specificity_pct": 100.0,
      "repo": "tests",
      "unguarded_fact_share_pct": 0.0
    },
    {
      "facts": 10,
      "grounding_coverage_pct": 80.0,
      "guard_catch_pct": 100.0,
      "guard_specificity_pct": 100.0,
      "repo": "tools",
      "unguarded_fact_share_pct": 0.0
    }
  ],
  "projection": {
    "available": false,
    "why": "no ungrounded-fabrication baseline supplied. ContextIQ cannot observe how often an un-grounded agent fabricates; pass baseline_per_100 (with baseline_source) measured on your own agent and model to get a projected reduction. The measured figures above stand on their own."
  },
  "repos": 7,
  "summary": "measured across 7 repo-partition(s), 218 facts: grounding coverage 96.79%, guard catch 50.46%, guard specificity 95.87%; 0.46% of facts are both ungroundable and unguarded",
  "unguarded_fact_share_pct": 0.46,
  "unguarded_spread_pct": [
    0.0,
    2.25
  ]
}
```

## Reproduce

```bash
tokengraph benchmark --all          # retrieval quality across all corpora
tokengraph test-map --benchmark     # test-discovery precision/recall/F1
tokengraph publish-benchmark        # regenerate this report + manifest
```

Verify the dataset is byte-identical by re-hashing `benchmarks/` and comparing `dataset_hash` in `benchmarks/MANIFEST.json`.
