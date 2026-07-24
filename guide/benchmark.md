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
