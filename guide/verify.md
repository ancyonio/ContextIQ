---
title: Verify
description: "Catch hallucinated files, symbols, and imports in AI output before you run it — with did-you-mean suggestions and an exit-1 gate."
head:
  - - meta
    - property: og:title
      content: "ContextIQ verify — the hallucination guard"
---

# Verify

`verify` audits an AI answer for **references that don't exist** in your repo —
fabricated files, symbols, and imports — and exits non-zero if it finds any.

## Verify an answer

```bash
tokengraph verify --answer-file response.txt
```

Every file / symbol / import the answer names is checked against the graph.
Anything missing is flagged, with **did-you-mean** suggestions for near matches.

## The verification family

| Command | Checks |
| --- | --- |
| `verify` | A free-form answer for fabricated files / symbols |
| `verify-output` | AI-generated **code** for fabricated files / symbols / local imports |
| `verify-plan` | A **plan's** refs + blast radius *before* you act |
| `grounding` | Quantifies the guard: fabrications caught vs. real refs flagged |

## In a plan-first workflow

```bash
# 1. Check the plan references real code and understand the blast radius
tokengraph verify-plan --answer-file plan.md

# 2. After generation, audit the produced code
tokengraph verify-output --answer-file patch.diff
```

## Why it matters

Hallucinated imports and symbols are the most common way AI-generated changes
fail. Verify turns that from a runtime surprise into a **pre-flight exit code** —
easy to wire into a hook or CI.

## Next steps

- Generate safely from the start: [Conventions & scaffolding](./conventions.md)
- See the measured guard: [Benchmark](./benchmark.md)
