---
title: Quick start
description: "Install ContextIQ, index your repo, then run the real workflow: ask, validate, judge, verify — all local, no API key."
head:
  - - meta
    - property: og:title
      content: "ContextIQ Quick Start — ask, validate, judge, verify"
  - - meta
    - property: og:description
      content: "Install once, build the graph, ask a real question, gate coverage, and check the answer is grounded."
---

# Quick start

The fastest path to the real ContextIQ workflow. Everything here runs **locally**
and needs **no API key**.

## 1. Install

```bash
pipx install "contextiq[all]"      # isolated global install (recommended)
# or
pip install "contextiq[all]"       # into the current environment
```

You can also run the single file directly with **zero dependencies** for the
core CLI:

```bash
python tokengraph_all.py --help
```

::: tip
Both invocations are equivalent. This guide uses the `tokengraph` console script;
substitute `python tokengraph_all.py` if you're running from source.
:::

## 2. Build the graph

```bash
tokengraph index
```

This parses your repo into `.tokengraph/graph.db`. You rarely run this again —
the graph **auto-refreshes on every query**, so it never goes stale.

## 3. Get a context pack for a task

```bash
tokengraph context "add retry logic to the http client" -b 6000
```

You get a token-budgeted pack of only the relevant symbols — paste it into your
AI assistant instead of dumping whole files. The pack prints to stdout; save it
to a file with `-o` when you want to reuse it (e.g. to judge the answer later):

```bash
tokengraph context "add retry logic to the http client" -b 6000 -o context.md
```

## 4. Ask, with intent + coverage + risk

```bash
tokengraph ask "explain the auth flow"
```

`ask` returns a focused pack plus metadata: what it thinks the task *is*, how
well the context *covers* it, and the *risk* of acting on it.

## 5. Validate the coverage (CI-friendly gate)

```bash
tokengraph validate "auth login token" --min-coverage 0.6
```

Exits non-zero if coverage is below the threshold — wire it into a hook so an
agent never acts on thin context.

## 6. Judge whether the answer is grounded

Save your assistant's answer, then score it against the context you gave it:

```bash
tokengraph judge --answer-file response.txt --context-file context.md
```

Outputs a 0.0–1.0 grounding score with PASS/FAIL.

## 7. Verify — catch fabricated files & symbols

```bash
tokengraph verify --answer-file response.txt
```

Flags any file / symbol / import the answer references that **doesn't exist** in
your repo (with did-you-mean suggestions). Exits non-zero if any are found.

## The loop, in one line

```
ask  →  validate  →  (AI answers)  →  judge  →  verify
```

## Next steps

- Wire it into your editor once: [MCP server](./mcp.md)
- Learn each stage: [Retrieval](./retrieval.md) · [Validate](./validate.md) · [Judge](./judge.md) · [Verify](./verify.md)
- Generate code safely: [Conventions & scaffolding](./conventions.md)
- Prove the savings: [Savings dashboard](./savings.md)
