---
title: What is ContextIQ?
description: "A local, single-file AST code graph that gives AI coding agents token-efficient context — retrieve, validate, judge, and verify, all offline."
head:
  - - meta
    - property: og:title
      content: "ContextIQ — token-efficient context for AI coding agents"
  - - meta
    - property: og:description
      content: "Index your repo into a local SQLite code graph and serve a token-budgeted context pack of only the symbols relevant to a task."
---

# What is ContextIQ?

**ContextIQ** is a local, single-file **AST code graph** that gives AI coding
agents token-efficient context. Instead of letting an agent read whole files
(and re-read them every session), ContextIQ indexes your repository into a local
SQLite graph of symbols + call / import / inheritance edges, then serves a
**token-budgeted "context pack"** of only the symbols relevant to a task.

> **Naming:** the product is **ContextIQ**. Its CLI is `tokengraph_all.py`, and
> the MCP server registers under the id `tokengraph` — those identifiers are kept
> stable so existing wiring keeps working.

## The core idea

| Without ContextIQ | With ContextIQ |
| --- | --- |
| Agent opens whole files to find relevant code | Agent asks for a task and gets only the relevant symbols |
| Re-reads the codebase every session | Graph persists in `.tokengraph/graph.db` across sessions |
| Risk of stale understanding after edits | Graph re-indexes incrementally before every query |
| Savings are invisible / unprovable | Every retrieval appends to a ledger; `gain` reports tokens + dollars saved |

A context pack contains: relevant symbols (full body when small, signature when
large), their callers / callees / base classes as signatures, matching indexed
source chunks, and a list of anything **dropped** for budget so the agent can
request it by name. Near-duplicate pieces are removed automatically.

## What makes it different

- **One file, zero required dependencies** to run the CLI ([`tokengraph_all.py`](../tokengraph_all.py)).
- **Deep parsing for 25+ languages** with a full call / import / inheritance
  graph, plus regex indexing for 30+ more. See [Languages](./languages.md).
- **Model-agnostic and offline** — it emits context packs, never calls an LLM,
  so it works with any model, cloud or local. See [Local LLMs](./local-llms.md).
- **The full loop, not just retrieval** — `ask` → `validate` → `judge` →
  `verify` closes the gap from *finding* context to *trusting* the answer.
- **Grounded generation** — detect house `conventions`, `scaffold`
  convention-matched files, and `verify-plan` before you act.
- **Provable savings** — a realized-savings ledger and a `gain` dashboard.

## Next steps

- **Start here:** [Quick start](./quick-start.md) — install and run the real workflow in minutes.
- **Daily driver:** [Retrieval](./retrieval.md) · [Validate](./validate.md) · [Judge](./judge.md) · [Verify](./verify.md)
- **Wire your editor:** [MCP server](./mcp.md)
- **Reference:** [CLI](./cli.md) · [When to use what](./when-to-use.md)
- **Proof:** [Savings](./savings.md) · [Benchmark](./benchmark.md)
