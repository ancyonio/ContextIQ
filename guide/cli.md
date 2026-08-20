---
title: CLI reference
description: "Every ContextIQ command grouped by purpose: retrieval, graph navigation, trust gates, grounded generation, savings, and setup."
head:
  - - meta
    - property: og:title
      content: "ContextIQ CLI reference"
---

# CLI reference

All commands are subcommands of `tokengraph` (or `python tokengraph_all.py`).
Add `--help` to any command for its full flags. Run against the current repo, or
point elsewhere with the global `--path PATH`.

## Retrieval

| Command | Purpose |
| --- | --- |
| `context "task" -b 6000` | Token-budgeted context pack for a task |
| `ask "task"` | Focused pack + intent / coverage / risk / cost metadata |
| `semantic "query" -n 10` | Find symbols by **meaning** (embeddings) |
| `federated "task" --root ../a --root ../b` | Retrieve across multiple repos, merged |
| `diff-context` | Budgeted pack for exactly what the git diff touches |

## Graph navigation

| Command | Purpose |
| --- | --- |
| `modules` | Token-count table of top-level dirs — **call first** |
| `arch` | Whole-repo overview: modules, hubs, cycles, languages, routes |
| `explain FILE` | Signatures + imports + external callers for a file |
| `skeleton FILE` | Every signature in a file, no bodies |
| `summary FILE` | Compact module summary |
| `callers` / `callees QNAME` | Trace the call graph (full symbol source: MCP `get_symbol`) |
| `impact QNAME` | Blast radius: callers / subclasses / tests |
| `method-impact QNAME` | Function-level: who breaks / deps / overrides / call sites |
| `test-map [TARGET]` | Map implementations ↔ tests (+ coverage %) |
| `lines FILE 40 80` | Surgical line range (secret-scanned, sandboxed) |
| `map imports\|hierarchy\|routes\|hubs` | Project graph views |

`map routes` covers Flask/FastAPI, Django URLconfs, Express, NestJS, Rails,
Spring, Go and Next.js-style file-based endpoints, including declarations split
across several lines. Each route carries the `handler` symbol it belongs to —
taken from the decorated function, the handler named in the call, or the
enclosing one — which is the link between an app's HTTP surface and its call
graph. Where no handler can be identified honestly (a Django route declared
inside `urlpatterns`), the field is simply absent rather than guessed.

## Trust gates

| Command | Purpose |
| --- | --- |
| `validate "task" --min-coverage 60` | Coverage gate (percent, default 60); exit 1 if insufficient |
| `judge --answer-file A --context-file C` | Score if an answer is grounded (0–100%) |
| `verify --answer-file A` | Flag fabricated files / symbols; exit 1 if any |
| `grounding` | Quantify the guard: fabrications caught vs real refs flagged |

## Grounded generation

| Command | Purpose |
| --- | --- |
| `conventions` | Detect naming / layout / test / export style + conformance |
| `scaffold NAME --kind module --apply` | Propose (or write) a convention-matched file |
| `verify-plan` | Check a plan's file/symbol refs + blast radius before acting |
| `verify-output` | Audit AI-generated code for fabricated files / symbols / imports |
| `review` | Audit the working/staged diff for scope drift, hub edits, missing tests |
| `create "task" --apply` | Orchestrate scaffold → plan → verify-output → review |

## Cost, savings & optimization

| Command | Purpose |
| --- | --- |
| `cost` | Estimate USD of an API call before sending it |
| `prompt-score` | Score a prompt's clarity / specificity / context / action |
| `summarize-chat` | Compress a long transcript into a token-cheap brief |
| `dedupe` | Remove near-duplicate context blocks |
| `squeeze` | Shrink a pasted stacktrace / CI log / JSON blob |
| `measure "task"` / `report` | Token savings for a task / aggregated |
| `gain` | Realized savings from the ledger (tokens + $), dashboards |
| `status` | One-line repo snapshot (branch / index / savings) |

## Setup & operations

| Command | Purpose |
| --- | --- |
| `index` | Build / rebuild the graph (MCP tool: `reindex`) |
| `serve` | Run the MCP server (stdio or HTTP) |
| `ide-setup` | Wire the MCP server + rules (default: Claude Code + VS Code/Copilot; `--all` for every editor) |
| `ide-plugin` | Scaffold installable VS Code / Neovim / JetBrains plugins |
| `langs` | List parseable languages by extraction tier |
| `evidence` | Deterministic, hash-grounded evidence pack (audit/CI) |
| `hallucination` | Multi-repo, reproducible grounding benchmark |
| `doctor` / `health` | Diagnose the install / index |

## See also

- [Quick start](./quick-start.md) for the recommended everyday flow.
- [When to use what](./when-to-use.md) to pick the right command for a task.
