# Copilot instructions for this repo

This project provides **ContextIQ**, a local code-graph MCP server (registered
under the id `tokengraph`), for token-efficient context. In agent mode, **prefer
its tools over reading whole files**.

## Use the graph first

1. **`find_relevant_context(task)`** — default. A token-budgeted pack of the most
   relevant symbols (core bodies, neighbor signatures, indexed chunks, module
   summaries). Read this instead of opening many files.
2. **`search_semantic(query)`** — find symbols by meaning when the exact name is
   unknown.
3. **`get_symbol(qname)`** — full source of one symbol.
4. **`get_callers(qname)` / `get_callees(qname)`** — call-graph navigation.
5. **`get_module_summary(file)`** — a few-token overview of a file.
6. **`file_skeleton(file)`** — all signatures in a file, no bodies.
7. **`estimate_savings(task)`** — token savings vs. reading files whole.

Open a file in full only when the pack drops something you still need, or when
you must edit it.

## Freshness

The graph auto-refreshes on every tool call (changed files are reparsed before
answering), so results are never stale. Copilot has no edit hook, so if you want
continuous pre-warming run `python tokengraph_all.py watch` alongside your
session; otherwise the per-call refresh already keeps it correct.

## Setup

The MCP server is configured in `.vscode/mcp.json`. Start it from that file
(VS Code 1.102+, agent mode) or reload the window. It needs `pip install fastmcp`.
