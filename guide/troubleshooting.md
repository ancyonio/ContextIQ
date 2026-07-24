---
title: Troubleshooting
description: "Common ContextIQ issues and fixes — install, indexing, semantic search, MCP wiring, and freshness."
head:
  - - meta
    - property: og:title
      content: "ContextIQ troubleshooting"
---

# Troubleshooting

Start with the built-in diagnostics — they catch most problems:

```bash
tokengraph doctor      # diagnose the install / environment
tokengraph health      # index + graph health
tokengraph status      # branch / index freshness / savings snapshot
```

## Install

- **`tokengraph: command not found`** — install with `pipx install "contextiq[all]"`,
  or run from source with `python tokengraph_all.py …`.
- **Missing optional features** — install the extras: `pip install "contextiq[all]"`.

## Indexing

- **Stale or empty results** — the graph auto-refreshes, but you can force it:
  `tokengraph reindex`. Confirm counts with `tokengraph stats`.
- **A file isn't parsed deeply** — its tree-sitter grammar may be missing; it
  falls back to regex. Check `tokengraph langs` and `tokengraph diagnose-extractors`.

## Semantic search

- **`semantic` returns nothing / errors** — warm the embedding model once:
  `tokengraph embed-warm`, then retry. Check `embedding_status`.

## MCP / editor wiring

- **Editor doesn't see the tools** — run `tokengraph ide-setup` and confirm with
  `tokengraph ide-setup --verify` (exits 1 on any unwired editor).
- **Windsurf / Cline not wired** — they're per-user configs; add `--global`.
- **Server won't start** — run `tokengraph serve` directly to see the error.

## Coverage / trust gates

- **`validate` always fails** — lower `--min-coverage`, raise `--budget`, or make
  the task string more specific.
- **`verify` flags real symbols** — the graph may be behind; `reindex` and retry.

## Still stuck?

Open an issue with the output of `tokengraph doctor` and `tokengraph status`.

## Next steps

- Reference: [CLI](./cli.md)
- Setup: [MCP server](./mcp.md)
