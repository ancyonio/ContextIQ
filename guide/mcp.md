---
title: MCP server & editor wiring
description: "Run ContextIQ as an MCP server and wire it into Claude Code, Cursor, VS Code, Windsurf, and Zed with one command."
head:
  - - meta
    - property: og:title
      content: "ContextIQ MCP server & editor wiring"
---

# MCP server & editor wiring

ContextIQ ships an **MCP server** (50+ tools) so MCP-capable agents can call the
graph directly — no copy-paste. It registers under the id `tokengraph`.

## One-command setup

```bash
tokengraph ide-setup
```

This writes both the **MCP server config** and **steering rules** for every
project-local editor it detects. Verify each wiring:

```bash
tokengraph ide-setup --verify        # exits 1 if a requested editor isn't wired
```

Target a single editor, or include per-user editors:

```bash
tokengraph ide-setup --editor cursor
tokengraph ide-setup --global        # also Windsurf (~/.codeium) and Cline
tokengraph ide-setup --no-rules      # MCP server only, skip steering rules
tokengraph ide-setup --plugins       # also scaffold VS Code / Neovim / JetBrains plugins
```

Supported editors: `claude`, `vscode`, `cursor`, `zed`, `continue`,
`jetbrains`, `nvim`, `windsurf`, `cline`.

## Running the server manually

```bash
tokengraph serve                 # stdio (default for editors)
tokengraph serve --http          # HTTP transport
```

## Key MCP tools

Once wired, the agent should **prefer the graph over reading whole files**:

| Tool | Use it for |
| --- | --- |
| `find_relevant_context(task)` | The default — a budgeted pack for a task |
| `search_semantic(query)` | Find a symbol by meaning when you don't know its name |
| `get_symbol(qname)` | Full source of one symbol |
| `get_callers` / `get_callees(qname)` | Trace the call graph |
| `get_method_impact(qname)` | Who breaks on a signature change, before editing |
| `get_test_map(target)` | The tests for a file/symbol |
| `get_architecture_overview()` | Orient in one call |
| `validate` / `judge` / `verify` | The trust gates, callable inline |

## Freshness

The graph **auto-refreshes on every tool call** — changed files are reparsed
before you get an answer. A `PostToolUse` hook can also pre-warm it after each
edit. You normally don't need to call `reindex` manually.

## Next steps

- [Retrieval](./retrieval.md) — the tools your agent will lean on most.
- [Local LLMs](./local-llms.md) — pipe packs into Ollama / llama.cpp with no API key.
