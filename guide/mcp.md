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

This writes both the **MCP server config** and **steering rules**. By default it
targets **Claude Code + VS Code/Copilot** (the two editors nearly every repo
uses), plus any other editor it detects a footprint for (an existing `.cursor/`,
`.zed/`, `GEMINI.md`, etc.). Verify each wiring:

```bash
tokengraph ide-setup --verify        # exits 1 if a requested editor isn't wired
```

Widen the set, target a single editor, or preview first:

```bash
tokengraph ide-setup --all           # wire every supported editor (old default)
tokengraph ide-setup --editor cursor
tokengraph ide-setup --dry-run       # print the exact file list; write nothing
tokengraph ide-setup --global        # also Windsurf (~/.codeium) and Cline
tokengraph ide-setup --no-rules      # MCP server only, skip steering rules
tokengraph ide-setup --plugins       # also scaffold VS Code / Neovim / JetBrains plugins
```

Pin a team's editor set in `gen-context.config.json` so no flags are needed in
CI or hooks:

```json
{ "ide": { "editors": ["claude", "vscode"] } }
```

Supported editors: `claude`, `vscode`, `cursor`, `zed`, `continue`,
`jetbrains`, `nvim`, `gemini`, `roo`, `opencode`, `windsurf`, `cline`.
The default set is `claude` + `vscode` plus any editor detected in the repo;
`--all` selects every one above.

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

## Paths & the repo root

Two rules keep every file-taking tool working in any workspace:

**1. `file` arguments are repo-relative, forward-slash paths** — exactly as
indexed. `src/app/main.py` works; `D:\repo\src\app\main.py` and
`src\app\main.py` match nothing and return an empty result (not an error).
This applies to `get_module_summary`, `file_skeleton`, `explain_file`,
`get_lines`, `set_module_summary`, `get_test_map`, and friends.

**2. "Repo-relative" means relative to the server's root**, resolved once at
launch: `serve --path PATH` if given, else `$TOKENGRAPH_ROOT`, else the
directory the server was launched from. To get the right root in every
workspace:

- **Register per-workspace** (recommended): run `tokengraph ide-setup` inside
  each repo. The project-local config launches the server with that repo as
  its working directory, so the default root is always correct.
- **Using one global registration instead?** Make the root explicit per
  launch — e.g. `"env": { "TOKENGRAPH_ROOT": "${workspaceFolder}" }` in
  editors that expand variables (VS Code, Cursor). Never hardcode one repo's
  path in a global entry.
- **Don't set `TOKENGRAPH_ROOT` globally** in your shell profile — it silently
  overrides the working directory for every workspace.

## Freshness

The graph **auto-refreshes on every tool call** — changed files are reparsed
before you get an answer. A `PostToolUse` hook can also pre-warm it after each
edit. You normally don't need to call `reindex` manually.

## Next steps

- [Retrieval](./retrieval.md) — the tools your agent will lean on most.
- [Local LLMs](./local-llms.md) — pipe packs into Ollama / llama.cpp with no API key.
