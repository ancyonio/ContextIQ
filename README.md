# ContextIQ

A local, single-file **AST code graph** that gives AI coding agents token-efficient context. Instead of letting the agent read whole files (and re-read them every session), ContextIQ indexes your repo into a local SQLite graph of symbols + call/inheritance edges, then serves a **token-budgeted "context pack"** of only the symbols relevant to a task.

> **Naming:** the product is **ContextIQ**. Its CLI is `tokengraph_all.py` and the MCP server registers under the id `tokengraph` — those identifiers are kept stable so existing wiring keeps working.

The graph **auto-refreshes on every query**, so it never goes stale — even right after an edit.

- **One file, zero required dependencies** to run the CLI ([tokengraph_all.py](tokengraph_all.py)).
- Works as an **MCP server** (50 tools) for MCP-capable clients, with one-command wiring for VS Code / Cursor / Windsurf / Zed / Claude Code and locally generated plugin projects for VS Code / Neovim / JetBrains. Marketplace publication remains a separate release step.
- **Deep parsing with a full call / import / inheritance graph for 25+ languages** — **Python** (stdlib `ast`) and, via tree-sitter, **Java / Go / TypeScript / JavaScript / C / C++ / C# / Rust / PHP / Ruby / Kotlin / Swift / Scala / Lua / Bash / Solidity / Perl / Erlang / Julia / R / Haskell / OCaml / Nim / PowerShell / Dart**. Plus lightweight regex indexing for **30+ more languages** — SQL / Elixir / Clojure / F# / Groovy / Zig / Crystal / Haxe / Objective-C / Visual Basic / Tcl / Pascal / GDScript and markup/config (Vue, Svelte, HTML, CSS, YAML, TOML, XML, INI, GraphQL, Terraform, Protobuf, **Markdown**, Dockerfile, …). Every deep-parsed language falls back to regex when its grammar isn't installed. Run `python tokengraph_all.py langs` to list them all.
- **Grounded creation & guardrails** — close the loop from *retrieval* to *safe code generation*: detect house **`conventions`** (and **`conventions --fix`** to auto-rename outliers to house style), **`scaffold`** convention-matched files, **`verify-plan`** / **`verify-output`** to flag fabricated files / symbols / local imports, **`review`** a diff for scope-drift, hub-edits, missing tests and breaking changes, and **`create`** to orchestrate them through a gated pipeline.
- **Realized-savings ledger** — every `context` / `ask` / `measure` / `generate` call (and the MCP context tool) appends its pack-vs-whole-file delta to a privacy-safe, count-only ledger. **`gain`** rolls it up into a token + **dollar** report with `--since` windows, per-op breakdown, daily/weekly/monthly trends, and a self-contained **`--html`** dashboard. **`status`** gives a one-line repo snapshot (branch / dirty / index freshness / notes / cumulative savings).
- **Monorepo-aware** — **`generate --monorepo`** discovers every nested package by manifest and writes per-package context; **`--each`** does the same per immediate sub-directory.
- **Auditable & measurable** — deterministic, hash-grounded **`evidence`** packs for CI, and a reproducible multi-repo **`hallucination`** reduction benchmark.
- **Cost & context optimization** — **model-aware token counting** and pre-flight **`cost`** estimation (input+output USD across **GPT / Claude / Gemini / Llama**, `--compare` to pick the cheapest sufficient model), **`prompt-score`** to catch under-specified prompts before they burn a round-trip, automatic **context deduplication** in every pack (plus **`dedupe`** for ad-hoc blocks), and **`summarize-chat`** to compress a long session into a token-cheap brief. All deterministic and local.

---

## Why it saves tokens

| Without ContextIQ | With ContextIQ |
|---|---|
| Agent opens whole files to find relevant code | Agent calls `find_relevant_context(task)` and gets only the relevant symbols |
| Re-reads the codebase every session | Graph persists in `.tokengraph/graph.db` across sessions |
| Risk of stale understanding after edits | Graph re-indexes incrementally before every query |
| Savings are invisible / unprovable | Every retrieval appends to a savings ledger; `gain` reports tokens + dollars saved over time |

A context pack contains: relevant symbols (full body when small, signature when large), their callers/callees/base classes as signatures, matching indexed source chunks, and a list of anything dropped for budget so the agent can request it by name. Near-duplicate pieces (e.g. an indexed chunk that merely re-shows a symbol body already in the pack) are removed automatically and reported under `deduped`.

---

## Install

The CLI needs **only Python 3.10+**. Pick a channel:

```bash
pipx install "contextiq[all]"      # isolated global install (the npm -g / Volta equivalent)
uvx contextiq context "the task"   # zero-install run (the npx equivalent)
pip install "contextiq[all]"       # into the current environment
docker run --rm -v "$PWD:/repo" -w /repo contextiq context "the task"
```

The `[all]` extra pulls `tree-sitter-language-pack` (deep parsing for all 25+ languages),
`fastmcp` (MCP `serve`), `tiktoken` (accurate token counts), and `sentence-transformers`.
Slimmer extras: `[mcp]`, `[tokens]`, `[langpack]`, `[treesitter]`, `[ann]`, and
`[dashboard]`. With zero extras the
single file still runs the full CLI (regex parsing + heuristic token counts).

Set `TOKENGRAPH_OFFLINE=1` to disable remote configuration loading and optional
neural-model loading. The deterministic hash embedding remains available. For large
indexes, install `[ann]` and set `TOKENGRAPH_VECTOR_BACKEND=hnsw`; exact cosine remains
the default and fallback.

To ship binaries / publish, run `tokengraph dist` (CI release workflow, Dockerfile,
Homebrew formula, `install.sh`) and `tokengraph freeze --build` for a local binary.

> **Recommended: use a virtualenv.** FastMCP pulls in `starlette`/`cryptography`, which can conflict with other globally-installed packages. Isolating ContextIQ avoids that:
> ```bash
> python -m venv .venv
> .venv/Scripts/python.exe -m pip install fastmcp tiktoken   # Windows
> # .venv/bin/python -m pip install fastmcp tiktoken          # macOS/Linux
> ```
> Then point the MCP configs' `"command"` at that venv's python (see below).

---

## CLI usage

Run from your repo root (the default `--path` is the current directory).

```bash
# Build / update the local graph
python tokengraph_all.py index

# Get a token-budgeted context pack for a task (prints markdown)
python tokengraph_all.py context "add retry logic to the http client" -b 6000

# Other lookups
python tokengraph_all.py semantic "retry with backoff"  # find symbols by meaning
python tokengraph_all.py skeleton path/to/file.py      # signatures only, no bodies
python tokengraph_all.py summary path/to/file.py       # compact module summary
python tokengraph_all.py callers my.module.func        # who calls it
python tokengraph_all.py callees my.module.func        # what it calls
python tokengraph_all.py measure "add retry logic"     # token savings vs whole files
python tokengraph_all.py report --tasks-file tasks.txt -o report.md --csv report.csv  # aggregate with/without report
python tokengraph_all.py report --tasks-file tasks.txt --csv runs.csv --append        # same, accumulated across runs
python tokengraph_all.py gain                           # realized savings from the ledger (tokens + $)
python tokengraph_all.py gain --since 30d --all         # window + daily/weekly/monthly trends
python tokengraph_all.py gain --report                  # per-workspace dashboard -> .tokengraph/token-usage.html
python tokengraph_all.py gain --serve                   # live dashboard on 127.0.0.1 (no Streamlit/deps)
python tokengraph_all.py status                         # one-line repo snapshot (branch/index/savings)
python tokengraph_all.py stats                          # graph counts
python tokengraph_all.py langs                          # parseable languages

# Orientation, impact, and surgical fetch
python tokengraph_all.py modules                        # token table of top-level dirs (call first)
python tokengraph_all.py explain path/to/file.py        # signatures + imports + external callers
python tokengraph_all.py impact my.module.func          # blast radius (callers/subclasses/tests)
python tokengraph_all.py lines path/to/file.py 40 80    # exact line range (secret-scanned, sandboxed)
python tokengraph_all.py map imports                     # import graph (or: hierarchy | routes | hubs)
python tokengraph_all.py map routes                      # HTTP endpoints (Flask/FastAPI/Express/Spring/Go)
python tokengraph_all.py map hubs                        # fan-in/out ranking + import cycles
python tokengraph_all.py import-scip index.scip.json     # add precise external REFERENCES edges

# Decision support, gating, and cost control
python tokengraph_all.py ask "the task" --json          # focused pack + intent/coverage/risk/cost
python tokengraph_all.py validate "the task"            # coverage gate (exit 1 if insufficient)
python tokengraph_all.py judge --answer "..." --context-file ctx.md  # groundedness (exit 1 if not)
python tokengraph_all.py verify --answer-file reply.md  # flag fabricated files/symbols (exit 1 if any)
cat build.log | python tokengraph_all.py squeeze        # shrink a pasted stacktrace/CI-log/JSON blob
python tokengraph_all.py suggest-tool "the task"        # model-tier hint (fast/balanced/powerful)
python tokengraph_all.py routing                         # per-file model-tier hints
python tokengraph_all.py learn path/to/file.py          # reinforce a file (--bad to penalise)

# Cost, prompt quality, dedup, and session compression
python tokengraph_all.py cost --text "the prompt" --model claude-sonnet --output-tokens 800  # $ before you send
python tokengraph_all.py cost --text-file prompt.txt --compare   # rank GPT/Claude/Gemini/Llama cheapest-first
python tokengraph_all.py prompt-score --text "fix the retry logic in count_tokens"  # rate a prompt 0–100 + tips
cat snippets.txt | python tokengraph_all.py dedupe --threshold 0.8  # drop near-duplicate context blocks
python tokengraph_all.py summarize-chat --text-file session.txt --max-tokens 400   # compress a chat transcript

# Grounded creation: retrieval -> safe code generation
python tokengraph_all.py conventions                     # detect file-naming / per-dir / test / export style + conformance
python tokengraph_all.py conventions --check             # list non-conforming files (rename suggestions), exit 1 if any — CI gate
python tokengraph_all.py conventions --fix --dry-run     # preview renames that would bring outliers to house style
python tokengraph_all.py conventions --fix               # apply them (uses `git mv` when possible; refuses to clobber)
python tokengraph_all.py scaffold "payment processor" --kind class            # propose a convention-matched file + skeleton
python tokengraph_all.py scaffold "payment processor" --kind class --apply    # actually create it (refuses on conflict)
python tokengraph_all.py verify-plan --plan-file plan.md # check a plan's refs + blast radius + fabricated symbols
python tokengraph_all.py verify-output --answer-file generated.py  # audit generated code for fabricated files / symbols / local imports
python tokengraph_all.py review --staged                 # audit the diff for scope-drift / hub-edits / missing-tests / breaking-changes
python tokengraph_all.py create "add retry to http client" --kind module             # dry-run: scaffold -> plan -> review
python tokengraph_all.py create "add retry to http client" --answer-file out.py --apply  # full gated pipeline, writes scaffold

# Auditable evidence + guard measurement
python tokengraph_all.py evidence "the task" -o evidence.json  # deterministic, hash-grounded pack (context_hash + anchor_coverage)
python tokengraph_all.py grounding                       # quantify the hallucination guard (fabrications caught vs real flagged)
python tokengraph_all.py hallucination -o HALLUCINATION.md  # multi-repo, reproducible hallucination-reduction benchmark + report

# Cross-session memory (local to the repo)
python tokengraph_all.py memory --add "decided to use FTS5"  # append a note
python tokengraph_all.py memory                          # read notes + checkpoints
python tokengraph_all.py checkpoint "phase-1" --note "indexing done"

# Multi-assistant export + distribution + editor integration
python tokengraph_all.py generate --adapter claude --adapter cursor  # write CLAUDE.md / .cursorrules / copilot / windsurf / AGENTS.md
python tokengraph_all.py generate --monorepo --adapter copilot       # per-package context for every nested manifest-detected package
python tokengraph_all.py generate --each --adapter claude            # per immediate sub-directory (workspace of repos)
python tokengraph_all.py ide-setup                       # one-command MCP wiring for VS Code / Cursor / Windsurf / Zed / Claude Code
python tokengraph_all.py ide-setup --workspace-root repo-a --workspace-root repo-b  # multi-root wiring
python tokengraph_all.py ide-plugin                      # scaffold installable plugins: VS Code .vsix / Neovim Lua / JetBrains
python tokengraph_all.py freeze --build                  # build a standalone binary now (PyInstaller)
python tokengraph_all.py dist                            # scaffold release CI + Dockerfile + Homebrew formula + install.sh

python tokengraph_all.py watch                          # keep the graph fresh on a poll loop
python tokengraph_all.py serve                          # run as an MCP server (stdio)
```

`--path` is a **global** flag and goes *before* the subcommand:

```bash
python tokengraph_all.py --path /some/repo index
```

### `context` options

| Flag | Default | Meaning |
|---|---|---|
| `-b, --budget` | 6000 | Max tokens in the pack |
| `-d, --depth` | 1 | How far to expand along call/inheritance edges |
| `--max-body` | 1600 | Largest symbol body shown in full before falling back to signature + chunks |
| `-o, --out` | — | Write the pack to a file instead of stdout |
| `--no-refresh` | off | Skip the freshen-on-query reindex and use the graph as-is |

---

## Semantic search (embeddings)

Seeding is **hybrid**: lexical (FTS5) results and **semantic** (embedding) results
are combined with reciprocal-rank fusion, so a task phrased differently than the
code still finds the right symbols. This is fully offline with **zero required
deps** — the default backend is a deterministic hashing embedding (token +
char-trigram). It captures lexical/structural overlap robustly but is not a true
neural model.

For real neural semantic search, opt into a local model:

```bash
pip install sentence-transformers
export TOKENGRAPH_EMBEDDINGS=st               # Windows: $env:TOKENGRAPH_EMBEDDINGS="st"
export TOKENGRAPH_EMBED_MODEL=all-MiniLM-L6-v2   # optional, this is the default
python tokengraph_all.py index                # rebuild vectors with the model
```

Vectors live in the `vectors` table (float32 blobs) and are compared by cosine in
Python — fine for repo-scale symbol counts.

---

## Staying fresh (no stale mapping)

Three layers, in order of how aggressively they keep the graph current:

1. **Freshen-on-query (always on, the correctness layer).** Every MCP tool and the CLI `context`/`skeleton`/`callers`/`callees` commands run an incremental reindex *before* answering. A query can never read stale line spans. A **mtime + size fast path** makes this just a `stat()` sweep when nothing changed, so it's cheap.
2. **Pre-warm hook (Claude Code).** [.claude/settings.json](.claude/settings.json) has a `PostToolUse` hook that re-indexes after every `Edit`/`Write`/`MultiEdit`/`NotebookEdit`, so the query-time refresh is usually a no-op. *(Open `/hooks` once or restart Claude Code so the hook registers.)*
3. **`watch` daemon (optional, any client).** `python tokengraph_all.py watch --interval 2` continuously re-indexes on a poll loop — the lower-latency option for large repos, and the pre-warm equivalent for Copilot (which has no hook mechanism).

---

## Proving the savings: the ledger, `gain`, and the dashboard

Optimizing tokens only matters if you can *prove* it. ContextIQ keeps a running, privacy-safe ledger of realized savings and turns it into a token + dollar report.

**How it accumulates.** Every retrieval — the CLI `context` / `ask` / `measure` / `generate` commands and the MCP `find_relevant_context` tool — appends one line to `.context/gain.ndjson` with its pack size, the whole-file baseline, the delta, and a timestamp. The token-reducing tools (`squeeze` / `dedupe` / `summarize-chat`) log their own before→after delta the same way. Only genuine savings are recorded (a run that doesn't reduce anything is skipped). The lines are **count-only**: never a path, a query, or any source. Opt a single run out with `--no-track`, or disable globally with `TOKENGRAPH_NO_TRACK=1` (`SIGMAP_NO_TRACK=1` also honored).

```bash
python tokengraph_all.py gain                       # totals: tokens saved, reduction %, $ projection, per-op table
python tokengraph_all.py gain --since 7d            # window the ledger (7d / 12h / 90m / an ISO date)
python tokengraph_all.py gain --all                 # add daily/weekly/monthly trend buckets + a sparkline
python tokengraph_all.py gain --model claude-opus   # price the saved tokens against a specific model
python tokengraph_all.py gain --top 5 --json        # machine-readable rollup for CI / spreadsheets
python tokengraph_all.py gain --report              # write .tokengraph/token-usage.html for this workspace
python tokengraph_all.py gain --serve               # live dashboard on 127.0.0.1 (no Streamlit, no deps)
python tokengraph_all.py gain --html gain.html      # same report, written to a path you choose
python tokengraph_all.py gain --reset               # clear the ledger
```

### The per-workspace dashboard: `.tokengraph/token-usage.html`

Because `.tokengraph/` is created in **every** workspace you use ContextIQ in, each one carries its own dashboard next to its own graph. It is **regenerated automatically on every logged op**, so it is always current — just open it.

- **Zero dependencies, fully self-contained.** All CSS/JS/charts are inlined; no CDN, no network, no server. Double-click the file and it works offline, in any client.
- **Six sections, all from the local ledger + graph** — *Overview* (hero cost avoided, a reduction gauge, and scorecards for saved / **sent (consumed)** / cost of tokens sent / reduction / runs / baseline / avg per run / leverage / files covered), *Savings* (baseline → avoided → sent waterfall, a stacked avoided-&-sent trend, and a 26-week activity heatmap), *Operations* (tokens avoided by op, share of runs, and a per-op table), *Cost by model* (the same token counts priced against every model, input **and** output list prices), *Workspace* (files / symbols / edges / chunks / summaries / graph size and a language breakdown), and the raw *Activity log*.
- **Enterprise UI, built in** — design tokens for color/type/space, light **and** dark themes (OS setting plus a toggle that overrides it), WCAG-minded markup (skip link, labelled controls, `aria-live` status, `scope`d table headers, visible focus rings, reduced-motion support), skeleton loading and explicit empty states, and a responsive layout down to phone width. No numbers are shown that the ledger did not record.
- **Interactive without a server** — the pricing-model and date-range selectors re-filter an inlined snapshot client-side, so switching to `claude-opus` or "last 7 days" works from `file://`.
- **Live two ways.** Served (`gain --serve`) the page polls `/data.json` and redraws in place. Opened as a file it reloads periodically to pick up the rewritten snapshot — the page detects which mode it is in and shows a `live` or `snapshot` badge.

```bash
python tokengraph_all.py gain --serve --port 8787   # http://127.0.0.1:8787 (loopback only)
```

> **`dashboard.py` (Streamlit) has been removed.** The report above replaces it and shows strictly more, with no `streamlit`/`plotly` install and no server process. The `dashboard` subcommand remains only as a signpost to `gain --report` / `gain --serve`.

Example:

```
savings (all time): 209,426 tokens saved across 2 run(s)  (96.1% reduction)
  projected cost saved: $0.6283 @ claude-sonnet ($3.0/1M tok)
  op               runs        saved  reduction
  context             1      104,983      97.2%
  measure             1      104,443      94.9%
```

> The dollar figure is an **indicative projection** — saved tokens × the model's list input price (`claude-opus`/`claude-sonnet`/`claude-haiku`/`gpt-4o`/`gpt-4o-mini`/`gpt-4.1`/`gemini-1.5-pro`/`gemini-1.5-flash`). It estimates avoided input cost, not a billing reconciliation.

`status` gives the at-a-glance version for any session:

```bash
python tokengraph_all.py status
# branch=main  dirty=3  indexed=15 files / 676 symbols  index=1m ago
# notes=2  savings=209,426 tok over 2 run(s)
# report=/path/to/repo/.tokengraph/token-usage.html
```

The same realized rollup is available to agents via the **`savings_ledger`** MCP tool, and `health` grades the project off the same `usage.ndjson` trend.

---

## Working across a monorepo

`generate` can fan out across a workspace instead of treating it as one repo:

```bash
python tokengraph_all.py generate --monorepo --adapter copilot --adapter claude
# discovers every nested package by manifest (package.json / pyproject.toml / go.mod /
# Cargo.toml / pom.xml / build.gradle / composer.json / Gemfile) and writes per-package context

python tokengraph_all.py generate --each --adapter claude
# treats every immediate sub-directory as its own project (a workspace of repos)
```

Each package gets its own context file(s) written in place, so per-package steering stays scoped and the agent never loads a sibling package it doesn't need. The single graph still answers cross-package queries when you want them.

---

## Grounded creation & guardrails

Retrieval keeps an agent oriented; these close the loop to **safe generation** — every step is deterministic and graph-backed (no LLM), and the pipeline writes nothing unless you pass `--apply`.

```bash
python tokengraph_all.py conventions          # learn house style (+ --check to gate non-conforming files in CI)
python tokengraph_all.py scaffold "rate limiter" --kind class --apply   # create a convention-matched file (refuses on conflict)
python tokengraph_all.py verify-plan --plan-file plan.md                 # do the plan's refs exist? blast radius? fabricated symbols?
python tokengraph_all.py verify-output --answer-file out.py              # did the generated code invent files / symbols / local imports?
python tokengraph_all.py review --staged                                 # scope-drift / hub-edits / missing-tests / breaking-changes
python tokengraph_all.py create "add retry to http client" --apply       # orchestrate all of the above through a gated pipeline
```

- **`conventions`** detects naming (global + per-directory), the test pattern, export style (public/private), and a **conformance score**; `--check` lists outliers with rename suggestions and exits non-zero (a CI gate); **`--fix`** applies those renames to bring outliers to house style (preferring `git mv`, refusing to clobber, with `--dry-run` to preview).
- **`scaffold`** proposes a path + skeleton that match house style and **refuses on conflict**; `--apply` writes it.
- **`verify-plan` / `verify-output`** flag fabricated files, symbols, and **repo-local imports** (third-party packages are never flagged). `verify-output` is the "did the model hallucinate?" gate to run on generated code.
- **`review`** audits a diff for scope-drift, edits to high-fan-in hub files, missing test changes, and **breaking changes** (a removed symbol that still has live callers).
- **`create`** is a gated state machine — each stage must pass before the next; dry-run by default.

### Auditable evidence & the hallucination benchmark

```bash
python tokengraph_all.py evidence "the task" -o evidence.json   # byte-stable, hash-grounded pack for CI
python tokengraph_all.py hallucination -o HALLUCINATION.md      # reproducible multi-repo reduction benchmark
```

`evidence` emits a deterministic JSON artifact (same index + task → identical `context_hash`) with per-file `reason` / `confidence` / `source_lines` / `related_tests` / `risk_label` and an `anchor_coverage` proving each cited symbol resolves to a real line span. `hallucination` partitions the repo and reports a modeled, reproducible codebase-fact hallucination-reduction figure with a per-repo spread.

---

## Editor integration & distribution

```bash
python tokengraph_all.py ide-setup     # one-command MCP wiring: VS Code / Cursor / Windsurf / Zed / Claude Code (+ Neovim & JetBrains snippets)
python tokengraph_all.py ide-plugin    # scaffold installable plugins: packageable VS Code .vsix, lazy.nvim Lua, Gradle JetBrains
python tokengraph_all.py freeze --build # build a standalone binary now (PyInstaller)
python tokengraph_all.py dist          # CI release workflow + Dockerfile + Homebrew tap + install.sh + PUBLISHING.md
```

`ide-setup` merges non-destructively into each editor's MCP config and can repeat
`--workspace-root` for multi-root workspaces. `ide-plugin` emits packageable plugin
projects; it does not imply marketplace publication. `dist` scaffolds the separate
publishing workflow.

---

## Use as an MCP server

Both clients launch `python tokengraph_all.py serve` over stdio and call the same tools. The config files are already in this repo:

### Claude Code — [.mcp.json](.mcp.json)
```json
{
  "mcpServers": {
    "tokengraph": {
      "type": "stdio",
      "command": "python",
      "args": ["d:/05. TECHNICAL/ContextIQ/tokengraph_all.py", "serve"],
      "env": { "TOKENGRAPH_ROOT": "." }
    }
  }
}
```
Claude Code auto-detects `.mcp.json` and prompts to trust the server.

### GitHub Copilot (agent mode) — [.vscode/mcp.json](.vscode/mcp.json)
```json
{
  "servers": {
    "tokengraph": {
      "type": "stdio",
      "command": "python",
      "args": ["d:/05. TECHNICAL/ContextIQ/tokengraph_all.py", "serve"],
      "env": { "TOKENGRAPH_ROOT": "${workspaceFolder}" }
    }
  }
}
```
In VS Code 1.102+, open the file and click **Start** on the server (or reload the window).

> Using a venv? Change `"command": "python"` to the venv interpreter, e.g.
> `"d:/05. TECHNICAL/ContextIQ/.venv/Scripts/python.exe"`.
> `TOKENGRAPH_ROOT` sets the repo root the server indexes (honored by `--path`).

### Tools the server exposes

| Tool | Purpose |
|---|---|
| `find_relevant_context(task, budget_tokens=6000, depth=1, max_body_tokens=1600)` | **Start here.** Token-budgeted pack (hybrid lexical+semantic seeding) of the most relevant symbols + chunks + summaries |
| `ask(task, budget_tokens=6000, depth=1)` | Like `find_relevant_context` but returns structured metadata: intent, coverage %, risk, cost/savings, top files + the pack |
| `list_modules()` | Token-count table of top-level directories — **call first** to scope retrieval to one module |
| `search_semantic(query, limit=12)` | Find symbols by meaning when you don't know the name |
| `get_symbol(qname)` | Full source of one symbol by qualified name (`module.Class.method`) |
| `get_lines(file, start, end)` | Surgical fetch of an exact line range — clamped, **secret-scanned**, sandboxed to the repo |
| `get_callers(qname)` / `get_callees(qname)` | Edges into / out of a symbol |
| `get_impact(qname)` | Blast radius: direct + transitive callers, subclasses, files & tests touched |
| `explain_file(file)` | Signatures + imports + external callers (who depends on a file) |
| `get_map(kind)` | Project graph: `imports`, `hierarchy` (class inheritance), `routes` (HTTP endpoints), or `hubs` (fan-in/out + import cycles) |
| `get_module_summary(file)` | Compact, few-token overview of a whole file |
| `set_module_summary(file, summary)` | Cache an agent-written summary (persists until the file changes) |
| `file_skeleton(file)` | Signatures for every definition in a file (no bodies) |
| `get_routing()` / `suggest_tier(task)` | Model-tier hints (fast/balanced/powerful) with cost + candidate models |
| `validate(task, min_coverage=60)` | Coverage gate — is the assembled context sufficient before answering? |
| `judge(answer, context)` | Score whether an answer is grounded in a context (hallucination guard) |
| `verify(answer)` | Flag fabricated file paths / code symbols in an answer, with "did you mean?" suggestions (deterministic, no LLM) |
| `squeeze(text, kind="auto")` | Shrink a pasted stacktrace / CI log / JSON blob before it costs tokens — drops vendor frames & build noise, keeps the diagnostics |
| `learn(file, good, weight=1.0)` | Reinforce / penalise a file's local ranking weight |
| `read_memory(limit=20)` / `write_memory(text, kind)` / `create_checkpoint(label, note)` | Cross-session decision log + git-snapshot checkpoints, stored **locally** |
| `estimate_savings(task)` | Tokens in the pack vs. reading the referenced files whole (single what-if) |
| `savings_report(tasks)` | Aggregate with/without savings across many tasks |
| `savings_ledger(since="", model="claude-sonnet", top=0)` | **Realized** savings from the persistent ledger: totals, reduction %, per-op breakdown, dollar projection, and daily/weekly/monthly trends |
| `estimate_call_cost(prompt, model="claude-sonnet", expected_output_tokens=500, compare=False)` | Price an API call **before** sending it — model-aware input+output USD; `compare=True` ranks GPT/Claude/Gemini/Llama cheapest-first |
| `count_tokens_model(text, model="gpt-4o")` | Model-aware token count across the GPT / Claude / Gemini / Llama tokenizer families |
| `score_prompt_quality(prompt)` | Score a **prompt** 0–100 on clarity / specificity / context / actionability, with concrete fix suggestions (distinct from `judge`, which scores an *answer*) |
| `dedupe_context(blocks, threshold=0.8)` | Remove near-duplicate context snippets (packs are already deduped automatically; this is for ad-hoc blocks) |
| `summarize_chat(transcript, max_tokens=400)` | Compress a chat transcript into a token-cheap brief — decisions, action items, open questions, code entities touched |
| `reindex()` | Force a full rescan (auto-refresh already runs per call) |

**Grounded creation & guardrails** (retrieval → safe code generation):

| Tool | Purpose |
|---|---|
| `conventions()` | Detect house style — naming (global + per-directory), layout, test pattern, export style, and a conformance score with rename suggestions |
| `scaffold(name, kind="module", apply=False)` | Propose (or `apply=True` create) a convention-matched file + skeleton; refuses on conflict, never overwrites |
| `verify_plan(plan)` | Check a plan's file/symbol refs: which exist, which are new, blast radius, and any fabricated symbols |
| `verify_output(answer)` | Audit AI-generated code for fabricated files, symbols **and local imports** (external packages aren't flagged) |
| `review_diff(staged=False)` | Audit the working/staged diff for scope-drift, hub-edits, missing tests, and breaking changes (removed symbols with live callers) |
| `create(task, kind="module", answer="", apply=False)` | Gated state machine: scaffold → verify-plan → verify-output → review; dry-run unless `apply=True` |
| `evidence(task, budget_tokens=6000)` | Deterministic, hash-grounded evidence pack for audit/CI — per-file `reason`/`confidence`/`source_lines`/`related_tests`/`risk_label` + `context_hash` + `anchor_coverage` |
| `hallucination_benchmark(sample_per_repo=40, baseline_per_100=99.8)` | Reproducible multi-repo codebase-fact hallucination-reduction benchmark (no LLM) |

> **Privacy:** all output is secret-scanned (AWS/GitHub/JWT/DB-URL/SSH/GCP/Stripe/Twilio/Slack + generic password patterns redacted to `[REDACTED]`). Memory, checkpoints, and learning weights never leave the repo.

### How the cost & context-optimization tools behave

All five are **deterministic and local** (no LLM, no network). Their heuristics are tuned deliberately — know the trade-offs:

- **`estimate_call_cost` / `count_tokens_model`** — token counts use tiktoken's `cl100k_base` corrected by a per-family ratio (GPT/Claude 1.00, Gemini 0.98, **Llama 1.10**), so a Llama call reports a few more tokens than the GPT/Claude base for the same text. Prices are **list-price constants** in `MODEL_PRICES_PER_1M` ([tokengraph_all.py](tokengraph_all.py)) — update them there when a provider changes pricing. Cost is usually dominated by `expected_output_tokens`, not the prompt.
- **`dedupe_context`** (and the automatic per-pack dedup) — uses **5-gram shingle containment**, precision-over-recall: it reliably collapses exact / near-exact repeats and keeps the *longest* variant as canonical, but it will **not** merge heavy paraphrases (that would need embeddings). Lower `threshold` (default `0.8`) for looser matching.
- **`summarize_chat`** — extractive and recall-favoring: it would rather over-capture a decision than miss one, so the "Decisions" bucket can include the odd question. It nets real savings only on **genuinely long** transcripts — on a short one the section scaffolding can cost more than it saves.
- **`score_prompt_quality`** — scores a *prompt* (not an answer) on clarity / specificity / context / actionability; a low score comes with concrete fix suggestions. Complements `judge`, which scores whether an *answer* is grounded.

---

## Using it across multiple projects

`tokengraph_all.py` is a single self-contained file that operates on **whatever
repo root you point it at** — you don't copy it into each project. The graph DB
(`.tokengraph/graph.db`) is created *inside each target project*, so projects
stay isolated. Keep the script in one stable location and reference its absolute
path everywhere.

### Option A — one-off CLI on any repo

`--path` is global and goes *before* the subcommand:

```bash
python "/path/to/tokengraph_all.py" --path "/path/to/other-project" index
python "/path/to/tokengraph_all.py" --path "/path/to/other-project" context "the task"
```

### Option B — register the MCP server once, globally (recommended)

Register at **user scope** so every project gets it without per-repo config. The
key: set the root to the working directory, because both clients launch the
server with the *current project* as its cwd — so one registration indexes
whatever project you're in, each into its own `.tokengraph/`.

**Claude Code** (run once, from anywhere):
```bash
claude mcp add --scope user --transport stdio tokengraph \
  -e TOKENGRAPH_ROOT=. \
  -- python "/path/to/tokengraph_all.py" serve
```

**GitHub Copilot** — put this in your **user** `mcp.json` (Command Palette →
"MCP: Open User Configuration"), not a per-workspace file:
```json
{
  "servers": {
    "tokengraph": {
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/tokengraph_all.py", "serve"],
      "env": { "TOKENGRAPH_ROOT": "${workspaceFolder}" }
    }
  }
}
```

### Per-project extras (optional)

The global server alone can query any repo. Copy these only into projects you
work in heavily:

- **Steering files** — [CLAUDE.md](CLAUDE.md) and
  [.github/copilot-instructions.md](.github/copilot-instructions.md), so the
  agent prefers the graph over whole-file reads.
- **Pre-warm hook** — [.claude/settings.json](.claude/settings.json); its
  `${CLAUDE_PROJECT_DIR:-.}` path is already portable, so it works as-is when
  copied.
- **Gitignore** — add `.tokengraph/` to each project's `.gitignore` (it's a
  rebuildable cache, not source).

### Two things to get right

1. **The `python` you launch must have `fastmcp`.** If you isolated it in a venv
   (recommended — see Install), point `command` at that venv's interpreter by
   **absolute path** so it works regardless of which project is open, e.g.
   `"/path/to/.venv/Scripts/python.exe"` (Windows) or `.venv/bin/python`.
2. **Keep the script at a stable path.** Move it somewhere durable (e.g.
   `~/tools/`) before registering; if you move it later, update the path in the
   global MCP registration.

---

## How it works

```
source files ──parse──► symbols + edges + embeddings + summaries ──resolve──► SQLite graph (.tokengraph/graph.db)
                                                                                   │
        task ──(lexical FTS5 + semantic vectors, RRF-fused)──► seeds ──expand edges──► budget assembly ──► context pack
```

- **Parse:** Python via stdlib `ast`; other languages via tree-sitter profiles or a conservative regex fallback. All paths emit the same `Symbol` / edge records, plus a per-symbol embedding and a per-file summary.
- **Store:** SQLite + FTS5 (with a `LIKE` fallback) + a `vectors` table + a `summaries` table, WAL mode for safe concurrent read/reindex. Incremental: unchanged files (by mtime+size, then hash) are skipped.
- **Edges:** `CALLS`, `INHERITS`, `IMPORTS`, and `DEFINES` (parent→child, which also answers "where is X defined"). Call/inherit resolution is best-effort and **scope-aware**: exact qname → **import-aware** (prefer the module a leaf was imported from) → **same enclosing scope** (a sibling in the same class/module — resolves intra-class `self.helper()` calls the old rule dropped) → same-file → unique global match, else dropped. The index report includes an `edge_resolution_pct` metric.
- **Precise references:** `import-scip` and the MCP `ingest_scip` tool consume JSON from `scip print --json`, map definition/reference occurrences to indexed symbols, and add `REFERENCES` edges used by context expansion and impact analysis. The built-in AST/tree-sitter resolver remains the zero-dependency fallback.
- **Retrieve:** hybrid seed search (lexical + semantic, reciprocal-rank fused) → BFS over `CALLS`/`INHERITS` edges → tiered budget fill (full bodies → signatures → **module summaries** → indexed chunks → dropped-by-name).

`.gitignore` is respected by default (a lightweight matcher; also skips `.git`, `node_modules`, `__pycache__`, `build`, `dist`, virtualenvs, etc.).

---

## Tests

```bash
python -m pytest -q                              # or: python -m unittest tests.test_contextiq_all -v
```

The suite lives in [tests/](tests/); `pyproject.toml` sets `pythonpath = ["."]` so the single-file `tokengraph_all` module stays importable from there.

**131 tests.** The core suite uses only the standard library; optional integration
tests exercise FastMCP when installed. Coverage includes incremental cross-file edge
retention, hard serialized token budgets, offline/privacy behavior, targeted indexing,
corpus benchmarking, multi-root editor wiring, real MCP client calls, dashboard ledger
concurrency, language extractors, and the grounded-creation pipeline.

---

## Benchmark & savings

Self-benchmark on **this repository** (5 files, 419 symbols, 881 edges):

```bash
python tokengraph_all.py benchmark         # corpus Recall@5, MRR, irrelevant-token ratio, latency
python tokengraph_all.py measure "task"    # token savings vs reading files whole (single task)
python tokengraph_all.py gain --all        # cumulative realized savings over time (tokens + $)
```

| Metric | Value | How |
|---|--:|---|
| Retrieval quality | Repository-dependent | `benchmark` uses `benchmarks/retrieval_tasks.json` human-authored tasks |
| Token savings (example task) | **95.9%** | pack ≈2,449 tok vs ≈59,678 tok reading the 3 referenced files whole |

> These are this repo's own numbers; hit@5 is high because the project is a single
> dense module. Run the same two commands inside your repo for representative figures —
> savings scale with codebase size (the larger the repo, the more you avoid re-reading).

For the hallucination-guard's effect, run `python tokengraph_all.py grounding` (fabrications caught vs. real refs flagged) or `python tokengraph_all.py hallucination -o HALLUCINATION.md` (a reproducible, multi-repo reduction report).

---

## Token-Efficient Development Playbook

Use these repo assets to keep GHCP and Claude Code focused and avoid large token consumption:

- [TokenEfficiency.md](TokenEfficiency.md) — full guide with context strategy, anti-patterns, checklists, and measurable savings workflows.
- [.prompts/bug-fix.md](.prompts/bug-fix.md) — minimal-context bug fixing template.
- [.prompts/code-review.md](.prompts/code-review.md) — risk-first review template.
- [.prompts/test-generation.md](.prompts/test-generation.md) — targeted test creation template.
- [.prompts/architecture-review.md](.prompts/architecture-review.md) — architecture review template.

Suggested daily flow:

1. Start with `find_relevant_context(task)` (or `python tokengraph_all.py context "task" -b 4000`).
2. Pull only missing symbols with `get_symbol` / `file_skeleton`.
3. Use one `.prompts/` template to keep requests structured and short.
4. Measure a single task with `python tokengraph_all.py measure "task"`.
5. Track the cumulative payoff with `python tokengraph_all.py gain --all` (or `gain --html gain.html` for a shareable dashboard) — every step above already feeds the ledger.

---

## Files

| Path | What |
|---|---|
| [tokengraph_all.py](tokengraph_all.py) | The entire tool (CLI + parsers + store + retriever + MCP server) |
| [tests/test_contextiq_all.py](tests/test_contextiq_all.py) | Test suite (stdlib `unittest` / `pytest`, zero deps) |
| [pyproject.toml](pyproject.toml) | Packaging metadata + `tokengraph`/`contextiq` entry points and optional extras |
| [.mcp.json](.mcp.json) | Claude Code MCP config |
| [.vscode/mcp.json](.vscode/mcp.json) | GitHub Copilot MCP config |
| [.claude/settings.json](.claude/settings.json) | Claude Code pre-warm hook |
| [CLAUDE.md](CLAUDE.md) | Steers Claude Code to prefer the graph over whole-file reads |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Same steering for GitHub Copilot |
| `.tokengraph/graph.db` | The generated local graph (safe to delete; rebuilt by `index`) |
| `.context/gain.ndjson` | Count-only realized-savings ledger read by `gain` (safe to delete; add to `.gitignore`) |
| `.context/usage.ndjson` | Per-run metric log (timestamp + reduction %) read by `health` / `status` |

Generated on demand (not committed): `ide-plugins/` (`ide-plugin`), distribution kit + `PUBLISHING.md` (`dist`), and per-assistant context files (`generate`).
=======
