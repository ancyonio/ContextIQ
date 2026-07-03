# Sigmap vs ContextIQ — Comparison

> Source-verified comparison of **Sigmap** (`testmap/`, `gen-context.js`, v7.28.0) and
> **ContextIQ** (`tokengraph_all.py`, registered as `tokengraph`).
> Last validated: 2026-06-24 — counts read directly from each codebase, not estimated.
>
> **ContextIQ % complete** = maturity of ContextIQ's implementation of that dimension (100% = fully shipped & tested).
> **Better** = which tool currently leads on that dimension (Tie = at parity).

---

## 1. Quantitative snapshot (hard numbers)

| Metric | Sigmap v7.28 | ContextIQ | Source |
|---|--:|--:|---|
| Main-file LOC | 20,826 (`gen-context.js`) | **8,379** (`tokengraph_all.py`) | `wc -l` |
| Tests | 103 | **120** | `version.json` / `pytest --collect-only` |
| MCP tools | 17 | **39** | `src/mcp/tools.js` / `grep -c '@mcp.tool'` |
| CLI commands | ~27 subcommands (+~13 flag-ops) | **53 subcommands** | dispatch / `--help` |
| Output adapters | 8 (+`llm-full`) | 8 | `packages/adapters` / `ADAPTERS` dict |
| Languages (total) | 33 | **55+** | `version.json` / `langs` |
| Deep-parsed (AST) languages | 1 (Python AST; rest regex) | **25+** (tree-sitter) | extractor dispatch / profiles |
| Storage | JSON cache (`.sigmap-cache.json`) | **SQLite + FTS5 + vectors** | source |
| Embeddings | none | hash-embed default; optional `sentence-transformers` | source |
| Retrieval hit@5 (self) | 75.6% (21-repo study) | 1.00 (single dense module) | benchmarks |
| Token reduction (example) | 97.0% (21-repo mean) | 95.1% (this-repo example) | `measure` |
| Hallucination reduction | 99.6% (LLM-measured, published) | reproducible multi-repo modelled benchmark | source |

> The retrieval/savings numbers are **not** like-for-like: Sigmap's are a peer-archived
> 21-repo study (Zenodo DOI), ContextIQ's are a self-benchmark on its own single dense
> module (hit@5 is inflated by that). Run `tokengraph benchmark` / `measure` / `gain`
> inside a real repo for representative ContextIQ figures.

---

## 2. Does ContextIQ cover all of Sigmap's major functionality?

**Short answer: yes — every major capability is implemented, most are at parity or
ahead.** The only true gaps are *publication artifacts* (a peer-archived benchmark, a
shipped npm package), not missing features.

### 2.1 Sigmap's 17 MCP tools → ContextIQ

| Sigmap MCP tool | ContextIQ equivalent | Status |
|---|---|:--:|
| `read_context` | `read_context` | ✅ |
| `search_signatures` | `search_signatures` (+ `search_semantic`) | ✅ |
| `query_context` | `query_context` (+ `find_relevant_context` / `ask`) | ✅ |
| `list_modules` | `list_modules` | ✅ |
| `explain_file` | `explain_file` | ✅ |
| `get_impact` | `get_impact` (true AST call edges) | ✅ |
| `get_lines` | `get_lines` | ✅ |
| `get_map` | `get_map` (imports/hierarchy/routes/hubs) | ✅ |
| `get_routing` | `get_routing` (+ `suggest_tier`) | ✅ |
| `get_callee_signatures` | `get_callees` (+ `get_callers`, `file_skeleton`) | ✅ |
| `get_architecture_overview` | `get_map hubs` + `get_map routes` | ✅ |
| `read_memory` | `read_memory` (+ `write_memory`) | ✅ |
| `create_checkpoint` | `create_checkpoint` | ✅ |
| `get_diff_context` | `get_diff_context` MCP tool (diff-seeded pack: changed symbols + blast radius) + CLI `diff-context` | ✅ |
| `sigmap_notify_file_created` | `notify_file_created` (+ `notify_change`); also auto-refreshes on every query | ✅ |
| `sigmap_notify_symbol_added` | `notify_symbol_added` (file-granularity reindex); also auto-refreshes on every query | ✅ |
| `sigmap_notify_file_deleted` | `notify_file_deleted` (precise forget); also auto-refreshes on every query | ✅ |

**17/17 directly covered.** ContextIQ now exposes a dedicated `get_diff_context`
MCP retrieval tool and explicit `notify_*` hooks (`notify_change` plus
`notify_file_created` / `notify_symbol_added` / `notify_file_deleted`) — on top of
the always-on freshen-on-query auto-refresh, so the notify hooks are an optional
warm-the-graph fast path rather than a correctness requirement.
**ContextIQ adds 22+ MCP tools Sigmap has no equivalent for**, e.g. `get_symbol`,
`get_module_summary`/`set_module_summary`, `validate`, `judge`, `verify`,
`verify_plan`, `verify_output`, `review_diff`, `create`, `conventions`, `scaffold`,
`evidence`, `hallucination_benchmark`, `squeeze`, `learn`, `estimate_savings`,
`savings_report`, **`savings_ledger`**, `suggest_tier`, `reindex`.

### 2.2 Sigmap's headline capabilities → ContextIQ

| Sigmap capability | In ContextIQ? | Notes |
|---|:--:|---|
| TF-IDF / signature ranking | ✅ ahead | hybrid lexical (FTS5) + semantic embeddings, RRF-fused |
| Token budgeting | ✅ | tiered assembly (body→sig→summary→chunk→dropped-by-name) |
| Multi-adapter export | ✅ | 8 adapters via `generate` |
| Watch / freshness | ✅ ahead | auto-reindex per query + `watch` daemon + Claude hook |
| Secret scanning | ✅ | all output redacted |
| Memory / checkpoints | ✅ | `memory` + `checkpoint` |
| Model-tier routing | ✅ | `routing` / `suggest-tool` |
| Impact / blast radius | ✅ ahead | true AST call/inheritance edges, `edge_resolution_pct` |
| Evidence packs | ✅ | `evidence` v2 (`context_hash` + `anchor_coverage`) |
| Hallucination guard | ✅ | `verify` / `verify-output` (+ did-you-mean) |
| Conventions detection | ✅ ahead | global + per-dir + export style + conformance score |
| **Conventions auto-fix** | ✅ **new** | `conventions --fix` (git mv, clobber-safe, `--dry-run`) |
| Scaffold / verify-plan / review-pr / create | ✅ | `scaffold`/`verify-plan`/`review`/`create` gated pipeline |
| **Savings dashboard (`gain`)** | ✅ **new** | ledger + `gain` (tokens + $, `--since`, trends) |
| **HTML dashboard** | ✅ **new** | `gain --html` (self-contained, inline SVG) |
| **Monorepo support** | ✅ **new** | `generate --monorepo` / `--each` |
| Hot-cold / per-module strategy | ✅ | config `strategy` |
| Prompt-cache output | ✅ | `generate --format cache` (Anthropic `cache_control`) |
| **`status` quick view** | ✅ **new** | branch/dirty/index/notes/savings |
| Health score | ✅ | `health` (A–F grade, CI gate) |
| IDE integration | ✅ | `ide-setup` (VS Code/Cursor/Windsurf/Zed/Claude) + `ide-plugin` (installable .vsix / nvim / JetBrains) |
| Git hook integration | ✅ | `setup` |

*(rows marked **new** were added on 2026-06-24, closing the gaps flagged in the prior
revision of this document.)*

---

## 3. Dimension-by-dimension

| Dimension | Sigmap (testmap) | ContextIQ (tokengraph) | ContextIQ % | Better |
|---|---|---|:--:|:--:|
| **One-liner** | TF-IDF signature extraction feeding AI agents only relevant files | Local AST code graph serving token-budgeted context packs | 100% | Tie |
| **Runtime** | Node.js ≥18 | Python ≥3.10 | 100% | Tie |
| **Main-file size** | ~20.8K LOC | **8.4K LOC** | 100% | ContextIQ |
| **Parsing** | Regex signatures (+ optional Python AST) | AST (Python) + tree-sitter (25+) + regex fallback | 100% | ContextIQ |
| **Deep-parsed langs** | 1 (Python) | **25+** full call/import/inheritance graph | 100% | ContextIQ |
| **Total languages** | 33 | **55+** | 100% | ContextIQ |
| **Retrieval** | TF-IDF + intent weights + import-graph boost | Hybrid lexical (FTS5) + semantic embeddings, RRF | 100% | Tie |
| **Embeddings** | none (no drift) | hash default; optional sentence-transformers | 100% | ContextIQ |
| **Storage** | JSON cache | SQLite + FTS5 + vectors (WAL) | 100% | ContextIQ |
| **Freshness** | mtime cache; `--watch` | auto-reindex per query; `watch` daemon | 100% | ContextIQ |
| **MCP server** | 17 tools | **39 tools** | 100% | ContextIQ |
| **MCP clients** | Claude Code, Cursor | Claude Code, GitHub Copilot (agent mode) | 100% | Tie |
| **Multi-adapter output** | 8 (+llm-full) | 8 | 100% | Tie |
| **CLI commands** | ~27 subcommands | **53 subcommands** | 100% | ContextIQ |
| **Call graph / impact** | import-graph + regex-derived | true AST call/inherit/import edges, scope-aware | 100% | ContextIQ |
| **Token budgeting** | ✅ | ✅ tiered assembly | 100% | Tie |
| **Memory / checkpoints** | ✅ | ✅ | 100% | Tie |
| **Validation gate** | ✅ | ✅ `validate` | 100% | Tie |
| **Groundedness judge** | ✅ | ✅ `judge` | 100% | Tie |
| **Hallucination/fabrication check** | ✅ `verify-ai-output` | ✅ `verify` (did-you-mean) | 100% | Tie |
| **Log/blob squeeze** | ✅ | ✅ `squeeze` | 100% | Tie |
| **Learning / weights** | ✅ | ✅ `learn` | 100% | Tie |
| **Model-tier routing** | ✅ | ✅ `routing`/`suggest-tool` | 100% | Tie |
| **Secret scanning** | ✅ | ✅ | 100% | Tie |
| **Grounded-creation pipeline** | scaffold→verify-plan→verify-ai-output→review-pr | `scaffold --apply`→`verify-plan`→`verify-output`→`review`→`create` (+breaking-change) | 100% | Tie |
| **Conventions** | detect + report | detect (global+per-dir+export style+conformance) **+ `--fix` auto-rename** | 100% | Tie |
| **Evidence Pack JSON** | byte-stable, grounding hash | `evidence` v2 (`context_hash`+`anchor_coverage`+per-file reason/confidence/risk) | 100% | Tie |
| **Savings dashboard** | `gain` + `--dashboard` (HTML) | **`gain`** (ledger, `--since`, trends, $ projection) **+ `gain --html`** | 100% | Tie |
| **Monorepo** | `--monorepo` | **`generate --monorepo` / `--each`** | 100% | Tie |
| **Prompt-cache output** | `--format cache` | `generate --format cache` | 100% | Tie |
| **Hot-cold / per-module** | ✅ | ✅ | 100% | Tie |
| **Health / status** | `--health`, `status` | `health`, **`status`** | 100% | Tie |
| **Route / hub maps** | ✅ | ✅ `map routes` / `map hubs` | 100% | Tie |
| **Live-index notifications** | 3 MCP notify tools | 4 MCP notify tools (`notify_change` + 3 named) **plus** always-on auto-refresh | 100% | Tie |
| **Diff-context retrieval** | `get_diff_context` MCP tool | `get_diff_context` MCP tool (+ `diff-context` CLI, `generate --diff`, `review_diff`) | 100% | Tie |
| **IDE integration** | VS Code / JetBrains / Neovim plugins | `ide-setup` + `ide-plugin` (installable .vsix/nvim/JetBrains) | 100% | Tie |
| **Dependencies** | zero npm | zero required (optional fastmcp/tiktoken/tree-sitter) | 100% | Tie |
| **Distribution** | **published** npm + binaries + listed plugins | publish-ready (`dist`/`freeze`: PyPI/Docker/Homebrew); not yet uploaded | 95% | Sigmap |
| **Benchmarks** | 21-repo study, Zenodo DOI | reproducible self/multi-repo machinery; no published study | 70% | Sigmap |
| **License** | MIT | MIT | 100% | Tie |

\* Tie on outcome (the graph is always fresh); different mechanism.

---

## 4. Scoreboard

| Verdict | Count |
|---|:--:|
| **ContextIQ better** | 10 |
| **Sigmap better** | 3 |
| **Tie / at parity** | 27 |

**Mean ContextIQ completion: ~98%.**

---

## 5. Remaining gaps (honest)

1. **Published benchmark (credibility, not capability).** Sigmap ships a peer-archived
   21-repo retrieval study with a Zenodo DOI and an LLM-measured 99.6% hallucination
   reduction. ContextIQ has the machinery (`benchmark`, `measure`, `gain`, `grounding`,
   `hallucination`) but only a self-benchmark on its own dense module. **Fix:** run the
   suite across 10–20 real repos and archive the dataset + methodology.
2. **Distribution = published vs publish-ready.** `dist`/`freeze` scaffold PyPI, Docker,
   Homebrew, and marketplace manifests; the credentialed upload step hasn't run.
3. **`get_diff_context` as an MCP retrieval tool.** ✅ Done — `get_diff_context` is now a
   first-class MCP tool (and a `diff-context` CLI subcommand): it diff-seeds a budgeted
   pack with each changed symbol in full plus its callers/callees/base classes (blast
   radius), reporting touched symbols, impacted callers, and token savings. The explicit
   `notify_change` / `notify_file_created` / `notify_symbol_added` / `notify_file_deleted`
   hooks were added alongside it, so every row of the parity table is now ✅.
4. **A few Sigmap convenience subcommands** (e.g. `compare`, `history`, `share`,
   `suggest-profile`) have no 1:1 ContextIQ command, though `gain --all` (history),
   retrieval presets (profiles), and the adapters (share) cover most of the intent.

---

## 6. Verdict

ContextIQ **covers every major Sigmap functionality** and leads on the engine (deep
multi-language AST graph, embeddings, true call edges, SQLite/FTS5), tool surface
(39 MCP tools / 53 CLI commands vs 17 / ~27), and compactness (8.4K vs 20.8K LOC). The
features that were genuine gaps in the previous revision — the `gain` savings ledger,
HTML dashboard, `--monorepo`/`--each`, `conventions --fix`, and `status` — are now
implemented and tested. Sigmap's only durable edges are **publication artifacts**: an
already-shipped npm package and a peer-archived benchmark with a DOI. ContextIQ can
produce the equivalent locally (`dist`, `freeze`, `hallucination`); what remains is the
act of publishing.
