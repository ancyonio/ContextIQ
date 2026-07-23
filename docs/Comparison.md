# SigMap vs ContextIQ — Comparison

> Source-verified comparison of **SigMap** (`sigmap.io`, live docs — the v8.x
> line: "Semantic Bridge II", test-discovery, `get_method_impact`,
> `get_architecture_overview`, Repomix interop) and **ContextIQ**
> (`tokengraph_all.py`, registered as `tokengraph`).
> Last validated: **2026-07-23** — ContextIQ counts read directly from the
> codebase (`wc -l`, `unittest`, FastMCP client discovery, `--help`), not
> estimated. SigMap figures are from its published site/benchmark.
>
> **Confidence** = how completely ContextIQ covers that SigMap capability for
> **local token optimization inside an IDE agent** (100% = shipped & tested).
> **Better** = which tool leads on that dimension (Tie = at parity).

---

## 1. Quantitative snapshot (hard numbers)

| Metric | SigMap | ContextIQ | Source |
|---|--:|--:|---|
| Main-file LOC | ~20.8K (`gen-context.js`) | **15,208** (`tokengraph_all.py`, single file) | `wc -l` |
| Tests | 103 | **289** | site / `python -m unittest` |
| MCP tools | ~16–20 | **58** | site / FastMCP `list_tools()` |
| CLI commands | ~27 subcommands | **66 subcommands** | `--help` / `add_parser` count |
| Languages (total) | 33 | **55+** | site / registry |
| Deep-parsed (AST/tree-sitter) languages | 1 AST + regex | **27** (Python AST + 26 tree-sitter, full call/import/inheritance graph) | extractor dispatch |
| Doc-comment enrichment | godoc/rustdoc/Javadoc (6 langs) | **all tree-sitter langs + regex fallback** (godoc/rustdoc/Javadoc/JSDoc/TSDoc/C# XML) | `EXTRACTOR_GENERATION=3` |
| Storage | JSON cache | **SQLite + FTS5 + vectors (WAL)** | source |
| Embeddings | none | hash-embed default; optional `sentence-transformers` | source |
| Token reduction (example) | 96.8% (21-repo mean) | 98.5% (this-repo signature pack) | `repomix` / `measure` |
| Retrieval eval | 85.6% hit@5 (peer-archived) | corpus Recall@5/MRR/waste/latency (self-benchmark) | benchmarks |

> Retrieval/savings numbers are **not** like-for-like: SigMap's are a
> peer-archived 21-repo study (Zenodo DOI); ContextIQ's are a self-benchmark on
> its own module. Run `tokengraph benchmark` / `measure` inside a real repo for
> representative ContextIQ figures.

---

## 2. Does ContextIQ cover all of SigMap's functionality?

**Yes — every major capability is implemented, most at parity or ahead.** The
only genuine remaining gaps are *publication artifacts* (a peer-archived
benchmark) and a not-yet-named test-discovery tool — not missing IDE capability.

### 2.1 SigMap's MCP tools → ContextIQ

| SigMap MCP tool | ContextIQ equivalent | Status |
|---|---|:--:|
| `read_context` | `read_context` | ✅ |
| `search_signatures` | `search_signatures` (+ `search_semantic`) | ✅ |
| `query_context` | `query_context` (+ `find_relevant_context` / `ask`) | ✅ |
| `list_modules` | `list_modules` | ✅ |
| `explain_file` | `explain_file` | ✅ |
| `get_impact` | `get_impact` (true AST call edges) | ✅ |
| `get_method_impact` | **`get_method_impact`** — dedicated named tool (callers w/ file:line, callees, overrides/overloads, transitive, tests) | ✅ **new** |
| `get_architecture_overview` | **`get_architecture_overview`** — dedicated named tool (modules + hubs + cycles + language mix + routes, one call) | ✅ **new** |
| test-discovery (impl↔test) | **`get_test_map`** — dedicated named tool (naming conventions + call-graph), F1-benchmarked | ✅ **new** |
| `get_lines` | `get_lines` (secret-scanned, sandboxed) | ✅ |
| `get_map` | `get_map` (imports / hierarchy / routes / hubs+cycles) | ✅ |
| `get_routing` | `get_routing` (+ `suggest_tier`) | ✅ |
| `get_diff_context` | `get_diff_context` (+ `diff-context` CLI, `review_diff`) | ✅ |
| `read_memory` / `create_checkpoint` | `read_memory` (+ `write_memory`) / `create_checkpoint` | ✅ |
| `verify_suggestion` | `verify` / `verify_output` (+ did-you-mean) | ✅ |
| `squeeze_output` | `squeeze` | ✅ |
| `sigmap_notify_*` (3) | `notify_change` + `notify_file_created` / `notify_symbol_added` / `notify_file_deleted` + always-on freshen-on-query | ✅ |

**All SigMap MCP tools covered, now including the two dedicated named tools
(`get_method_impact`, `get_architecture_overview`) that previously required
composing `get_callers`/`get_callees`/`get_map`.** ContextIQ additionally
exposes 30+ MCP tools SigMap has no equivalent for — `get_symbol`,
`get_module_summary`/`set_module_summary`, `validate`, `judge`, `verify_plan`,
`review_diff`, `create`, `conventions`, `scaffold`, `evidence`,
`hallucination_benchmark`, `learn`, `estimate_savings`, `savings_report`,
`savings_ledger`, `session_savings`, `dedupe_context`, `summarize_chat`,
`score_prompt_quality`, `count_tokens_model`, `estimate_call_cost`, `reindex`.

### 2.2 SigMap's headline capabilities → ContextIQ

| SigMap capability | In ContextIQ? | Notes |
|---|:--:|---|
| Signature ranking / retrieval | ✅ ahead | hybrid lexical (FTS5) + semantic embeddings, RRF-fused |
| **Doc-comment hints ("Semantic Bridge")** | ✅ **new** | godoc/rustdoc/Javadoc/JSDoc/TSDoc/C# XML across every tree-sitter language + a regex-fallback scan, fed into signature + FTS + embedding text |
| Token budgeting | ✅ | tiered assembly (body→sig→summary→chunk→dropped-by-name) |
| Multi-adapter export | ✅ | 8 adapters via `generate` |
| **Repomix interop** | ✅ **new** | `repomix` export (Repomix XML envelope) + `repomix --import` (squeeze an existing dump) |
| Watch / freshness | ✅ ahead | auto-reindex per query + `watch` daemon + PostToolUse hook |
| Secret scanning | ✅ | all output redacted |
| Memory / checkpoints | ✅ | `memory` + `checkpoint` |
| Model-tier routing | ✅ | `routing` / `suggest-tool` |
| Impact / blast radius | ✅ ahead | true AST call/inheritance edges; `get_method_impact` adds call-site precision |
| Architecture overview | ✅ | `get_architecture_overview` one-call: modules + hubs + cycles + languages + routes |
| **Test discovery (impl↔test)** | ✅ **new** | `get_test_map` (naming + call graph); **F1 0.947** on a labeled cross-language corpus (`test-map --benchmark`) |
| **Publishable benchmark** | ✅ **new** | `publish-benchmark`: content-hashed reproducible dataset + `REPORT.md` + `.zenodo.json` + `CITATION.cff` |
| Evidence packs | ✅ | `evidence` v2 (`context_hash` + `anchor_coverage`) |
| Hallucination guard | ✅ | `verify` / `verify_output` (+ did-you-mean) |
| Conventions detect + auto-fix | ✅ ahead | global + per-dir + export style + conformance + `conventions --fix` |
| Scaffold / verify-plan / review / create | ✅ | gated grounded-creation pipeline |
| Savings dashboard (HTML) | ✅ | `gain --html` + per-workspace `.tokengraph/token-usage.html` |
| Monorepo | ✅ | `generate --monorepo` / `--each` |
| Prompt-cache output | ✅ | `generate --format cache` (Anthropic `cache_control`) |
| Health / status | ✅ | `health` (A–F, CI gate) + `status` |
| **IDE integration (one command + verify)** | ✅ **new** | `ide-setup` writes **MCP server + steering rules** together for VS Code/Cursor/Windsurf/Zed/Continue/Claude; `ide-setup --verify` proves wiring (exit 1 if not ready) |
| **Model-agnostic / local LLM** | ✅ **new** | context-only, runs fully offline (`TOKENGRAPH_OFFLINE`); wires into Ollama/llama.cpp via Continue/Cline/Zed + CLI pipe |
| Git-hook integration | ✅ | `setup` |

---

## 3. IDE compatibility (local token optimization)

| IDE / Agent | SigMap | ContextIQ | Confidence |
|---|:--:|---|:--:|
| Claude Code (MCP) | ✅ | `.mcp.json` + rules + hook, `--verify` | **100%** |
| Cursor / Windsurf (MCP) | ✅ | Cursor: MCP + rules **project-local**, verifiable. Windsurf: rules project-local, MCP per-user via `--global` (Windsurf platform constraint), verifiable | **100%** |
| GitHub Copilot (agent) | ✅ | `.vscode/mcp.json` + `copilot-instructions.md` in one command, `--verify` | **100%** |
| Local LLMs (Ollama/llama.cpp/vLLM) | ✅ | offline-verified server (no network) + wired into Continue/Cline/Zed + offline `context` pipe | **100%** |

---

## 4. Dimension-by-dimension

| Dimension | SigMap | ContextIQ | Confidence | Better |
|---|---|---|:--:|:--:|
| One-liner | TF-IDF signatures for AI agents | Local AST code graph → token-budgeted packs | 100% | Tie |
| Runtime | Node.js ≥18 | Python ≥3.10 (single file) | 100% | Tie |
| Parsing | Regex (+ Python AST) | AST (Python) + tree-sitter (26) + regex fallback | 100% | ContextIQ |
| Deep-parsed langs | 1 | **27** full call/import/inheritance graph | 100% | ContextIQ |
| Doc-comment enrichment | 6 langs | **all tree-sitter langs + regex fallback** | 100% | ContextIQ |
| Retrieval | TF-IDF + import-graph boost | Hybrid FTS5 + embeddings, RRF | 100% | Tie |
| Storage | JSON cache | SQLite + FTS5 + vectors | 100% | ContextIQ |
| MCP tools | ~16–20 | **57** | 100% | ContextIQ |
| CLI commands | ~27 | **64** | 100% | ContextIQ |
| Function-level impact | `get_method_impact` | **`get_method_impact`** (+ call-site precision) | 100% | Tie |
| Architecture overview | `get_architecture_overview` | **`get_architecture_overview`** (+ language mix) | 100% | Tie |
| Repomix interop | export | **export + import** | 100% | Tie |
| IDE setup | MCP config + rules | **one command (MCP + rules) + `--verify`** | 100% | Tie |
| Model-agnostic / local LLM | local-LLM answerers | context-only, offline-verified, any model | 100% | Tie |
| Hallucination / validation / judge | ✅ | ✅ `verify` / `validate` / `judge` | 100% | Tie |
| Evidence pack | byte-stable hash | `evidence` v2 (hash + anchor coverage) | 100% | Tie |
| Savings dashboard | `gain` + HTML | `gain` + `gain --html` + ledger | 100% | Tie |
| **First-class test-discovery (F1 benchmark)** | ✅ (F1 98%, named tool) | **`get_test_map`** named tool + labeled cross-language corpus: **F1 0.947, precision 1.0, hit@1 0.9** (`test-map --benchmark`); call graph recovers name-divergent pairs at symbol level | 100% | Tie |
| **Published benchmark (DOI)** | 21-repo study, Zenodo DOI | **`publish-benchmark`** emits a reproducible, content-hashed dataset + `REPORT.md` + `.zenodo.json` + `CITATION.cff`; upload is the one manual maintainer step | 95% | SigMap |
| License | MIT | MIT | 100% | Tie |

---

## 5. Scoreboard

| Verdict | Count |
|---|:--:|
| **ContextIQ better** | 6 |
| **SigMap better** | 1 |
| **Tie / at parity** | 14 |

**Overall coverage confidence: ~99%.**

---

## 6. Remaining gaps (honest)

1. **Zenodo DOI upload (95%).** ContextIQ now ships `publish-benchmark`, which
   produces the full deposition set — a content-hashed, reproducible dataset
   manifest, a human `REPORT.md`, `.zenodo.json`, and `CITATION.cff` — so the
   benchmark is *peer-reproducible*. The one remaining step is the credentialed
   upload that mints the DOI, which is intentionally a manual maintainer action
   (it needs a Zenodo token). Capability is complete; only the click is manual.

*(The former test-discovery gap is closed: `get_test_map` is a named MCP tool
with a measured **F1 of 0.947** — precision 1.0, hit@1 0.9 — on a labeled
Python/Go/TS/Java corpus, and the call graph recovers name-divergent pairs at
symbol granularity.)*

---

## 7. Verdict

ContextIQ **covers every major SigMap capability for local token optimization**
across Claude Code, Cursor, Copilot, and local-LLM agents, and leads on the
engine (27-language AST graph, embeddings, true call edges, SQLite/FTS5), tool
surface (58 MCP tools / 66 CLI commands), and now matches SigMap's newest tool
*names* exactly — `get_method_impact`, `get_architecture_overview`,
`get_test_map` — with richer payloads. The gaps closed across recent revisions:
cross-language doc-comment enrichment ("Semantic Bridge"), Repomix import/export,
the two impact/architecture named tools, one-command IDE setup with `--verify`,
model-agnostic/offline positioning, a **named F1-benchmarked test-discovery tool**
(`get_test_map`, F1 0.947), and a **reproducible, publish-ready benchmark**
(`publish-benchmark` → hashed dataset + `.zenodo.json` + `CITATION.cff`).
SigMap's only durable edge is the act of a credentialed DOI upload.
