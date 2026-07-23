# Working in this repo (Claude Code)

This project ships **ContextIQ**, a local code-graph MCP server (registered under
the id `tokengraph`), that gives you token-efficient context. **Prefer it over
reading whole files.**

## Use the graph first

Before opening files to understand the codebase:

1. **`find_relevant_context(task)`** — your default. Returns a token-budgeted pack
   of the most relevant symbols (full bodies for the core, signatures for
   neighbors, indexed chunks, module summaries) for a task. Read this instead of
   grepping and opening files.
2. **`search_semantic(query)`** — when you don't know the identifier. Finds
   symbols by meaning ("retry with backoff" → the reattempt helper).
3. **`get_symbol(qname)`** — full source of one symbol by qualified name.
4. **`get_callers(qname)` / `get_callees(qname)`** — trace the call graph.
5. **`get_method_impact(qname)`** — before editing a function: who breaks on a
   signature change (with call sites), its dependencies, and overrides.
6. **`get_test_map(target)`** — the tests for a file/symbol (naming + call graph);
   omit the target for the whole-repo impl↔test map and coverage %.
7. **`get_architecture_overview()`** — orient in one call: modules, hub files,
   import cycles, language mix, and route totals.
8. **`get_module_summary(file)`** — what a file is for, in a few tokens.
9. **`file_skeleton(file)`** — every signature in a file, no bodies.
10. **`estimate_savings(task)`** — proves how many tokens the pack saved.

Only fall back to opening a file whole when the pack explicitly drops something
you need (it lists dropped symbols by name) or you must edit it.

Symbols now carry their **doc comments** across every language (godoc / rustdoc /
Javadoc / JSDoc / TSDoc, not just Python docstrings), so `search_semantic` finds
code by what it does even when the identifier is opaque.

## Freshness

The graph **auto-refreshes on every tool call** — changed files are reparsed
before you get an answer, so it's never stale. A `PostToolUse` hook in
`.claude/settings.json` also pre-warms it after each edit. You normally don't
need to call `reindex` manually.

## Tips

- Spend a tight budget: `find_relevant_context(task, budget_tokens=4000)`.
- If you write a good mental summary of a module, persist it with
  `set_module_summary(file, summary)` so future packs reuse it.
- The CLI mirror is `python tokengraph_all.py context "task"` if you need it.
- One-command editor wiring: `tokengraph ide-setup` writes the MCP server **and**
  steering rules; `ide-setup --verify` proves each editor is wired.
- ContextIQ is **model-agnostic and offline** — it emits context packs, never
  calls an LLM, so it works with any model (cloud or local Ollama/llama.cpp).
