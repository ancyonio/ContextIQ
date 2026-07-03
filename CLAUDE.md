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
5. **`get_module_summary(file)`** — what a file is for, in a few tokens.
6. **`file_skeleton(file)`** — every signature in a file, no bodies.
7. **`estimate_savings(task)`** — proves how many tokens the pack saved.

Only fall back to opening a file whole when the pack explicitly drops something
you need (it lists dropped symbols by name) or you must edit it.

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
