---
title: Retrieval
description: "How ContextIQ builds a token-budgeted context pack — context, ask, semantic search, and surgical line fetches."
head:
  - - meta
    - property: og:title
      content: "ContextIQ retrieval — context, ask, semantic"
---

# Retrieval

Retrieval is the core of ContextIQ: turn a task into a **token-budgeted pack** of
only the symbols that matter.

## `context` — the workhorse

```bash
tokengraph context "add retry logic to the http client" -b 6000
```

`-b / --budget` caps the pack size in tokens. The pack includes:

- relevant symbols (full body when small, signature when large),
- their callers / callees / base classes as signatures,
- matching indexed source chunks,
- a **dropped list** — anything cut for budget, named so you can request it.

Near-duplicate pieces are removed automatically and reported under `deduped`.

## `ask` — retrieval with judgment

```bash
tokengraph ask "explain the auth flow" -b 6000
tokengraph ask "explain the auth flow" --json --validate
```

`ask` adds metadata on top of the pack: detected **intent**, **coverage**,
**risk**, and estimated **cost**. Use `--json` for machine output and
`--validate` to fail if the JSON drifts from its schema.

## `semantic` — find code by meaning

When you don't know the identifier:

```bash
tokengraph semantic "retry with backoff" -n 10
```

Finds the reattempt helper even if it's named something opaque. Powered by local
embeddings; warm the model once with `tokengraph embed-warm`.

## `lines` — surgical fetch

When you need an exact range and nothing else:

```bash
tokengraph lines path/to/file.py 40 80
```

Secret-scanned and sandboxed. Cheaper than opening the whole file.

## `federated` — across repos

```bash
tokengraph federated "find the auth middleware" --root ../svc-a --root ../svc-b
```

Merges a per-repo-sectioned pack from several roots — useful in a microservice
or polyrepo setup.

## Session reuse (MCP)

Over MCP, pass a stable `session` id to `find_relevant_context(task,
session="…")` so symbols already sent this session (and unchanged) are
referenced by name instead of resent — repeated retrievals in one conversation
cost far fewer tokens. (The CLI `context` command has no session flag.)

## Next steps

- Gate it: [Validate](./validate.md)
- Navigate deeper: `impact`, `method-impact`, `test-map` in the [CLI reference](./cli.md)
- Prove it: [Savings](./savings.md)
