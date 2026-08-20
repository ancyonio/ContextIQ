---
title: Languages
description: "Deep call/import/inheritance graphs for 25+ languages via tree-sitter and Python's ast — including SQL schemas and Vue/Svelte components — plus regex indexing for 30+ more."
head:
  - - meta
    - property: og:title
      content: "ContextIQ language support"
---

# Languages

ContextIQ parses code in **tiers**. Run `tokengraph langs` to list every language
and its tier for your install.

## Deep parse (full graph) — 25+ languages

A full call / import / inheritance graph, with doc comments carried across:

**Python** (via the stdlib `ast`) and, via **tree-sitter**:
Java · Go · TypeScript · JavaScript · C · C++ · C# · Rust · PHP · Ruby ·
Kotlin · Swift · Scala · Lua · Bash · Solidity · Perl · Erlang · Julia · R ·
Haskell · OCaml · Nim · PowerShell · Dart · SQL.

Every deep-parsed language **falls back to regex** when its tree-sitter grammar
isn't installed.

### SQL is parsed as schema, not as text

`.sql` / `.ddl` files produce a real graph rather than a list of table names:

| Extracted | Becomes |
| --- | --- |
| `CREATE TABLE` / `CREATE VIEW` / `CREATE INDEX` / `CREATE FUNCTION` | symbols |
| every `column_definition` | a symbol scoped to its table (`schema.orders.user_id`) |
| `REFERENCES`, inline or as a named `CONSTRAINT` | a `REFERENCES` edge between tables |
| the tables a view or SQL function selects from | `REFERENCES` edges (lineage) |
| `ALTER TABLE … REFERENCES` in a migration | `REFERENCES` edges from that file to both tables |
| the comment above a statement | the symbol's doc comment, so `semantic` finds it |

So `impact schema.users` answers "what breaks if this table changes" — the
tables whose foreign keys point at it, the views that read it, and the
migrations that touch it — the same way it does for a function.

### Vue and Svelte components

`.vue` / `.svelte` are markup on the outside and TS/JS on the inside, so the
`<script>` block is handed to the TypeScript grammar and everything else is
blanked out — blanked rather than stripped, so every symbol keeps the line
number it has in the real file. An SFC frontend therefore gets the same call
graph as any other TypeScript, including calls that cross into your API client,
at the cost of no extra grammar.

## Regex index — 30+ more languages

Lightweight symbol indexing for:
Elixir · Clojure · F# · Groovy · Zig · Crystal · Haxe · Objective-C ·
Visual Basic · Tcl · Pascal · GDScript, and markup/config — HTML,
CSS, YAML, TOML, XML, INI, GraphQL, Terraform, Protobuf, **Markdown**,
Dockerfile, and more.

## When a grammar is missing

Deep-parse languages fall back to regex when their tree-sitter grammar isn't
installed, and that fallback is silent in the worst way: definitions still
appear, so the index looks healthy, but **no call edges are produced** — an
empty `get_callers` then means "not extractable here", not "nothing calls
this". `index` and `doctor` now say so directly, naming the languages and the
file counts, and `langs --repo` shows the full split. The fix is one install:

```bash
pip install 'contextiq[langpack]'
```

## Coverage is measured against code

`langs --repo` reports call-graph coverage as a share of **code** files.
Markdown and YAML can never carry a call graph, so counting them would measure
the repo's docs-to-code ratio rather than the extractor — a repo shipping 52
docs and 55 modules read as 49.5% covered when every one of its code files had
a full graph. Both counts are shown, so nothing is hidden by the narrowing.

## Exact references: SCIP

The tree-sitter tier resolves cross-file calls **by name**, which is
best-effort by construction: a same-named method in another file can absorb an
edge that belonged elsewhere. Where that precision matters — a large polyglot
repo, a refactor whose blast radius has to be right — run your language's SCIP
indexer and feed the result in:

```bash
scip-typescript index && tokengraph import-scip index.scip.json   # or scip-java / scip-go
```

That adds compiler-exact `REFERENCES` edges, which context expansion and
`impact` use directly. The built-in resolver stays as the zero-dependency
fallback, so this is an upgrade, never a prerequisite. `doctor` points it out
when a meaningful share of your code is resolving by name and no SCIP index has
been ingested.

## Doc comments everywhere

Symbols carry their **doc comments** across every language — godoc, rustdoc,
Javadoc, JSDoc, TSDoc, not just Python docstrings — so `semantic` search finds
code by what it *does*, even when the identifier is opaque.

## Check your install

```bash
tokengraph langs                    # languages by extraction tier
tokengraph diagnose-extractors      # self-test every extractor (CI gate)
```

## Next steps

- Find code by meaning: [Retrieval → semantic](./retrieval.md)
- Whole-repo language mix: `tokengraph arch` in the [CLI reference](./cli.md)
