---
title: Languages
description: "Deep call/import/inheritance graphs for 25+ languages via tree-sitter and Python's ast, plus regex indexing for 30+ more."
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
Haskell · OCaml · Nim · PowerShell · Dart.

Every deep-parsed language **falls back to regex** when its tree-sitter grammar
isn't installed.

## Regex index — 30+ more languages

Lightweight symbol indexing for:
SQL · Elixir · Clojure · F# · Groovy · Zig · Crystal · Haxe · Objective-C ·
Visual Basic · Tcl · Pascal · GDScript, and markup/config — Vue, Svelte, HTML,
CSS, YAML, TOML, XML, INI, GraphQL, Terraform, Protobuf, **Markdown**,
Dockerfile, and more.

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
