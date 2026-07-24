---
title: Conventions & scaffolding
description: "Close the loop from retrieval to safe code generation — detect house style, scaffold convention-matched files, and orchestrate a gated create pipeline."
head:
  - - meta
    - property: og:title
      content: "ContextIQ grounded generation — conventions, scaffold, create"
---

# Conventions & scaffolding

Retrieval finds the right context; **grounded generation** makes sure new code
fits the repo it's joining. ContextIQ detects your house style and refuses to
generate anything that conflicts with it.

## Detect house style

```bash
tokengraph conventions
```

Reports naming, layout, test, and export conventions — plus a **conformance**
read on how consistently the repo follows them. Auto-rename outliers with
`conventions --fix`.

## Scaffold a convention-matched file

```bash
tokengraph scaffold user_service --kind module
tokengraph scaffold user_service --kind module --apply
```

Proposes a file placed and named to match your conventions, with a skeleton.
`--apply` writes it and **refuses on conflict** — it never overwrites.

Kinds: `module`, `class`, `function`, `component`, `test`.

## Orchestrate the whole pipeline

```bash
tokengraph create "add a rate limiter to the http client" --apply
```

`create` runs the gated pipeline end to end:

```
scaffold → verify-plan → verify-output → review
```

Without `--apply` it's a **dry run** — you see the plan and diff before anything
touches disk. Pass `--answer-file` to feed generated code into the
`verify-output` stage.

## Review a diff

```bash
tokengraph review
```

Audits the working/staged diff for **scope drift**, **hub edits**, and **missing
tests** — a fast second pair of eyes before you commit.

## Next steps

- Understand the guards it uses: [Verify](./verify.md)
- Reference every flag: [CLI](./cli.md)
