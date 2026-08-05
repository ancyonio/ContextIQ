---
title: Validate
description: "Gate coverage before an agent acts — validate returns a coverage score and exits non-zero when context is too thin."
head:
  - - meta
    - property: og:title
      content: "ContextIQ validate — the coverage gate"
---

# Validate

`validate` answers one question: **is this context enough to act on?** It's the
guardrail between retrieval and action.

## Basic use

```bash
tokengraph validate "auth login token"
```

It retrieves a pack for the task and reports a **coverage** score.

## Set a threshold (CI / hook gate)

```bash
tokengraph validate "auth login token" --min-coverage 60
```

Exits **non-zero** when coverage falls below the threshold — so an agent or a CI
step can refuse to proceed on thin context instead of guessing.

## Flags

| Flag | Meaning |
| --- | --- |
| `-b, --budget` | Token budget for the trial pack |
| `-d, --depth` | Graph traversal depth |
| `--min-coverage` | Fail below this coverage percentage (0–100, default 60) |
| `--json` | Machine-readable output |

## Where it fits

```
ask → validate → (AI answers) → judge → verify
        ▲
   stop here if coverage is low —
   refine the task or widen the budget
```

If coverage is low, the usual fixes are: make the task string more specific,
raise `--budget`, or increase `--depth`.

## Next steps

- After the answer comes back: [Judge](./judge.md) and [Verify](./verify.md)
- Wire it into a `PreToolUse` hook — see [MCP server](./mcp.md)
