---
title: When to use what
description: "A decision guide mapping common tasks to the right ContextIQ command — from understanding code to editing safely and proving savings."
head:
  - - meta
    - property: og:title
      content: "ContextIQ — when to use what"
---

# When to use what

A quick map from *what you're trying to do* to *the command that does it*.

## Understanding code

| You want to… | Use |
| --- | --- |
| Orient in an unfamiliar repo | `arch`, then `modules` |
| Get context for a specific task | `context "task"` or `ask "task"` |
| Find code when you don't know its name | `semantic "meaning"` |
| See one file's shape without bodies | `skeleton FILE` |
| Read an exact line range | `lines FILE 40 80` |

## Before editing

| You want to… | Use |
| --- | --- |
| Know what breaks if I change this function | `method-impact QNAME` |
| See a symbol's full blast radius | `impact QNAME` |
| Find the tests that cover this | `test-map TARGET` |
| Check a plan references real code | `verify-plan` |

## Generating code

| You want to… | Use |
| --- | --- |
| Match the repo's file conventions | `conventions`, then `scaffold` |
| Run the whole gated pipeline | `create "task"` |
| Audit AI output for fabrications | `verify` / `verify-output` |
| Review a diff before committing | `review` |

## Trusting an answer

| You want to… | Use |
| --- | --- |
| Confirm context is sufficient first | `validate "task" --min-coverage 0.6` |
| Score if the answer is grounded | `judge --answer-file … --context-file …` |
| Catch fabricated files / symbols | `verify --answer-file …` |

## Proving value

| You want to… | Use |
| --- | --- |
| See tokens + dollars saved | `gain` |
| Open a savings dashboard | `gain --report` |
| Produce an audit-grade evidence pack | `evidence "task"` |

## Next steps

- Full flag list: [CLI reference](./cli.md)
- The recommended everyday loop: [Quick start](./quick-start.md)
