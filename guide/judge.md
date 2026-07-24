---
title: Judge
description: "Score whether an AI answer is actually grounded in the context you gave it — a 0.0–1.0 grounding score with PASS/FAIL."
head:
  - - meta
    - property: og:title
      content: "ContextIQ judge — is the answer grounded?"
---

# Judge

`judge` scores whether an AI answer is **supported by the code context** you
supplied — a cheap, local check for confident-but-unsupported answers.

## Use it

Save the context pack to a file with `-o`, capture the assistant's answer, then
judge one against the other:

```bash
tokengraph context "retry logic" -b 4000 -o context.md   # save the pack
# … paste context.md into your assistant, save its reply to response.txt …
tokengraph judge --answer-file response.txt --context-file context.md
```

Or pass strings inline:

```bash
tokengraph judge --answer "the retry helper lives in http/client.py" \
                 --context "$(tokengraph context 'retry logic' -b 4000)"
```

## Output

A grounding **score (0.0–1.0)** with a PASS/FAIL indication. A low score means
the answer asserts things the context doesn't back up — treat it as a prompt to
retrieve more or push back on the model.

## Flags

| Flag | Meaning |
| --- | --- |
| `--answer` / `--answer-file` | The answer to score (inline or file) |
| `--context` / `--context-file` | The context it should be grounded in |
| `--json` | Machine-readable output |

## Judge vs. Verify

- **Judge** asks *"is the reasoning supported by this context?"* (a soft score).
- **[Verify](./verify.md)** asks *"does every file/symbol it names actually exist?"*
  (a hard, exit-1 check).

Run both: judge catches hand-wavy answers, verify catches fabricated references.

## Next steps

- Hard-check references: [Verify](./verify.md)
- Quantify the whole guard over a corpus: [Benchmark](./benchmark.md)
