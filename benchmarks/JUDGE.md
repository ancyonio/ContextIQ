# LLM-judged answer quality (`quality_retention`)

The retrieval benchmark proves the *required symbols and facts are present* in a
pack. It cannot prove a model *answers correctly* from them. `judge-eval` closes
that gap: it answers each held-out question twice — once from the ContextIQ pack,
once from the **full text of the files that contain the answer** — and has an
independent judge grade both against a rubric.

The headline metric is:

```
quality_retention = pack_score / full_score
```

**1.0 means compression cost no answer quality.**

## Where the artifact comes from

- **CI (weekly):** the `llm-quality` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
  runs on a `cron` schedule (it makes real API calls, so it never runs per-push)
  and uploads `judge-eval.json` as a build artifact via `actions/upload-artifact`.
  Download it from the run's **Artifacts** section.
- **Locally:**

  ```bash
  export ANTHROPIC_API_KEY=...          # or: ant auth login
  python tokengraph_all.py judge-eval -o judge-eval.json --json
  ```

## Latest archived result

> _Not yet committed._ `quality_retention` requires live model calls (real
> cost), so it is produced by the scheduled CI job / a local run, not by the
> deterministic offline suite. Drop the newest `judge-eval.json` next to this
> file and record its headline here:
>
> | Date | Model | Questions | pack_score | full_score | quality_retention |
> |---|---|--:|--:|--:|--:|
> | — | — | — | — | — | — |

Keeping the number here (rather than only as an ephemeral CI artifact) makes the
central "fewer tokens, same answer quality" claim auditable over time.
