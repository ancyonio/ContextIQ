---
title: Savings & dashboard
description: "Every retrieval appends to a privacy-safe ledger; gain rolls it up into a token + dollar report with trends and a self-contained HTML dashboard."
head:
  - - meta
    - property: og:title
      content: "ContextIQ savings ledger & gain dashboard"
---

# Savings & dashboard

ContextIQ makes savings **provable**. Every `context` / `ask` / `measure` /
`generate` call (and the MCP context tool) appends its pack-vs-whole-file delta
to a privacy-safe, count-only **ledger**.

## Roll it up

```bash
tokengraph gain                       # tokens + $ saved, all time
tokengraph gain --since 30d --all     # last 30 days + daily/weekly/monthly trends
tokengraph gain --model claude-sonnet # price the projection with a specific model
```

## The HTML dashboard

```bash
tokengraph gain --report              # writes .tokengraph/token-usage.html
tokengraph gain --html board.html     # self-contained file anywhere
tokengraph gain --serve --port 8080   # live dashboard on 127.0.0.1 (no Streamlit)
```

The `--report` / `--html` output is the dashboard shown in the project README —
a single self-contained file you can open or embed.

## Metrics for your stack

```bash
tokengraph gain --prometheus --out ciq.prom   # Prometheus/OpenMetrics text
tokengraph gain --grafana --out board.json    # importable Grafana dashboard
```

Scrape the `contextiq_*` metrics into Grafana / OTel, or drop the `.prom` file
into a node_exporter textfile collector.

## One-line snapshot

```bash
tokengraph status
```

Branch, index freshness, notes, and cumulative savings in a single line — handy
in a prompt or a shell banner.

## Measure a single task

```bash
tokengraph measure "add retry logic"           # savings vs a grep+read baseline
tokengraph report --tasks-file tasks.txt -o report.md --csv report.csv
```

## Next steps

- Understand what's counted: [Retrieval](./retrieval.md)
- Audit-grade evidence: `evidence` and [Benchmark](./benchmark.md)
