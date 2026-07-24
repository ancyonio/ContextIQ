# ContextIQ Guide

Source Markdown for the ContextIQ documentation. Each file is authored to render
cleanly on GitHub **and** to be built into an HTML docs site (e.g. VitePress).

## Contents

| Guide | What it covers |
| --- | --- |
| [What is ContextIQ?](./index.md) | The core idea and what makes it different |
| [Quick start](./quick-start.md) | Install and run the real workflow in minutes |
| [Retrieval](./retrieval.md) | `context` · `ask` · `semantic` · `lines` · `federated` |
| [Validate](./validate.md) | The coverage gate before an agent acts |
| [Judge](./judge.md) | Score whether an answer is grounded |
| [Verify](./verify.md) | Catch fabricated files / symbols / imports |
| [Conventions & scaffolding](./conventions.md) | Grounded code generation |
| [MCP server & editor wiring](./mcp.md) | Wire into Claude Code, Cursor, VS Code, … |
| [Savings & dashboard](./savings.md) | The ledger and the `gain` dashboard |
| [Benchmark & evidence](./benchmark.md) | Reproducible grounding benchmark |
| [Languages](./languages.md) | 25+ deep-parsed, 30+ regex-indexed |
| [Local LLMs & offline use](./local-llms.md) | Model-agnostic, no API key |
| [When to use what](./when-to-use.md) | Task → command decision guide |
| [CLI reference](./cli.md) | Every command, grouped by purpose |
| [Troubleshooting](./troubleshooting.md) | Common issues and fixes |

## Building the HTML site

These files use portable relative links (`./name.md`) and YAML frontmatter
(`title`, `description`, `head` OG tags) so a static-site generator can turn them
into HTML without edits. See the project README / repo root for the docs build
setup.
