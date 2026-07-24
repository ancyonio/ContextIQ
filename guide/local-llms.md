---
title: Local LLMs & offline use
description: "ContextIQ is a context layer, not a model — pipe a pack straight into Ollama, llama.cpp, LM Studio, or vLLM with no API key and no telemetry."
head:
  - - meta
    - property: og:title
      content: "ContextIQ with local LLMs — offline, no API key"
---

# Local LLMs & offline use

ContextIQ is **model-agnostic**: a context layer, not a model. It parses your
repo, ranks what matters, and emits a plain-text / Markdown (or JSON) pack. It
**never calls an LLM**, so it works with any model — cloud or local.

## Pipe a pack into a local model

```bash
tokengraph context "explain the auth flow" | ollama run llama3
```

No API key, no telemetry, nothing leaves the box. The same works with any
runner that reads a prompt on stdin — llama.cpp, LM Studio, vLLM.

## Why this matters

- **Privacy** — your code never goes to a third party just to build context.
- **Cost** — a tight pack keeps even large local-model prompts fast and cheap.
- **Portability** — swap models freely; the context layer doesn't change.

## Pair it with the trust gates

The [validate](./validate.md) / [judge](./judge.md) / [verify](./verify.md)
gates are also fully local — so you can run the *entire* retrieve → answer →
check loop offline, with a local model doing the answering.

## Model-aware costing (when you do use a cloud model)

```bash
tokengraph cost              # estimate input+output USD before sending
tokengraph cost --compare    # pick the cheapest model that's still sufficient
```

Covers GPT / Claude / Gemini / Llama pricing — deterministic and local.

## Next steps

- Wire it into an editor: [MCP server](./mcp.md)
- Keep prompts tight: `prompt-score`, `dedupe`, `summarize-chat` in the [CLI reference](./cli.md)
