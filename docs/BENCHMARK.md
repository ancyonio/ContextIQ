# Benchmark: methodology, reproduction & DOI archival

ContextIQ ships a reproducible benchmark. This is the complete end-to-end runbook: measure → publish artifacts → verify → mint a DOI.

## What is measured

- **Retrieval quality** — Recall@5, symbol recall, answerable rate, and irrelevant-token ratio across every corpus under `benchmarks/` (`tokengraph benchmark --all`).
- **Test discovery** — precision / recall / F1 / hit@1 of the implementation↔test mapping on the labeled `benchmarks/testmap/` corpus (`tokengraph test-map --benchmark`).
- **Hallucination guard** — grounding coverage + guard catch/specificity (`tokengraph publish-benchmark --full`).

## End-to-end runbook

```bash
# 0. install (single file; [all] pulls tree-sitter + tokenizers)
pip install 'contextiq[all]'

# 1. run the individual benchmarks (optional — step 2 runs them for you)
tokengraph benchmark --all            # retrieval quality across all corpora
tokengraph test-map --benchmark       # impl<->test precision/recall/F1/hit@1

# 2. build the publish-ready artifacts (REPORT.md, MANIFEST.json,
#    .zenodo.json, CITATION.cff). --full also runs the hallucination guard.
tokengraph publish-benchmark --full

# 3. verify reproducibility: MANIFEST.json's dataset_hash is a SHA-256
#    over every benchmark input, so an identical hash == identical dataset.
tokengraph publish-benchmark --json | grep dataset_hash   # compare to a peer's

# 4. edit .zenodo.json + CITATION.cff — set the author name / ORCID /
#    affiliation before archiving.

# 5. preview the Zenodo deposition (no token needed, nothing uploaded)
tokengraph zenodo-publish --dry-run

# 6. create a DRAFT on the safe sandbox and review it in the web UI
export ZENODO_TOKEN=...    # sandbox.zenodo.org/account/settings/applications
tokengraph zenodo-publish

# 7. mint the PERMANENT DOI on production (irreversible — needs both flags)
export ZENODO_TOKEN=...    # zenodo.org/account/settings/applications
tokengraph zenodo-publish --production --publish
```

The DOI Zenodo mints turns the benchmark from *reproducible* into *peer-archived*.

> **Archived:** this benchmark is published at [10.5281/zenodo.21535773](https://doi.org/10.5281/zenodo.21535773). Cite it with the metadata in [CITATION.cff](../CITATION.cff).

## `zenodo-publish` safety model

Every default is the safe one; each escalation is explicit:

| Guard | Default | Behavior |
|---|---|---|
| Target | sandbox | `sandbox.zenodo.org`; `--production` for the real site |
| State | draft (reversible) | `--publish` required to actually publish |
| DOI mint | off | needs `--production --publish` **together** — no accidental mint |
| Offline | refuses | blocked when `TOKENGRAPH_OFFLINE` is set |
| Token | none | `--token` or `ZENODO_TOKEN`; never logged; `--dry-run` needs none |

What gets uploaded: `benchmarks/REPORT.md`, `benchmarks/MANIFEST.json`, `CITATION.cff`, and `.zenodo.json`. `zenodo-publish --dry-run` prints the exact deposition plan (endpoints, files, metadata) before anything leaves your machine.
