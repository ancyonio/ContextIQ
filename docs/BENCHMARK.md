# Benchmark: methodology & how to archive

ContextIQ ships a reproducible benchmark. This document explains how to reproduce it and how to archive it with a DOI.

## What is measured

- **Retrieval quality** — Recall@5, symbol recall, answerable rate, and irrelevant-token ratio across every corpus under `benchmarks/` (`tokengraph benchmark --all`).
- **Test discovery** — precision / recall / F1 / hit@1 of the implementation↔test mapping on the labeled `benchmarks/testmap/` corpus (`tokengraph test-map --benchmark`).
- **Hallucination guard** — grounding coverage + guard catch/specificity (`tokengraph publish-benchmark --full`).

## Reproduce

```bash
pip install 'contextiq[all]'
tokengraph publish-benchmark --full
```

Then confirm `benchmarks/MANIFEST.json`'s `dataset_hash` matches — it is a SHA-256 over every benchmark input, so an identical hash proves an identical dataset.

## Archive with a DOI

The generated `.zenodo.json` and `CITATION.cff` are deposition-ready (set the author/ORCID/affiliation first). `tokengraph zenodo-publish` deposits the artifacts and mints the DOI directly — sandbox and *draft* by default, so nothing is permanent until you opt in:

```bash
# 1. dry run — see exactly what will be uploaded, no token needed
tokengraph zenodo-publish --dry-run

# 2. create a DRAFT on the safe sandbox to review it
export ZENODO_TOKEN=...        # from sandbox.zenodo.org/account/settings/applications
tokengraph zenodo-publish

# 3. mint the PERMANENT DOI on production (irreversible)
export ZENODO_TOKEN=...        # from zenodo.org/account/settings/applications
tokengraph zenodo-publish --production --publish
```

The DOI Zenodo mints is what turns this from *reproducible* into *peer-archived*. Minting requires `--production --publish` together, so a stray run can never publish by accident.
