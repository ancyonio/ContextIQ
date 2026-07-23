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

## Archive with a DOI (maintainer step)

The generated `.zenodo.json` and `CITATION.cff` are deposition-ready. Before uploading, set the author/ORCID/affiliation in both. Then, with a Zenodo token:

```bash
# create a deposition, upload REPORT.md + MANIFEST.json + the benchmarks/ tree, then publish
# see https://developers.zenodo.org/#deposstions — the upload needs your credentials, so it is intentionally not automated here.
```

The DOI Zenodo mints is what turns this from *reproducible* into *peer-archived*.
