# Contributing to ContextIQ

ContextIQ ships as a single module, `tokengraph_all.py` (registered as the
`tokengraph` MCP server). This guide covers local development and the
maintainer-only release process, so the release scripts live in one discoverable
place.

## Development setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e '.[all]'   # Windows
# .venv/bin/python -m pip install -e '.[all]'          # macOS / Linux
```

The `[all]` extra pulls `tree-sitter-language-pack`, `fastmcp`, `tiktoken`, and
`sentence-transformers`. With zero extras the CLI still runs (regex parsing +
heuristic token counts).

## Running the tests

```bash
# Fast, deterministic — matches CI (hash embeddings, no telemetry)
TOKENGRAPH_EMBEDDINGS=off TOKENGRAPH_NO_TRACK=1 python -m unittest tests.test_contextiq_all
```

`pytest -q` also works. CI (`.github/workflows/ci.yml`) runs the suite on
Ubuntu + Windows across Python 3.10 and 3.12.

## Quality gates (optional locally, useful before a release)

```bash
tokengraph benchmark --all --check          # retrieval quality; exit 1 below thresholds
tokengraph test-map --benchmark --check      # impl<->test F1; exit 1 below --min-f1
tokengraph health --strict                   # composite A-F grade; exit 1 if context is stale
```

## Making a change

1. Prefer the graph over reading whole files (`find_relevant_context`,
   `get_symbol`, `get_method_impact`, `get_test_map`) — see `CLAUDE.md`.
2. Keep the change in `tokengraph_all.py` unless a test needs a small update.
3. Add or update focused tests in `tests/test_contextiq_all.py`.
4. If extraction starts producing symbols/edges an existing index would not
   contain, bump `EXTRACTOR_GENERATION` so cached graphs rebuild.
5. Run the suite. Don't add required dependencies.

---

## Releasing (maintainers)

The build and publish helpers are **maintainer-only** and live outside the
shipped package (`tools/`) or as CLI subcommands. Run them from a git checkout,
in order:

### 1. Bump the version

Edit `version` in `pyproject.toml` (single source of truth). Update any
changelog notes.

### 2. Verify green

```bash
TOKENGRAPH_EMBEDDINGS=off python -m unittest tests.test_contextiq_all
tokengraph benchmark --all --check
```

### 3. Build + verify the wheel/sdist

```bash
python tools/build_dist.py            # wipes stale artifacts, builds, and verifies
                                      # every artifact matches the pyproject version
python tools/build_dist.py --check    # also run `twine check` (needs twine)
```

`dist/` is gitignored — the artifacts are release outputs, not committed.
`tools/build_dist.py` is the dev-only build wrapper (it needs the source tree,
so it is intentionally not a shipped `tokengraph` subcommand).

### 4. Publish to PyPI

```bash
python -m pip install twine            # if not already installed
python -m twine upload dist/*          # or `--repository testpypi` first
```

### 5. Tag the release

```bash
git tag -a v0.2.0 -m "ContextIQ 0.2.0"
git push origin v0.2.0
```

### 6. Optional: binaries and packaging scaffolds

```bash
tokengraph freeze --build              # standalone PyInstaller binary
tokengraph dist                        # scaffold release CI + Dockerfile + Homebrew + install.sh
```

### 7. Optional: archive the benchmark with a DOI

```bash
tokengraph publish-benchmark --full    # reproducible, content-hashed artifacts
tokengraph zenodo-publish --dry-run    # preview the deposition (no token)
tokengraph zenodo-publish --production --publish   # mint the permanent DOI
```

See `docs/BENCHMARK.md` for the full DOI runbook and the `zenodo-publish` safety
model (sandbox + draft by default; `--production --publish` required to mint).

---

## Where things live

| Concern | Command / file |
|---|---|
| The whole tool | `tokengraph_all.py` |
| Tests | `tests/test_contextiq_all.py` |
| Build wheel/sdist (dev) | `tools/build_dist.py` |
| Release CI / Docker / Homebrew scaffold | `tokengraph dist` |
| Standalone binary | `tokengraph freeze --build` |
| Reproducible benchmark + DOI | `tokengraph publish-benchmark` → `tokengraph zenodo-publish` |
| Editor wiring | `tokengraph ide-setup [--verify]` |
