# ContextIQ docs site (VitePress)

This is the documentation site build. The **content** lives one level up in
[`../guide`](../guide) — this folder only holds the VitePress config and build
tooling, so there's a single source of truth for the Markdown.

## Develop locally

```bash
cd docs-vp
npm install
npm run docs:dev        # http://localhost:5173
```

## Build static HTML

```bash
npm run docs:build      # output -> docs-vp/dist
npm run docs:preview    # serve the built site locally
```

## How it's wired

- `srcDir` points at `../guide`, so every `guide/*.md` becomes a page.
- `base` defaults to `/contextiq/` for GitHub Pages project sites. For a custom
  domain or user/org page, build with `DOCS_BASE=/ npm run docs:build`.
- `.github/workflows/docs.yml` builds this project and deploys `dist/` to Pages
  on every push to `main` that touches `guide/**` or `docs-vp/**`.

## One-time GitHub setup

Repo **Settings → Pages → Build and deployment → Source = "GitHub Actions"**.
After the first successful run, the site is live at
`https://<owner>.github.io/contextiq/`.
