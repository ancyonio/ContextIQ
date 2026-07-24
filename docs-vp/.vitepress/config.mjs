import { defineConfig } from 'vitepress'

// This VitePress project renders the Markdown authored in the repo-root
// `guide/` folder. That folder is the single source of truth; a prebuild step
// (`sync-guide.mjs`, wired via the `predocs:*` npm scripts) copies it into
// `docs-vp/guide/` — which is git-ignored — so Vite can resolve its own
// dependencies (a hard requirement: srcDir must live inside the project root).
//
//   npm install
//   npm run docs:dev       # local preview at http://localhost:5173
//   npm run docs:build     # static HTML -> docs-vp/dist
//
// GitHub Pages serves from a sub-path (https://<user>.github.io/contextiq/),
// so `base` defaults to '/contextiq/'. Override for a custom domain / user page:
//   DOCS_BASE=/ npm run docs:build
export default defineConfig({
  title: 'ContextIQ',
  description: 'Token-efficient code context for AI coding agents — retrieve, validate, judge, and verify, all offline.',
  lang: 'en-US',

  // Synced copy of repo-root guide/ (see sync-guide.mjs). Never edit in place.
  srcDir: 'guide',
  srcExclude: ['README.md'],
  outDir: './dist',
  cacheDir: './.vitepress/cache',

  base: process.env.DOCS_BASE || '/contextiq/',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,

  head: [
    ['meta', { name: 'theme-color', content: '#3c8772' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'ContextIQ' }]
  ],

  themeConfig: {
    nav: [
      { text: 'Guide', link: '/quick-start' },
      { text: 'CLI', link: '/cli' },
      { text: 'Benchmark', link: '/benchmark' },
      {
        text: 'v1.0',
        items: [
          { text: 'GitHub', link: 'https://github.com/contextiq/contextiq' },
          { text: 'Zenodo DOI', link: 'https://doi.org/10.5281/zenodo.21535772' }
        ]
      }
    ],

    sidebar: [
      {
        text: 'Introduction',
        items: [
          { text: 'What is ContextIQ?', link: '/' },
          { text: 'Quick start', link: '/quick-start' },
          { text: 'When to use what', link: '/when-to-use' }
        ]
      },
      {
        text: 'The workflow',
        items: [
          { text: 'Retrieval', link: '/retrieval' },
          { text: 'Validate', link: '/validate' },
          { text: 'Judge', link: '/judge' },
          { text: 'Verify', link: '/verify' },
          { text: 'Conventions & scaffolding', link: '/conventions' }
        ]
      },
      {
        text: 'Integrate',
        items: [
          { text: 'MCP server & editor wiring', link: '/mcp' },
          { text: 'Local LLMs & offline use', link: '/local-llms' },
          { text: 'Languages', link: '/languages' }
        ]
      },
      {
        text: 'Prove it',
        items: [
          { text: 'Savings & dashboard', link: '/savings' },
          { text: 'Benchmark & evidence', link: '/benchmark' }
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'CLI reference', link: '/cli' },
          { text: 'Troubleshooting', link: '/troubleshooting' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/contextiq/contextiq' }
    ],

    search: { provider: 'local' },

    editLink: {
      pattern: 'https://github.com/contextiq/contextiq/edit/main/guide/:path',
      text: 'Edit this page on GitHub'
    },

    footer: {
      message: 'Released under the terms in the repository LICENSE.',
      copyright: 'ContextIQ — model-agnostic, local, offline.'
    }
  }
})
