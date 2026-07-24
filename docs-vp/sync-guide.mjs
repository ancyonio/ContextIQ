// Copy the authoritative repo-root guide/ into docs-vp/guide/ before building.
//
// VitePress requires its content (srcDir) to live inside the project root so
// Vite can resolve `vue` / `vitepress` for each page. The source of truth stays
// at repo-root guide/; this makes a fresh, git-ignored copy on every build.
import { cpSync, rmSync, mkdirSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))
const src = resolve(here, '..', 'guide')
const dest = resolve(here, 'guide')

if (!existsSync(src)) {
  console.error(`[sync-guide] source folder not found: ${src}`)
  process.exit(1)
}

rmSync(dest, { recursive: true, force: true })
mkdirSync(dest, { recursive: true })
cpSync(src, dest, { recursive: true })
console.log(`[sync-guide] copied ${src} -> ${dest}`)
