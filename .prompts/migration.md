# Task: Migration

## Goal
Migrate {{FROM}} to {{TO}} across the codebase.

## Output
1. Every call site that must change (file:line)
2. A staged plan (independent batches, each independently shippable)
3. Risks + rollback
4. Verified by {{TEST_COMMAND}}

## Instructions
1. `search_semantic("{{FROM}}")` and `get_callers("{{FROM}}")` to enumerate the full blast radius — do not guess the count.
2. `get_method_impact("{{FROM}}")` to see what breaks on a signature change and which tests cover it.
3. Batch the edits by module; keep each batch green under `{{TEST_COMMAND}}`.
4. After each batch, `review --staged` to catch scope-drift and breaking changes (removed symbols with live callers).
