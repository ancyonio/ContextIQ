# Task: Safe Refactor

## Goal
Refactor {{TARGET}} to {{OUTCOME}} without changing observable behavior.

## Constraints
- No behavior change; no public API change unless stated.
- Keep the diff scoped to what the refactor requires.

## Output
1. Blast radius (callers / subclasses / tests that must still pass)
2. Refactor
3. Verified by {{TEST_COMMAND}}

## Instructions
1. `get_symbol("{{TARGET}}")` to read the current implementation.
2. `get_method_impact("{{TARGET}}")` to enumerate call sites, dependencies, overrides, and tests — this is the safety net for a behavior-preserving change.
3. `get_test_map("{{TARGET}}")` to find the tests that pin current behavior; run them first as a baseline.
4. Apply the refactor, keeping call-site signatures stable.
5. Re-run the baseline tests; `review --staged` to confirm no scope-drift or breaking changes.
