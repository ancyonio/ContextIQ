# Task: Bug Fix

## Symptom
Describe the error message, failing test, or incorrect behavior.

## Output
1. Root Cause
2. Minimal Fix
3. Verified by <command>

## Instructions
1. Use `find_relevant_context` with the symptom as the task to find the buggy code.
2. Confirm the root cause by inspecting the identified symbols.
3. Before changing a function, run `get_method_impact(qname)` to see who breaks (call sites), its dependencies, and the tests that exercise it.
4. Propose a small, targeted fix.
5. Run validation before finishing.
