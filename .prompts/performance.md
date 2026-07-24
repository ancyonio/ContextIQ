# Task: Performance Investigation

## Symptom
{{SLOW_BEHAVIOR}} is slow / allocates too much. Observed: {{MEASUREMENT}}.

## Output
1. Hot path (symbols on the critical route)
2. Root cause of the cost
3. Minimal fix + expected improvement
4. Verified by {{BENCHMARK_COMMAND}}

## Instructions
1. `search_semantic("{{SLOW_BEHAVIOR}}")` to find the entry point when you don't know the symbol name.
2. `get_callees(qname)` from the entry point to trace the work it does downstream.
3. `get_callers(qname)` on the suspected hot symbol to learn how often it is reached.
4. Fetch only the hot symbols with `get_symbol`; avoid loading whole files.
5. Apply the smallest change that removes the cost; re-measure with `{{BENCHMARK_COMMAND}}`.
