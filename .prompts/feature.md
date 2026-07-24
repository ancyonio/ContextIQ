# Task: Feature Implementation

## Goal
Implement {{FEATURE}}.

## Acceptance Criteria
- {{CRITERION_1}}
- {{CRITERION_2}}

## Constraints
- Preserve existing public APIs unless a criterion requires a change.
- Reuse existing helpers; do not add dependencies.
- Add focused tests only.

## Output
1. Design (files touched + why)
2. Implementation
3. Verified by {{TEST_COMMAND}}

## Instructions
1. `find_relevant_context("{{FEATURE}}")` to locate the code the feature extends.
2. `get_architecture_overview()` if the feature crosses modules, to place it correctly.
3. `conventions()` before creating any file, then `scaffold(name)` for a house-style skeleton.
4. Before editing a shared function, `get_method_impact(qname)` to size the blast radius.
5. Implement, then run `{{TEST_COMMAND}}`. Broaden tests only if the impact requires it.
