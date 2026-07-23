# Task: Code Review

## Goal
Review the current diff for bugs, regressions, and maintainability risks.

## Output
1. Critical Issues
2. Major Improvements
3. Minor Suggestions
4. Test Coverage Assessment

## Instructions
Check the @current_diff and reference specific lines in the source files.
Assess the blast radius with the ContextIQ call graph: `get_diff_context` for a budgeted pack of exactly what the diff touches, and `get_method_impact(qname)` for any changed function (who breaks, dependencies, overrides, tests touched).
