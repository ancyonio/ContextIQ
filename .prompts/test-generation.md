# Task: Test Generation

## Goal
Generate comprehensive unit or integration tests for a new or modified symbol.

## Output
1. Test File
2. Edge Cases Covered
3. Mocking Strategy
4. Verification Command

## Instructions
1. Inspect the target symbol using `get_symbol`.
2. Inspect existing test patterns using `search_semantic` or `file_skeleton` on existing `test_*.py` files.
3. Ensure the generated tests are runnable and follow project conventions.
