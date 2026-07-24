# Prompt templates

Short, reusable task prompts that keep AI coding assistants focused and
token-cheap. Each one leads with the outcome and drives the work through
ContextIQ tools (`find_relevant_context`, `get_method_impact`, …) instead of
whole-file reads.

| Template | Use for |
|---|---|
| [bug-fix.md](bug-fix.md) | Root-cause and minimally fix a failure |
| [feature.md](feature.md) | Implement a feature against acceptance criteria |
| [refactor.md](refactor.md) | Behavior-preserving change with a blast-radius check |
| [performance.md](performance.md) | Find and remove a hot path |
| [security-review.md](security-review.md) | Severity-ranked security review |
| [migration.md](migration.md) | Blast-radius-driven, staged migration |
| [code-review.md](code-review.md) | Risk-first review of a diff |
| [test-generation.md](test-generation.md) | Targeted test creation |
| [architecture-review.md](architecture-review.md) | Design / architecture review |

## Variable substitution

Placeholders come in two forms:

- **`{{UPPER_SNAKE}}`** — a substitutable variable. Fill it programmatically
  (e.g. `sed 's/{{TEST_COMMAND}}/pytest -q/g'`) or by hand before sending.
  The same token repeated in a file always means the same value.
- **`<freeform>`** — a human placeholder: replace the whole thing, including the
  angle brackets, with prose.

Keeping the machine-substitutable inputs as `{{VARS}}` lets a harness template a
prompt without a templating engine — a plain find-and-replace is enough.
