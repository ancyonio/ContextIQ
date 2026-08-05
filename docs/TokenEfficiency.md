# Token Optimization and Efficiency Guide

## GitHub Copilot, Claude Code, and ContextIQ

This guide explains how to reduce token usage, improve answer quality, and keep AI coding agents focused when working with GitHub Copilot (GHCP), Claude Code, and the local ContextIQ MCP server (registered as `tokengraph`).

The core idea is simple: give the agent the smallest useful slice of context, keep that context fresh, and ask for a specific outcome.

---

## 1. Token Efficiency Mindset

### 1.1 Optimize for relevant context, not maximum context

Large prompts often feel safer, but they usually make the model slower and less precise. The best prompt contains only the facts needed for the current task.

Avoid:

```text
Here is my entire codebase. Tell me what is wrong.
```

Prefer:

```text
Task: Root cause the failed settlement validation.

Relevant files:
- settlement_service.py
- validation_rules.py
- test_settlement_validation.py

Observed behavior:
FX trades with same-day settlement fail validation.

Expected behavior:
Same-day settlement should pass when currency pair is eligible.

Output:
1. Root cause
2. Minimal fix
3. Tests to add or update
```

Benefits:

- Fewer irrelevant tokens
- Faster responses
- Better code localization
- Less chance of broad, unfocused refactoring

### 1.2 Prefer references over pasted content

When the code already exists in the workspace, reference it instead of pasting it.

Use:

```text
Review @settlement_service.py and @test_settlement_validation.py for this failing scenario.
```

Instead of:

```text
<paste hundreds of lines of source code>
```

This is especially important in GHCP and Claude Code because both can use workspace files, MCP tools, and repo search. Pasting code duplicates context and can make the agent ignore fresher file contents.

### 1.3 Ask for an outcome, not a topic

Avoid:

```text
Analyze this.
```

Prefer:

```text
Find the smallest code change that fixes the failing retry behavior. Update tests and explain the risk.
```

Good task prompts specify:

- The goal
- The current symptom
- The files, commands, or logs involved
- The output format
- Any constraints, such as no public API changes or no new dependencies

### 1.4 Keep reusable context outside the chat

Do not repeatedly paste stable project information. Store it in repo files that agents can reference.

Recommended files:

- `.github/copilot-instructions.md` for GHCP workspace behavior
- `CLAUDE.md` for Claude Code behavior
- `README.md` for project setup and commands
- Architecture notes for long-lived design context
- Prompt templates for repeatable workflows

This repo already uses `.github/copilot-instructions.md` and `CLAUDE.md` to tell agents to use ContextIQ first.

One command wires both the MCP server and these steering files: `python tokengraph_all.py ide-setup` — by default into Claude Code + VS Code/Copilot (plus any editor detected in the repo); add `--all` for every editor, `--verify` to prove each is wired, `--global` for Windsurf/Cline. ContextIQ is **model-agnostic and offline** — it emits context packs and never calls an LLM, so the same setup serves cloud agents (Copilot, Claude Code, Cursor) and local models (Ollama, llama.cpp, vLLM) alike.

---

## 2. Use ContextIQ First

ContextIQ is the main token-saving mechanism in this repository. It indexes code into a local graph and returns small, task-focused context packs instead of forcing the agent to read whole files.

### 2.1 Default lookup flow

Use this order when exploring code:

1. `find_relevant_context(task)` - start here for most coding tasks.
2. `search_semantic(query)` - use when you know the behavior but not the symbol name. Cross-language doc comments (godoc/rustdoc/Javadoc/JSDoc/TSDoc) are indexed, so opaque identifiers still match by meaning.
3. `get_symbol(qname)` - fetch the full source for one specific symbol.
4. `get_callers(qname)` and `get_callees(qname)` - trace dependencies.
5. `get_method_impact(qname)` - function-level blast radius before an edit: who breaks (with call sites), dependencies, and overrides.
6. `get_test_map(target)` - the tests for a file/symbol (naming + call graph); omit the target for the whole-repo impl↔test map and coverage %.
7. `get_architecture_overview()` - one-call orientation: modules, hub files, import cycles, language mix, and route totals.
8. `file_skeleton(file)` - inspect signatures without reading bodies.
9. `get_module_summary(file)` - understand a file in a few tokens.
10. `estimate_savings(task)` - compare context pack size with full-file reading.

Only open full files when you need to edit them or when the context pack explicitly drops something required.

### 2.2 CLI examples

From the repository root:

```bash
python tokengraph_all.py index
python tokengraph_all.py context "fix stale module summary invalidation" -b 4000
python tokengraph_all.py semantic "call graph lookup"
python tokengraph_all.py skeleton tokengraph_all.py
python tokengraph_all.py measure "add retry support to MCP server startup"
python tokengraph_all.py method-impact tokengraph_all.Retriever.get_impact  # who breaks / deps / call sites
python tokengraph_all.py arch                                  # whole-repo overview in one call
python tokengraph_all.py test-map                              # implementations <-> tests + coverage %
python tokengraph_all.py repomix --out pack.xml                # export the signature map as a Repomix pack
```

For another repository:

```bash
python tokengraph_all.py --path "D:/work/other-repo" context "find auth middleware" -b 4000
```

### 2.3 Recommended budgets

Use smaller budgets first, then expand only when needed.

| Task type | Suggested budget |
|---|---:|
| Locate a symbol or behavior | 1500-3000 tokens |
| Small bug fix | 3000-5000 tokens |
| Medium feature | 5000-8000 tokens |
| Architecture or migration analysis | 8000-12000 tokens |

If a context pack says important symbols were dropped, request those symbols directly instead of increasing the budget blindly.

### 2.4 Freshness model

ContextIQ refreshes before queries, so agents should not assume the graph is stale after edits.

Freshness options:

- Query-time refresh: default correctness layer for CLI and MCP calls.
- Claude Code hook: `.claude/settings.json` can pre-warm the graph after edits.
- Watch mode: `python tokengraph_all.py watch` keeps the graph warm for any client.

For GHCP, watch mode can be useful because Copilot does not have Claude Code's edit hook mechanism.

---

## 3. GitHub Copilot (GHCP) Efficiency

### 3.1 Use the right mode

Use Ask mode for:

- Explanations
- Architecture questions
- Understanding errors
- Comparing approaches

Use Agent or Edit mode for:

- Code changes
- Refactoring
- Test creation
- Multi-file fixes
- Running verification commands

Do not paste large code blocks when the files are in the workspace. Ask GHCP to inspect or edit the files directly.

### 3.2 Control workspace noise

GHCP may draw context from open editors, recent changes, visible problems, and workspace metadata. Keep the working set clean.

Helpful habits:

- Close unrelated large files, generated output, and logs.
- Keep the failing test and target source file open.
- Reference specific files when scope matters.
- Include the exact command that reproduces a failure.
- Mention constraints before the agent starts editing.

Example:

```text
Fix the failure from `python -m pytest tests/test_contextiq_all.py -q`.
Scope the change to tokengraph_all.py unless the test needs a small update.
Use the existing graph-refresh pattern; do not add dependencies.
```

### 3.3 Use repo instructions for repeated rules

Put durable rules in `.github/copilot-instructions.md` instead of repeating them in every prompt.

Good instruction topics:

- Preferred tools and context strategy
- Test commands
- Code style constraints
- Project-specific architecture boundaries
- Generated files to avoid editing

For this repo, the most important GHCP rule is: use ContextIQ tools before opening whole files.

### 3.4 Ask for verification explicitly

Efficient prompts include the validation step.

```text
Implement the fix and run the narrowest relevant test. If the test cannot run, explain why.
```

This avoids a second prompt asking whether the change was checked.

---

## 4. Claude Code Efficiency

### 4.1 Use `CLAUDE.md` as persistent project context

Claude Code reads project guidance well. Keep stable operating rules in `CLAUDE.md`:

- How to inspect the codebase
- Which commands verify changes
- Which tools should be preferred
- Any project-specific constraints

This repo's `CLAUDE.md` tells Claude Code to use `find_relevant_context` first and open full files only when necessary.

### 4.2 Use structured XML for complex work

Claude handles tagged prompts well when the request has multiple parts.

```xml
<task>
Fix incorrect token savings reporting.
</task>

<context>
Use ContextIQ context first. The failure is in the CLI `measure` command.
</context>

<constraints>
- Keep the public CLI flags unchanged.
- Do not add dependencies.
- Add or update focused tests only.
</constraints>

<output>
Summarize the root cause, changed files, and test result.
</output>
```

### 4.3 Compress long sessions

At the end of a long session, ask for a compact continuation summary.

```text
Summarize this session in 400 words or less with:
- Decisions
- Files changed
- Commands run
- Remaining risks
- Best next prompt
```

Store durable project learnings in `CLAUDE.md` only when they should affect future sessions. Keep temporary task state in the chat or session notes.

### 4.4 Use phased exploration for large tasks

For large architecture or migration work, do not start with every file.

Better flow:

1. Ask for a code graph context pack.
2. Review module summaries and skeletons.
3. Fetch only the highest-value symbols.
4. Edit the exact files needed.
5. Run narrow verification.
6. Broaden tests only when the blast radius requires it.

---

## 5. Prompt Patterns That Save Tokens

### 5.1 Bug fix prompt

```text
Task: Fix a bug.

Symptom:
<what failed>

Reproduction:
<command or steps>

Expected behavior:
<what should happen>

Constraints:
- Keep the fix minimal.
- Follow existing patterns.
- Add or update focused tests.

Output:
Root cause, files changed, verification result.
```

### 5.2 Code review prompt

```text
Review the current diff for bugs, regressions, missing tests, and maintainability risks.
Prioritize findings by severity and include file references.
Do not rewrite the code unless I ask.
```

### 5.3 Feature implementation prompt

```text
Implement <feature>.

Acceptance criteria:
- <criterion 1>
- <criterion 2>
- <criterion 3>

Constraints:
- Preserve existing public APIs unless needed.
- Reuse existing helpers.
- Add focused tests.

Verification:
Run <test command>.
```

### 5.4 Architecture prompt

```text
Create an implementation plan for <goal>.

Context:
- Current stack: <stack>
- Target behavior: <target>
- Constraints: <constraints>

Output:
1. Proposed design
2. Impacted modules
3. Risks
4. Incremental tasks
5. Validation plan
```

### 5.5 Continuation prompt

```text
Continue from this summary:
<compact session summary>

Next task:
<specific next step>

Use the repo instructions and inspect current files before editing.
```

---

## 6. Context Layering for Agentic AI Projects

Separate context by how often it changes.

### Layer 1: Stable platform context

Rarely changes.

```text
Platform: Azure OpenAI
Framework: LangGraph
API: FastAPI
UI: Streamlit
Vector store: ChromaDB
Cloud: Azure
```

Store this in project docs or instructions.

### Layer 2: Domain context

Changes occasionally.

```text
Domain: Wholesale banking
Business area: Trade finance
Use case: Settlement exception agent
Regulatory constraints: Auditability, human approval, data retention
```

Store this in design docs or reusable prompt templates.

### Layer 3: Current task context

Changes every prompt.

```text
Current task: Add auto-remediation for eligible settlement exceptions.
Failure mode: Agent retries ineligible trades.
Verification: Run settlement workflow tests.
```

Put this in the active chat prompt.

Benefits:

- Less repeated context
- Cleaner prompts
- Better reuse across GHCP and Claude Code
- Easier handoff between architecture and implementation

---

## 7. Recommended GHCP + Claude Code Workflow

Use each tool where it is strongest, but keep the handoff compact.

| Stage | Best tool | Token-efficient handoff |
|---|---|---|
| Business problem framing | Claude Code | One-page problem summary |
| Architecture options | Claude Code | Decision table and tradeoffs |
| Implementation plan | Claude Code or GHCP | Task list with files and tests |
| Code changes | GHCP agent mode | File references and acceptance criteria |
| Debugging | GHCP or Claude Code | Error snippet plus reproduction command |
| Review | GHCP or Claude Code | Current diff and review criteria |
| Documentation | Claude Code | Final design and implementation summary |

Example flow:

```text
Business problem
  -> Claude Code: architecture and tradeoffs
  -> compact plan
  -> GHCP: implementation and tests
  -> GHCP or Claude Code: review
  -> Claude Code: final documentation
```

Do not move the whole conversation between tools. Move only decisions, constraints, file names, and the next action.

---

## 8. Measuring Efficiency

Track whether the workflow is actually saving tokens and time.

### 8.1 Practical metrics

Useful indicators:

- Number of files opened before the first useful answer
- Context pack size vs. whole-file size
- Number of clarification rounds
- Time to first patch
- Test pass/fail turnaround time
- Number of unrelated files changed

### 8.2 ContextIQ savings check

Use:

```bash
python tokengraph_all.py measure "fix module summary invalidation"
```

Look for:

- Pack tokens
- Equivalent full-file tokens
- Percentage saved
- Dropped symbols that need direct follow-up

### 8.2.1 Aggregate report across many tasks

`measure` covers one task. To produce a quantitative with/without report over a
representative set of tasks, use `report`:

```bash
python tokengraph_all.py report --tasks-file tasks.txt -o report.md --csv report.csv

# --append accumulates instead of overwriting: the CSV header is written once
# and each run adds its rows, while -o becomes a timestamped running log.
python tokengraph_all.py report --tasks-file tasks.txt -o runs.md --csv runs.csv --append
```

- `tasks.txt` is one task per line (`#` comments allowed); you can also pass
  tasks as arguments.
- The markdown report has three sections: a **repo baseline** (whole-repo
  tokens and how a typical pack compares at repo scale), an **aggregate**
  rollup (totals, overall and mean savings %, best/worst task), and a
  **per-task** table.
- `--csv` writes the per-task rows for spreadsheets or trend tracking.
- The same data is available to agents via the `savings_report(tasks)` MCP tool.

> Note: the "without" baseline is a proxy — the token cost of opening every
> distinct file each pack draws from (what a naive agent would read for the same
> coverage). It is not a capture of real session token logs.

### 8.3 Expected savings

| Technique | Typical savings |
|---|---:|
| File references instead of pasted code | 50%-90% |
| ContextIQ context packs | 60%-95% |
| Session summaries | 70%-95% |
| Prompt templates | 20%-50% |
| Smaller open working set in GHCP | 20%-60% |
| Layered context model | 50%-80% |

Savings vary by repository size, task scope, and how well the prompt identifies the target behavior.

---

## 9. Anti-Patterns

Avoid these habits because they waste tokens or increase risk:

- Pasting whole files that already exist in the workspace
- Asking broad questions like "analyze everything"
- Keeping unrelated generated files open during GHCP sessions
- Repeating stable architecture context in every prompt
- Asking for implementation before stating acceptance criteria
- Expanding context budget before checking dropped symbols
- Mixing multiple unrelated tasks into one agent turn
- Asking the agent to refactor and debug at the same time without priority
- Moving full transcripts between Claude Code and GHCP instead of summaries

---

## 10. Quick Checklists

### Before asking GHCP to edit code

- Is the task outcome specific?
- Are the relevant files or symbols named?
- Is the reproduction command included?
- Are constraints stated clearly?
- Is the expected verification command included?

### Before asking Claude Code for architecture or planning

- Is stable context in a file instead of pasted into chat?
- Are business constraints separated from implementation details?
- Is the requested output format clear?
- Is the desired level of detail stated?
- Is there a clear next action after the plan?

### Before increasing context size

- Did `find_relevant_context` return dropped symbols?
- Can a specific symbol be fetched with `get_symbol`?
- Would a file skeleton be enough?
- Would a module summary be enough?
- Is the extra context needed for the current decision?

---

## Key Takeaways

1. Start with the smallest useful context.
2. Use ContextIQ before reading whole files.
3. Reference workspace files instead of pasting code.
4. Put durable instructions in `.github/copilot-instructions.md` and `CLAUDE.md`.
5. Use GHCP for focused implementation and verification.
6. Use Claude Code for architecture, planning, synthesis, and complex reasoning.
7. Move summaries between tools, not entire conversations.
8. Measure savings with `estimate_savings` or the CLI `measure` command.

Applied consistently, these practices can reduce token usage by 50%-90% while improving answer quality, reviewability, and development speed.
