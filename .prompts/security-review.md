# Task: Security Review

## Scope
Review {{TARGET}} for security issues (auth, input handling, secrets, injection).

## Output
1. Findings ranked by severity (with file:line)
2. Exploit sketch for each high/critical
3. Minimal remediation per finding
4. What was checked and found clean

## Instructions
1. `find_relevant_context("{{TARGET}} authentication input validation")` to pull the security-relevant code.
2. `get_map(routes)` to enumerate entry points that take untrusted input.
3. `get_callers(qname)` on sinks (db exec, shell, file, template) to trace tainted paths to a source.
4. Do not paste secrets; ContextIQ redacts them — reference file:line instead.
5. Prefer the smallest fix that closes the class of bug, not just the instance.
