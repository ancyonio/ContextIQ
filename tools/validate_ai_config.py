#!/usr/bin/env python3
"""Validate AI token-efficiency configuration consistency.

This check keeps GHCP and Claude steering/config files aligned so token-efficient
workflows do not silently drift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)


def validate_instruction_docs(errors: list[str]) -> None:
    copilot = _load_text(ROOT / ".github" / "copilot-instructions.md")
    claude = _load_text(ROOT / "CLAUDE.md")

    copilot_l = copilot.lower()
    claude_l = claude.lower()

    _require(
        "prefer" in copilot_l and "whole" in copilot_l and "files" in copilot_l,
        "copilot instructions must explicitly prefer graph tools over whole-file reads",
        errors,
    )
    _require(
        "find_relevant_context(task)" in copilot,
        "copilot instructions must mention find_relevant_context(task)",
        errors,
    )
    _require(
        "prefer" in claude_l and "whole" in claude_l and "files" in claude_l,
        "CLAUDE.md must explicitly prefer graph tools over whole-file reads",
        errors,
    )
    _require(
        "find_relevant_context(task)" in claude,
        "CLAUDE.md must mention find_relevant_context(task)",
        errors,
    )


def _launches_tokengraph(entry: dict) -> bool:
    """Does this server entry actually start the ContextIQ MCP server?

    Two launch styles are legitimate and `tokengraph setup` picks between them
    based on whether the package is installed:

      * console script  -> command "tokengraph",   args [..., "serve"]
      * script fallback -> command <python>,       args [".../tokengraph_all.py", "serve"]

    The previous version hard-required ``command == "python"``, so a
    pip-installed ContextIQ wrote a config that this validator then rejected.
    """
    args = [str(a) for a in entry.get("args", [])]
    if not args or args[-1] != "serve":
        return False
    command = Path(str(entry.get("command", ""))).name.lower()
    command = command[:-4] if command.endswith(".exe") else command
    if command.startswith("tokengraph") or command.startswith("contextiq"):
        return True
    # Script fallback: any python interpreter, pointed at the module.
    if command.startswith("python") or command in {"py", "uv", "uvx"}:
        return any(Path(a).name == "tokengraph_all.py" for a in args)
    return False


# path -> (key the host reads, human name)
MCP_CONFIGS = {
    ".vscode/mcp.json": ("servers", "VS Code / Copilot"),
    ".mcp.json": ("mcpServers", "Claude Code"),
    ".cursor/mcp.json": ("mcpServers", "Cursor"),
}


def validate_mcp_configs(errors: list[str]) -> None:
    """Every MCP config present must declare a tokengraph server that launches.

    Configs are validated when they exist rather than being required, so a repo
    that only wires the hosts it uses still passes — but a config that exists
    and is broken is always caught.
    """
    seen = 0
    for rel, (key, label) in MCP_CONFIGS.items():
        path = ROOT / rel
        if not path.exists():
            continue
        seen += 1
        entry = _load_json(path).get(key, {}).get("tokengraph", {})
        _require(bool(entry), f"{rel} is missing {key}.tokengraph", errors)
        if not entry:
            continue
        _require(entry.get("type", "stdio") == "stdio",
                 f"{label}: tokengraph type must be stdio", errors)
        _require(
            _launches_tokengraph(entry),
            f"{label}: tokengraph args must invoke the server "
            f"(got command={entry.get('command')!r} args={entry.get('args')!r})",
            errors,
        )

    _require(seen > 0,
             "no MCP config found — run: python tokengraph_all.py ide-setup",
             errors)

    # .claude/settings.json has NO mcpServers key; Claude Code reads .mcp.json.
    # An mcpServers block there is silently ignored, so flag it rather than
    # letting a repo believe it is wired.
    claude_settings = ROOT / ".claude" / "settings.json"
    if claude_settings.exists():
        cfg = _load_json(claude_settings)
        _require(
            "mcpServers" not in cfg,
            ".claude/settings.json has an 'mcpServers' key, which Claude Code "
            "ignores — declare MCP servers in .mcp.json instead",
            errors,
        )


def validate_prompt_assets(errors: list[str]) -> None:
    prompts = ROOT / ".prompts"
    expected = {
        "architecture-review.md",
        "bug-fix.md",
        "code-review.md",
        "test-generation.md",
    }

    _require(prompts.exists(), ".prompts directory is missing", errors)
    if prompts.exists():
        actual = {p.name for p in prompts.glob("*.md")}
        for name in sorted(expected - actual):
            errors.append(f"missing prompt template: .prompts/{name}")


def main() -> int:
    errors: list[str] = []

    try:
        validate_instruction_docs(errors)
        validate_mcp_configs(errors)
        validate_prompt_assets(errors)
    except FileNotFoundError as exc:
        errors.append(f"missing required file: {exc.filename}")
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {exc.doc!r}: line {exc.lineno}, col {exc.colno}")

    if errors:
        print("[validate_ai_config] FAILED")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[validate_ai_config] OK")
    print("- instruction docs are aligned")
    print("- MCP configs are consistent")
    print("- prompt templates are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
