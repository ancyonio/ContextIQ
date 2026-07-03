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


def validate_mcp_configs(errors: list[str]) -> None:
    vscode_cfg = _load_json(ROOT / ".vscode" / "mcp.json")
    claude_cfg = _load_json(ROOT / ".mcp.json")

    vs_tg = vscode_cfg.get("servers", {}).get("tokengraph", {})
    cl_tg = claude_cfg.get("mcpServers", {}).get("tokengraph", {})

    _require(bool(vs_tg), ".vscode/mcp.json is missing servers.tokengraph", errors)
    _require(bool(cl_tg), ".mcp.json is missing mcpServers.tokengraph", errors)

    def _ends_with_tokengraph_serve(args: list[object]) -> bool:
        if len(args) < 2:
            return False
        script = str(args[-2])
        return Path(script).name == "tokengraph_all.py" and str(args[-1]) == "serve"

    if vs_tg:
        _require(vs_tg.get("type") == "stdio", "VS Code tokengraph type must be stdio", errors)
        _require(vs_tg.get("command") == "python", "VS Code tokengraph command must be python", errors)
        _require(
            _ends_with_tokengraph_serve(vs_tg.get("args", [])),
            "VS Code tokengraph args must end with [tokengraph_all.py, serve]",
            errors,
        )
        _require(
            vs_tg.get("env", {}).get("TOKENGRAPH_ROOT") == "${workspaceFolder}",
            "VS Code TOKENGRAPH_ROOT must be ${workspaceFolder}",
            errors,
        )

    if cl_tg:
        _require(cl_tg.get("type") == "stdio", "Claude tokengraph type must be stdio", errors)
        _require(cl_tg.get("command") == "python", "Claude tokengraph command must be python", errors)
        _require(
            _ends_with_tokengraph_serve(cl_tg.get("args", [])),
            "Claude tokengraph args must end with [tokengraph_all.py, serve]",
            errors,
        )
        _require(
            cl_tg.get("env", {}).get("TOKENGRAPH_ROOT") == ".",
            "Claude TOKENGRAPH_ROOT must be .",
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
