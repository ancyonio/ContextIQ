#!/usr/bin/env python3
"""Rebuild the (gitignored) dist/ folder cleanly and verify the artifacts.

`python -m build` already produces the wheel + sdist. The value this adds is the
*clean + verify* around it, which is where release mistakes actually happen:

  1. wipe stale build artifacts (dist/ accumulates old versions otherwise),
  2. build sdist + wheel for the current pyproject version,
  3. verify every produced filename carries that version, and read the wheel's
     own METADATA to confirm its `Version:` matches — catching a stale/mismatched
     build before it is ever uploaded,
  4. optionally run `twine check` when twine is installed.

Zero required deps beyond the `build` package (stdlib for parsing + verification).

Usage:
    python tools/build_dist.py              # clean, build sdist+wheel, verify
    python tools/build_dist.py --no-clean   # keep existing artifacts
    python tools/build_dist.py --wheel-only
    python tools/build_dist.py --check      # also run `twine check` if available
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
PYPROJECT = ROOT / "pyproject.toml"


def _read_pyproject() -> tuple[str, str]:
    """(name, version) from pyproject.toml — tomllib if present, else regex."""
    text = PYPROJECT.read_text(encoding="utf-8")
    try:
        import tomllib  # py3.11+
        proj = tomllib.loads(text).get("project", {})
        name, version = proj.get("name", ""), proj.get("version", "")
        if name and version:
            return name, version
    except ModuleNotFoundError:
        pass
    import re
    name = re.search(r'(?m)^\s*name\s*=\s*["\']([^"\']+)', text)
    version = re.search(r'(?m)^\s*version\s*=\s*["\']([^"\']+)', text)
    if not (name and version):
        sys.exit("build_dist: could not read name/version from pyproject.toml")
    return name.group(1), version.group(1)


def _clean() -> list[str]:
    removed = []
    if DIST.is_dir():
        for p in sorted(DIST.glob("*")):
            if p.suffix in (".whl", ".gz") or p.name.endswith(".tar.gz"):
                p.unlink()
                removed.append(p.name)
    return removed


def _build(sdist: bool, wheel: bool) -> None:
    args = [sys.executable, "-m", "build"]
    if sdist and not wheel:
        args.append("--sdist")
    elif wheel and not sdist:
        args.append("--wheel")
    print(f"$ {' '.join(args[1:])}")
    proc = subprocess.run(args, cwd=ROOT)
    if proc.returncode != 0:
        sys.exit(f"build_dist: `python -m build` failed (exit {proc.returncode})")


def _wheel_metadata_version(wheel: Path) -> str | None:
    """Read Version: from the wheel's *.dist-info/METADATA (no deps)."""
    try:
        with zipfile.ZipFile(wheel) as zf:
            meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
            for line in zf.read(meta).decode("utf-8", "replace").splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def _verify(name: str, version: str) -> list[str]:
    """Every artifact must carry `version`; the wheel's METADATA must match."""
    problems: list[str] = []
    artifacts = sorted(p for p in DIST.glob("*")
                       if p.suffix == ".whl" or p.name.endswith(".tar.gz"))
    if not artifacts:
        return ["no artifacts were produced in dist/"]
    for a in artifacts:
        if version not in a.name:
            problems.append(f"{a.name} does not contain version {version} "
                            f"(stale artifact?)")
    for wheel in DIST.glob("*.whl"):
        mv = _wheel_metadata_version(wheel)
        if mv and mv != version:
            problems.append(f"{wheel.name} METADATA Version={mv} != {version}")
    return problems


def _twine_check() -> bool:
    try:
        import twine  # noqa: F401
    except ModuleNotFoundError:
        print("  (twine not installed — skipping `twine check`; "
              "`pip install twine` to enable)")
        return True
    proc = subprocess.run([sys.executable, "-m", "twine", "check",
                           str(DIST / "*")], cwd=ROOT)
    return proc.returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild + verify dist/ artifacts.")
    ap.add_argument("--no-clean", action="store_true",
                    help="keep existing artifacts instead of wiping dist/ first")
    ap.add_argument("--wheel-only", action="store_true", help="build only the wheel")
    ap.add_argument("--sdist-only", action="store_true", help="build only the sdist")
    ap.add_argument("--check", action="store_true",
                    help="also run `twine check` (if twine is installed)")
    args = ap.parse_args(argv)
    if args.wheel_only and args.sdist_only:
        sys.exit("build_dist: --wheel-only and --sdist-only are mutually exclusive")

    name, version = _read_pyproject()
    print(f"building {name} {version}")

    if not args.no_clean:
        removed = _clean()
        if removed:
            print(f"  cleaned {len(removed)} stale artifact(s): "
                  f"{', '.join(removed)}")

    DIST.mkdir(exist_ok=True)
    _build(sdist=not args.wheel_only, wheel=not args.sdist_only)

    problems = _verify(name, version)
    print("\nartifacts in dist/:")
    for p in sorted(DIST.glob("*")):
        if p.suffix == ".whl" or p.name.endswith(".tar.gz"):
            print(f"  {p.name}  ({p.stat().st_size:,} bytes)")

    ok = not problems
    if args.check:
        ok = _twine_check() and ok
    if problems:
        print("\nVERIFICATION FAILED:", file=sys.stderr)
        for pr in problems:
            print(f"  - {pr}", file=sys.stderr)
        return 1
    print(f"\nOK - dist/ holds only {name} {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
