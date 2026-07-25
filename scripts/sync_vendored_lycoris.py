#!/usr/bin/env python3
"""Install the vendored (locally modified) LyCORIS over the pip-installed one.

`pip install lycoris-lora` gives the upstream package, which lacks the local
extension algos (glokr / tglokr / bokr / bora / gsokr / glora_boft) and the
Anima-specific fixes. Run this after creating or reinstalling the venv:

    python scripts/sync_vendored_lycoris.py

Use --check to only report drift (exit code 1 when they differ), which is what
tests/test_vendored_lycoris.py does.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "vendor" / "lycoris"
SKIP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints"}
# Repo-only docs about the vendoring itself; they do not belong in site-packages.
SKIP_FILE_NAMES = {"VENDOR.md"}


def installed_lycoris_dir() -> Path | None:
    """Locate the lycoris package inside the active interpreter's site-packages."""
    try:
        import lycoris  # noqa: PLC0415 - optional at import time
    except ImportError:
        return None
    if not lycoris.__file__:
        return None
    return Path(lycoris.__file__).resolve().parent


def vendored_files() -> list[Path]:
    return sorted(
        path
        for path in VENDORED.rglob("*")
        if path.is_file()
        and not SKIP_DIR_NAMES & set(path.parts)
        and path.name not in SKIP_FILE_NAMES
    )


def compare(target: Path) -> list[str]:
    """Return the relative paths that are missing or different in ``target``."""
    drifted: list[str] = []
    for src in vendored_files():
        rel = src.relative_to(VENDORED)
        dst = target / rel
        if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
            drifted.append(rel.as_posix())
    return drifted


def install(target: Path) -> int:
    copied = 0
    for src in vendored_files():
        rel = src.relative_to(VENDORED)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report drift, copy nothing")
    args = parser.parse_args()

    if not VENDORED.is_dir():
        print(f"vendored lycoris not found: {VENDORED}", file=sys.stderr)
        return 2

    target = installed_lycoris_dir()
    if target is None:
        print("lycoris is not importable — install requirements first.", file=sys.stderr)
        return 2

    drifted = compare(target)
    if args.check:
        if drifted:
            print(f"{len(drifted)} file(s) differ from vendor/lycoris:")
            for rel in drifted[:20]:
                print(f"  {rel}")
            print("run: python scripts/sync_vendored_lycoris.py")
            return 1
        print(f"vendor/lycoris matches {target}")
        return 0

    if not drifted:
        print(f"already in sync: {target}")
        return 0

    copied = install(target)
    print(f"copied {copied} file(s) to {target} ({len(drifted)} were stale)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
