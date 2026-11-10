"""Fail when a public release contains common private-repository artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}
FORBIDDEN_ROOTS = {"results", "run", "docs_private", "private"}
TEXT_SUFFIXES = {".cfg", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def patterns() -> dict[str, re.Pattern[str]]:
    # Build administrative patterns from fragments so the checker does not
    # report its own source as a finding.
    return {
        "secret-like API key": re.compile("\\b" + "sk-" + r"[A-Za-z0-9_-]{16,}\\b"),
        "private shared path": re.compile("/" + r"(?:shared|aistor|mnt/cloudstorfs)" + "/"),
        "internal registry": re.compile("registry" + r"\\.cluster\\.local"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--max-file-mb", type=float, default=5.0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    findings: list[str] = []

    for name in sorted(FORBIDDEN_ROOTS):
        if (root / name).exists():
            findings.append(f"forbidden top-level path: {name}/")

    max_bytes = int(args.max_file_mb * 1024 * 1024)
    checks = patterns()
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        relative = path.relative_to(root)
        if path.stat().st_size > max_bytes:
            findings.append(f"large file: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in checks.items():
            if pattern.search(text):
                findings.append(f"{label}: {relative}")

    if findings:
        print("Public release audit failed:")
        for finding in sorted(set(findings)):
            print(f"- {finding}")
        return 1
    print("Public release audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
