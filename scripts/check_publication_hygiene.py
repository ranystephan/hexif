#!/usr/bin/env python3
"""Fail when publication-only material leaks into the source tree."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 5 * 1024 * 1024
FORBIDDEN_NAMES = {
    "CLAUDE.md",
    "TODO_PUBLIC_RELEASE.md",
    ".DS_Store",
}
FORBIDDEN_PARTS = {
    ".claude",
    "archive",
    "benchmarks",
    "checkpoints",
    "logs",
    "notebooks",
    "outputs",
    "paper",
    "presentations",
    "runs",
    "trash",
    "wandb",
}
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".hdf5",
    ".ipynb",
    ".npy",
    ".npz",
    ".pdf",
    ".pt",
    ".pth",
    ".safetensors",
    ".svs",
    ".tif",
    ".tiff",
}
TEXT_SUFFIXES = {
    ".cff",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
FORBIDDEN_TEXT = {
    "machine-specific path": re.compile(
        r"(?:/scratch/(?:users/)?|/home/users/|/Users/)[A-Za-z0-9_.-]+/"
    ),
    "AI coauthor trailer": re.compile(r"co-authored-by:.*(?:claude|chatgpt|openai)", re.IGNORECASE),
    "internal assistant file": re.compile(r"\bCLAUDE\.md\b"),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / name.decode() for name in result.stdout.split(b"\0") if name]


def main() -> int:
    errors: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        if not path.exists():
            continue
        if path.name in FORBIDDEN_NAMES:
            errors.append(f"{relative}: forbidden publication artifact")
        if any(part in FORBIDDEN_PARTS for part in relative.parts[:-1]):
            errors.append(f"{relative}: forbidden directory")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"{relative}: forbidden binary or notebook type")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            errors.append(f"{relative}: {size} bytes exceeds 5 MiB")
        if relative == Path("scripts/check_publication_hygiene.py"):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or ".min." in path.name:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"{relative}: expected UTF-8 text")
            continue
        for label, pattern in FORBIDDEN_TEXT.items():
            if pattern.search(content):
                errors.append(f"{relative}: contains {label}")

    if errors:
        print("Publication hygiene check failed:", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Publication hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
