#!/usr/bin/env python3
"""Write a machine-readable provenance manifest for real run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_artifact(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return name, path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hash real inputs and outputs and record the execution environment."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=parse_artifact,
        metavar="NAME=PATH",
        help="Real input or output file to hash; repeat for every artifact.",
    )
    parser.add_argument(
        "--command",
        required=True,
        help="Exact shell command used for the run.",
    )
    args = parser.parse_args()

    if not args.artifact:
        parser.error("at least one --artifact is required")
    names = [name for name, _ in args.artifact]
    if len(names) != len(set(names)):
        parser.error("artifact names must be unique")

    git_status = git_value("status", "--porcelain")
    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_status),
            "status_porcelain": git_status.splitlines(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in args.artifact
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
