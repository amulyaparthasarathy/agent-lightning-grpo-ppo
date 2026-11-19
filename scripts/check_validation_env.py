#!/usr/bin/env python3
"""Validate local .env without printing values or making any network request."""
from __future__ import annotations
import argparse
from pathlib import Path
import stat
import subprocess

KEYS = {"HF_TOKEN", "WANDB_API_KEY"}


def check(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Credential file must be an existing regular file, not a symlink")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("Credential file is readable by others; run chmod 600 .env")
    root = Path(__file__).resolve().parents[1]
    relative = path.resolve().relative_to(root)
    tracked = subprocess.run(["git", "ls-files", "--", str(relative)], cwd=root,
                             capture_output=True, text=True, check=True)
    ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(relative)], cwd=root)
    if tracked.stdout.strip() or ignored.returncode != 0:
        raise ValueError("Credential file must be Git-ignored and untracked")
    found = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Expected KEY=value on line {number}")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in KEYS or key in found:
            raise ValueError(f"Unexpected or duplicate key on line {number}")
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unmatched quotes on line {number}")
            value = value[1:-1]
        if not value or "replace_with" in value or any(c.isspace() for c in value):
            raise ValueError(f"Missing, placeholder, or whitespace-containing value on line {number}")
        if any(c in value for c in "${}`\\\"'"):
            raise ValueError(f"Interpolation/escape syntax is not allowed on line {number}")
        found.add(key)
    if found != KEYS:
        raise ValueError("Both HF_TOKEN and WANDB_API_KEY are required")
    print("PASS: two non-placeholder keys; file is private, untracked and ignored. Values not printed.")
    print("This checks local file hygiene, not token validity. The services test checks authentication.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path(".env"))
    args = parser.parse_args()
    try:
        check(args.path)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        # Validation errors contain no credential values. Do not print raw subprocess output.
        parser.exit(1, f"FAIL: {exc if isinstance(exc, ValueError) else type(exc).__name__}\n")
