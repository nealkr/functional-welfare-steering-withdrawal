#!/usr/bin/env python3
"""Extract compact terminal/custody metadata from `hf jobs inspect` JSON."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import re
import sys
from typing import Any


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def byte_summary(value: bytes) -> dict[str, Any]:
    return {
        "bytes": len(value),
        "sha256": sha256(value),
        "crlf_count": value.count(b"\r\n"),
        "lf_count": value.count(b"\n"),
        "ends_newline": value.endswith((b"\n", b"\r")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-runner", type=pathlib.Path, required=True)
    args = parser.parse_args()
    value = json.load(sys.stdin)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("expected exactly one inspected job")
    job = value[0]
    command = job["command"]
    if not isinstance(command, list) or len(command) < 3:
        raise ValueError("unexpected job command")
    match = re.search(r'echo "([A-Za-z0-9+/=]+)"', command[2])
    if match is None:
        raise ValueError("inline base64 payload not found")
    submitted = base64.b64decode(match.group(1), validate=True)
    local = args.local_runner.read_bytes()
    normalize = lambda data: data.replace(b"\r\n", b"\n")
    compact = {
        key: value
        for key, value in job.items()
        if key not in ("command", "owner", "created_by")
    }
    compact.update(
        {
            "command_argv_prefix": command[:2],
            "submitted_payload": byte_summary(submitted),
            "local_lf_runner": byte_summary(local),
            "normalized_equal": normalize(submitted) == normalize(local),
            "normalized_sha256": {
                "submitted": sha256(normalize(submitted)),
                "local": sha256(normalize(local)),
            },
        }
    )
    print(json.dumps(compact, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
