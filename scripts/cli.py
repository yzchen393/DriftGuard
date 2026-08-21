from __future__ import annotations

import sys
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def require_file(path: str | Path, message: str) -> Path:
    target = Path(path)
    if not target.is_file():
        fail(message)
    return target
