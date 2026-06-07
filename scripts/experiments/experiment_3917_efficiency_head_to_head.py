#!/usr/bin/env python3
"""Exp 3917 efficiency head-to-head runner."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:  # pragma: no cover - direct script startup guard.
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.efficiency_head_to_head_3917 import cli_main


def main() -> int:
    return cli_main(["--repo-root", str(REPO_ROOT)])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
