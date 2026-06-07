#!/usr/bin/env python3
"""Run Exp 3926 valid efficiency head-to-head versus the competent judge."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.valid_efficiency_head_to_head_3926 import cli_main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
