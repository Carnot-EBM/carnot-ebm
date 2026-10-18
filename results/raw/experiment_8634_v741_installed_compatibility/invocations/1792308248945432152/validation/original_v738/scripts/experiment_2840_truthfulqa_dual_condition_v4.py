#!/usr/bin/env python3
"""Command entrypoint for Exp 2840 TruthfulQA dual-condition v4."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.truthfulqa_dual_condition_v4 import main


if __name__ == "__main__":
    raise SystemExit(main())
