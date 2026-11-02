#!/usr/bin/env python3
"""Run Exp 3891 capstone v359 forward-bet aggregation."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v359_3891 import write_artifact


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact(REPO_ROOT))
