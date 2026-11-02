#!/usr/bin/env python3
"""Run Exp 4018 capstone v371 GAP-4 precision/decentralization aggregation."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v371_4018 import write_artifact  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact(REPO_ROOT))
