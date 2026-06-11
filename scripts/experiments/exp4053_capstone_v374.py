#!/usr/bin/env python3
"""Run Exp 4053 capstone v374 decision-grade aggregation."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_v374_4053 import write_artifact  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact(REPO_ROOT))
