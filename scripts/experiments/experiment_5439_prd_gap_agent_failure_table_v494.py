#!/usr/bin/env python3
"""Run Exp5439 PRD gap and failure-taxonomy aggregation."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.prd_gap_agent_failure_table_v494_5439 import write_artifact  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact(REPO_ROOT))
