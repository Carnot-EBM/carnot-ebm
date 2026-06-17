#!/usr/bin/env python3
"""Run Exp 4319 off-ARC execution-verifier transfer accumulation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.verify.off_arc_execution_verifier_transfer_accumulate import run


if __name__ == "__main__":
    run()
