#!/usr/bin/env python3
"""Run Exp 2251 capstone end-to-end interop evaluation.

Spec: REQ-CAPSTONE-2251, SCENARIO-CAPSTONE-2251.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.capstone_e2e_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
