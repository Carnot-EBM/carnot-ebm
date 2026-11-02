#!/usr/bin/env python3
"""Run Exp 2246 CASAL vs AdamFLIP constraint-violation benchmark.

Spec: REQ-SAMPLE-2246, SCENARIO-SAMPLE-2246.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.casal_vs_adamflip import main


if __name__ == "__main__":
    raise SystemExit(main())
