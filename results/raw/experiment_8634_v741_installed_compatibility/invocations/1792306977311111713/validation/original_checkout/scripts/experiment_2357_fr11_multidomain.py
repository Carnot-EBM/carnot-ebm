#!/usr/bin/env python3
"""Run Exp 2357 FR-11 multidomain FST retention evaluation.

Spec: REQ-LEARN-2357, SCENARIO-LEARN-2357.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.fr11_multidomain_fst_retention import main


if __name__ == "__main__":
    raise SystemExit(main())
