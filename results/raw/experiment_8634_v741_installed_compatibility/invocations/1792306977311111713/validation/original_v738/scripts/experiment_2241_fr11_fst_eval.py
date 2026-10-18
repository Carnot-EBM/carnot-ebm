#!/usr/bin/env python3
"""Run Exp 2241 FR-11 Fast-Slow Training evaluation.

Spec: REQ-LEARN-2241, SCENARIO-LEARN-2241.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.fr11_fst_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
