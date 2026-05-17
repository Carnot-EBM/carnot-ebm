#!/usr/bin/env python3
"""Run Exp 2242 ActFocus + FST evaluation.

Spec: REQ-LEARN-2242, SCENARIO-LEARN-2242.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.actfocus_fst_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
