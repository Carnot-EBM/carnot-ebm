#!/usr/bin/env python3
"""Run the V599 independent capstone.

Spec ref: REQ-REPORT-6860.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6860_v599_independent_capstone import main


if __name__ == "__main__":
    raise SystemExit(main())
