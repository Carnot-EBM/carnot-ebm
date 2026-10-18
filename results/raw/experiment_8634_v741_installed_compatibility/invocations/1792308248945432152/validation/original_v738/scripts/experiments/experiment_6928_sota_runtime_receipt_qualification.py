#!/usr/bin/env python3
"""Run the V607 SOTA runtime receipt qualification.

Spec ref: REQ-REPORT-6928.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6928_sota_runtime_receipt_qualification import main


if __name__ == "__main__":
    raise SystemExit(main())
