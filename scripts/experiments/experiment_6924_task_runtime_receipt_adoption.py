#!/usr/bin/env python3
"""Run the V606 task-owned runtime receipt adoption fixture.

Spec ref: REQ-REPORT-6924.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6924_task_runtime_receipt_adoption import main


if __name__ == "__main__":
    raise SystemExit(main())
