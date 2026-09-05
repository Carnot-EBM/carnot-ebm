#!/usr/bin/env python3
"""Run the task-linked compute receipt contract from REQ-REPORT-7017."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7017_task_linked_compute_receipts import main


if __name__ == "__main__":
    raise SystemExit(main())
