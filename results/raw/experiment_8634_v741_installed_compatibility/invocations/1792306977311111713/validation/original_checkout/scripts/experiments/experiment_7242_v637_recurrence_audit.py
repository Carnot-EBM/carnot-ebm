#!/usr/bin/env python3
"""Run the V637 cold recurrence-memory causality and safety audit."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7242_v637_recurrence_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
