#!/usr/bin/env python3
"""Run the sealed V627 relational fixture build for REQ-HARNESS-7138."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7138_v627_relational_fixture import main


if __name__ == "__main__":
    raise SystemExit(main())
