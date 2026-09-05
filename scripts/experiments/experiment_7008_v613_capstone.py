#!/usr/bin/env python3
"""Run the V613 independent capstone for REQ-REPORT-7008."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from carnot.experiment_7008_v613_capstone import main


if __name__ == "__main__":
    raise SystemExit(main())
