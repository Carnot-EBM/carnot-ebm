#!/usr/bin/env python3
"""Thin CLI for the V662 independent decision-learning audit."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for location in (ROOT / "python", ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from carnot.experiment_7579_v662_decision_learning_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
