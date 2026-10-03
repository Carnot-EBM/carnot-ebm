#!/usr/bin/env python3
"""REQ-REPORT-8051: run the persistent learner from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8051_v697_feedback_constrained_learning import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
