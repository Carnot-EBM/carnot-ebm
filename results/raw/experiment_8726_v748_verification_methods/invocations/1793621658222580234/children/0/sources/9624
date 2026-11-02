#!/usr/bin/env python3
"""REQ-REPORT-8021: run the sealed decision reducer from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8021_v695_typed_decision_test import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
