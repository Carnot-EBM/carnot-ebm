#!/usr/bin/env python3
"""Run REQ-REPORT-7968 from any working directory without ambient imports."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7968_v691_response_role_targets import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
