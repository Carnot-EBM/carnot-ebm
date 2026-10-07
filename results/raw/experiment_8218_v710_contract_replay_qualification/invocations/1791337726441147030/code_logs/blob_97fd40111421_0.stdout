#!/usr/bin/env python3
"""REQ-REPORT-8210: direct CLI children resolve exact checkout imports."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.restricted_decision_audit_8210 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
