#!/usr/bin/env python3
"""REQ-REPORT-8241: explicit imports support replay outside the checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.delayed_benefit_audit_8241 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
