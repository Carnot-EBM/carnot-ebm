#!/usr/bin/env python3
"""REQ-REPORT-8249: let private cold replay execute outside the checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.evidence_view_execution_8249 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
