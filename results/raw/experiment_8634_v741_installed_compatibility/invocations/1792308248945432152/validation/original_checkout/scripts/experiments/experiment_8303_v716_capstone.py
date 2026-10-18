#!/usr/bin/env python3
"""REQ-REPORT-8303: expose bounded current execution and fresh private replay."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting.v716_capstone import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
