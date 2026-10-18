#!/usr/bin/env python3
"""Run Exp 3775 capstone v345."""

from __future__ import annotations

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from carnot.reporting.capstone_v345_recovery_3775 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
