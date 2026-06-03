#!/usr/bin/env python3
"""Run Exp 3764 capstone v344."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from carnot.reporting.capstone_v344_thesis_a_closed_3764 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
