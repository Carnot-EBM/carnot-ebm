#!/usr/bin/env python3
"""REQ-REPORT-8359: run the bounded capstone from any working directory."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("TMPDIR", "/tmp/carnot8359-disk")

from carnot.reporting.v720_capstone import main  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []

if __name__ == "__main__":
    raise SystemExit(main())
