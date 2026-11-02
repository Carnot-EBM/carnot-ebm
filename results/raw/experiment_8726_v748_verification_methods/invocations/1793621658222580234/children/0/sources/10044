#!/usr/bin/env python3
"""REQ-REPORT-8664: expose corrected transfer measurement and private cold replay."""

from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
from carnot.reporting.arc_reset_runner_8664 import main  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []

if __name__ == "__main__":
    raise SystemExit(main())
