#!/usr/bin/env python3
"""REQ-REPORT-8235: make the real CLI independent of ambient import paths."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.learning_validation_8235 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
