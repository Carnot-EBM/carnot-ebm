#!/usr/bin/env python3
"""REQ-REPORT-8209: resolve checkout imports for actual private CLI children."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.restricted_sealed_evaluation_8209 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
