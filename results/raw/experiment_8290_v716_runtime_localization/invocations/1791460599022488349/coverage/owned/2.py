#!/usr/bin/env python3
"""REQ-REPORT-8290: direct entrypoint also supports private cold readers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.runtime_localization_8290 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
