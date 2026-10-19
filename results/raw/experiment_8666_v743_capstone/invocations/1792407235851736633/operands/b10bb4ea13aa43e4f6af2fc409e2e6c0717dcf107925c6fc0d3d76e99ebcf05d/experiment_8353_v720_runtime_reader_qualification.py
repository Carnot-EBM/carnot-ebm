#!/usr/bin/env python3
"""REQ-REPORT-8353: standalone reader qualification without model loading."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.runtime_reader_execution_8353 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
