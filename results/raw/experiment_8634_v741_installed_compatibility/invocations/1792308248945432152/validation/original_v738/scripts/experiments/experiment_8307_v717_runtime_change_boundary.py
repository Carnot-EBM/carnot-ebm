#!/usr/bin/env python3
"""REQ-REPORT-8307: standalone audit entrypoint with private cold replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.runtime_change_execution_8307 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
