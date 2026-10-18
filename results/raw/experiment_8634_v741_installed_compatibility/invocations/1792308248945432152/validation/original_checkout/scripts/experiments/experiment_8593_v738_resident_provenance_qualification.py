"""REQ-REPORT-8593: keep direct execution independent of the caller's import path."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.resident_runner_8593 import main

if __name__ == "__main__":
    raise SystemExit(main())
