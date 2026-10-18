"""REQ-REPORT-8584: direct entry point keeps framed worker output unbuffered."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.resident_complete_runner_8584 import main

if __name__ == "__main__":
    raise SystemExit(main())
