"""REQ-REPORT-8596: direct unbuffered entry point for complete resident costs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.resident_complete_runner_8596 import main

if __name__ == "__main__":
    raise SystemExit(main())
