"""REQ-REPORT-8572: direct unbuffered entry point for private resident qualification."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.resident_runner_8572 import main

if __name__ == "__main__":
    raise SystemExit(main())
