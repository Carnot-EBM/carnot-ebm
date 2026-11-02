"""REQ-REPORT-8585: allow standalone CPU audit and replay from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.hallutruth_release_runner_8585 import main

if __name__ == "__main__":
    raise SystemExit(main())
