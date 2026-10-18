"""REQ-REPORT-8553: run the bounded release audit from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.hallmark_release_runner_8553 import main

if __name__ == "__main__":
    raise SystemExit(main())
