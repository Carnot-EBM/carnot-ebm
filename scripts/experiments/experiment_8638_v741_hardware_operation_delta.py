"""REQ-VERIFY-8638: standalone imports allow replay outside the checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.hardware_delta_runner_8638 import main

if __name__ == "__main__":
    raise SystemExit(main())
