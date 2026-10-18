"""REQ-VERIFY-8625: standalone imports keep fresh replay independent of the working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.hardware_operation_runner_8625 import main

if __name__ == "__main__":
    raise SystemExit(main())
