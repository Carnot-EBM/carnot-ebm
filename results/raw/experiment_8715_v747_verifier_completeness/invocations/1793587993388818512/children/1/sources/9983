"""REQ-VERIFY-8600: standalone CPU entry preserves bounded unbuffered execution."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.resident_hardware_runner_8600 import main

if __name__ == "__main__":
    raise SystemExit(main())
