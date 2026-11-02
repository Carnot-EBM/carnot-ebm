"""REQ-VERIFY-8589: standalone entry preserves unbuffered bounded CPU evidence work."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.resident_hardware_runner_8589 import main

if __name__ == "__main__":
    raise SystemExit(main())
