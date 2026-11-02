"""REQ-VERIFY-8613: standalone unbuffered entry for bounded CPU-only evidence mapping."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.hardware_operation_runner_8613 import main

if __name__ == "__main__":
    raise SystemExit(main())
