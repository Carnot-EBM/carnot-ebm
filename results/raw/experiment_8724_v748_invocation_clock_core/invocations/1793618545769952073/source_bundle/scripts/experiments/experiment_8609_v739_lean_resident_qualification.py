"""REQ-REPORT-8609: run the private qualification and exact-byte publication."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.lean_runner_8609 import main


if __name__ == "__main__":
    raise SystemExit(main())
