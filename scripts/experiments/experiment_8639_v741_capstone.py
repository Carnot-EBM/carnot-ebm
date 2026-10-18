"""REQ-REPORT-8639: bind standalone imports to the owned checkout."""

from pathlib import Path
import sys

print("[exp8639] phase=start", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v741_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
