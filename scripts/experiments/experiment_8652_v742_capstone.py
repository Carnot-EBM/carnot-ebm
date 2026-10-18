"""REQ-REPORT-8652: bind standalone imports to the owned checkout."""

from pathlib import Path
import sys

print("[exp8652] phase=start", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v742_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
