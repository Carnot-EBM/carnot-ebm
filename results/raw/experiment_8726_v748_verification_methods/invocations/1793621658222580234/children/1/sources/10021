"""REQ-REPORT-8626: bind direct invocation to the owned checkout before import."""

from pathlib import Path
import sys

print("[exp8626] phase=start", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v740_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
