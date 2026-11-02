"""REQ-REPORT-8614: direct execution binds explicit checkout imports."""

from pathlib import Path
import sys

print("[exp8614] phase=start", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v739_capstone import main

if __name__ == "__main__":
    raise SystemExit(main())
