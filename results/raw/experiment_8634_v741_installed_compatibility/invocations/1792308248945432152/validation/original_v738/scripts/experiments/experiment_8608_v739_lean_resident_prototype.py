"""REQ-REPORT-8608: direct execution uses explicit checkout paths."""

from pathlib import Path
import sys

print("[exp8608] phase=start", flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.lean_runner_8608 import main

if __name__ == "__main__":
    raise SystemExit(main())
