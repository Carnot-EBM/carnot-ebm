"""REQ-REPORT-8597: permit independent script-path audit and replay from private scratch."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.medexpert_evidence_runner_8597 import main

if __name__ == "__main__":
    raise SystemExit(main())
