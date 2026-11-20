"""REQ-REPORT-9095-V775: run reconciliation from any directory without a model load."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v775_capstone import main

raise SystemExit(main())
