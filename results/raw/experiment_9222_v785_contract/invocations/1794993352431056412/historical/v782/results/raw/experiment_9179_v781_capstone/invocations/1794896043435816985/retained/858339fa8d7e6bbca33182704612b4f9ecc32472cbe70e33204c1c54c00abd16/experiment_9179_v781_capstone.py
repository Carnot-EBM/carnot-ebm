"""REQ-REPORT-9179-V781: run checked CPU reconciliation from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v781_capstone import main

raise SystemExit(main())
