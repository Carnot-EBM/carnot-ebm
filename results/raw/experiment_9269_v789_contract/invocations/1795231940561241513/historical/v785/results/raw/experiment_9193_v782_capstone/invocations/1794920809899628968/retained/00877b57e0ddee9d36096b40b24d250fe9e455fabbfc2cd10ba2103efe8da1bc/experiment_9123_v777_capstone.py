"""REQ-REPORT-9123-V777: reconcile retained evidence from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v777_capstone import main

raise SystemExit(main())
