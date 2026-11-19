"""REQ-REPORT-9165-V780: reconcile original evidence from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v780_capstone import main

raise SystemExit(main())
