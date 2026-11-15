"""REQ-REPORT-9109-V776: reconcile from any directory without loading a model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v776_capstone import main

raise SystemExit(main())
