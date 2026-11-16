"""REQ-REPORT-9137-V778: reconcile retained evidence from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.v778_capstone import main

raise SystemExit(main())
