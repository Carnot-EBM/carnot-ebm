"""REQ-REPORT-9136-V778: inspect complete-cost prerequisites from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.complete_cost_runner_9136 import main

raise SystemExit(main())
