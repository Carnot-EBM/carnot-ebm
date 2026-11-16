"""REQ-REPORT-9150-V779: execute checked cost accounting from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting import complete_cost_9150
from carnot.reporting.complete_cost_runner_9136 import main

raise SystemExit(main(producer=complete_cost_9150))
