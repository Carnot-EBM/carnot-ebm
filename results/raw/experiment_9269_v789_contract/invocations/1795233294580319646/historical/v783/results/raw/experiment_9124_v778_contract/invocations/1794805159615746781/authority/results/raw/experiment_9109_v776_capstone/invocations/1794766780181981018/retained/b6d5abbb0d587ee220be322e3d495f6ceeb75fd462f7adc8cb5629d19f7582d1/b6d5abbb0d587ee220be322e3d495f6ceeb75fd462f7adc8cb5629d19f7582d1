"""REQ-REPORT-9094-V775: run the CPU susceptibility diagnostic from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.susceptibility_runner_9094 import main

raise SystemExit(main())
