"""REQ-REPORT-9084-V775: freeze exposed methods from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.human_methods_runner_9084 import main

raise SystemExit(main())
