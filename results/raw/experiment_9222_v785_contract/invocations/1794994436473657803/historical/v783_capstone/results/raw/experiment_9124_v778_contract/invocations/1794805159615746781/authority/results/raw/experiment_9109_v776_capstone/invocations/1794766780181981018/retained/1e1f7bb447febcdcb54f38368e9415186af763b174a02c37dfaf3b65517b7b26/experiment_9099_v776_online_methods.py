"""REQ-REPORT-9099-V776: qualify causal human methods from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.online_methods_runner_9099 import main

raise SystemExit(main())
