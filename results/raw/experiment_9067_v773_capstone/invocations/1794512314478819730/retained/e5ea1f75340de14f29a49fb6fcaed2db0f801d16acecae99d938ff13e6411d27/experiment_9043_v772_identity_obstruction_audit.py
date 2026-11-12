"""REQ-REPORT-9043-V772: run retained metadata diagnosis from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.identity_obstruction_runner_9043 import main

raise SystemExit(main())
