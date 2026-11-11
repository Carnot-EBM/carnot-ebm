"""REQ-REPORT-9001: run original human-source authority qualification from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.human_source_runner_9001 import main

raise SystemExit(main())
