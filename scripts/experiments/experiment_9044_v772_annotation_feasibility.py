"""REQ-REPORT-9044-V772: run ID-only annotation feasibility from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.annotation_feasibility_runner_9044 import main

raise SystemExit(main())
