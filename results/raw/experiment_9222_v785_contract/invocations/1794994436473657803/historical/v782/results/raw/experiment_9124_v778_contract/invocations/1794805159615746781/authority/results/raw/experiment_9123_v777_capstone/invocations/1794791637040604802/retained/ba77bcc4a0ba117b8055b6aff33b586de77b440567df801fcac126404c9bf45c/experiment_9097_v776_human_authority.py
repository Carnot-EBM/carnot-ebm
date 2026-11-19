"""REQ-REPORT-9097-V776: authenticate exposed human authority from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.human_authority_runner_9097 import main

raise SystemExit(main())
