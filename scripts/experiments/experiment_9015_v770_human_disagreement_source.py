"""REQ-REPORT-9015-V770: run original human disagreement qualification from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.human_disagreement_runner_9015 import main

raise SystemExit(main())
