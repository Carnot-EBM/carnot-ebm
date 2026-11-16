"""REQ-REPORT-9139-V779: standalone schedule qualification without a model load."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.validation_schedule_runner_9139 import main

raise SystemExit(main())
