"""REQ-REPORT-9153-V780: qualify measured costs without loading a model."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.validation_schedule_9153 import main

raise SystemExit(main())
