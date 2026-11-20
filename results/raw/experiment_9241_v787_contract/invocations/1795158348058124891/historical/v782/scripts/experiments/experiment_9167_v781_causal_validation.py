"""REQ-REPORT-9167-V781: trace CPU validation without any model load."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.causal_validation_9167 import main

raise SystemExit(main())
