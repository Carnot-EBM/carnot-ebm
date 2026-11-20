"""REQ-REPORT-9181-V782: explicit startup works outside checkout without PYTHONPATH."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.validation_diagnostics_9181 import startup

startup()
from carnot.reporting.causal_validation_9181 import main

raise SystemExit(main())
