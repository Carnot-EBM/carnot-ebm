"""REQ-REPORT-9111-V777: the script path works without an ambient PYTHONPATH."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.validation_diagnostics_9181 import startup

startup()
from carnot.reporting.forward_evidence_runner_9111 import main

raise SystemExit(main())
