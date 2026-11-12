"""REQ-REPORT-9029-V771: use the retained metadata audit from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.source_identity_runner_9029 import main

raise SystemExit(main())
