"""REQ-REPORT-9070-V774: audit the pinned release from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.adjudication_methods_runner_9070 import main

raise SystemExit(main())
