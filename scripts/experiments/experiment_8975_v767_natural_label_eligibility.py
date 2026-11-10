"""REQ-REPORT-8975: run the CPU access audit from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.natural_label_runner_8975 import main

raise SystemExit(main())
