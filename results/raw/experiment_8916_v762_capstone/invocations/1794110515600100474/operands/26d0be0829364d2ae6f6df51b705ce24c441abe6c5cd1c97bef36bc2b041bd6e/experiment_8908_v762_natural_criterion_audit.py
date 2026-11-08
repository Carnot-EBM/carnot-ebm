"""REQ-REPORT-8908: make the CPU audit callable from outside the checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.natural_criterion_runner_8908 import main

raise SystemExit(main())
