"""REQ-REPORT-9093-V775: account for unavailable service work from any directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.service_cost_runner_9093 import main

raise SystemExit(main())
