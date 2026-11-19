"""REQ-REPORT-9125-V778: standalone bounded qualification without model loading."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.receipt_qualification_9125 import main

raise SystemExit(main())
