#!/usr/bin/env python3
"""REQ-REPORT-8237: resolve imports for fresh replay outside the checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.margin_energy_runner_8237 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
