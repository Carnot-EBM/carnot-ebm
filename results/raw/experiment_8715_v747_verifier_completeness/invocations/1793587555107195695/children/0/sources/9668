#!/usr/bin/env python3
"""REQ-REPORT-8118: resolve imports for script-path execution outside checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8118_v702_fresh_acquisition_cost import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
