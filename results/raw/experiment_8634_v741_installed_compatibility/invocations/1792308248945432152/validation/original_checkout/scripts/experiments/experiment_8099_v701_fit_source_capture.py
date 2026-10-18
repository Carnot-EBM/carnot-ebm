#!/usr/bin/env python3
"""REQ-REPORT-8099-PUBLICATION: resolve imports during external-CWD replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8099_v701_fit_source_capture import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
