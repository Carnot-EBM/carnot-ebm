#!/usr/bin/env python3
"""REQ-REPORT-8098: direct CLI works independently of the caller's directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8098_v701_development_methods import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
