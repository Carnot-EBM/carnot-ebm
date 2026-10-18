#!/usr/bin/env python3
"""REQ-REPORT-8010: keep cold replay imports stable from an external directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8010_v694_source_intervention_protocol import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
