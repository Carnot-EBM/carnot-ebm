#!/usr/bin/env python3
"""REQ-REPORT-7984: run cached CPU ablation without an ambient PYTHONPATH."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7984_v692_evidence_ablation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
