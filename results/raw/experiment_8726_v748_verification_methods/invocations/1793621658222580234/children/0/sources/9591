#!/usr/bin/env python3
"""REQ-REPORT-7980: resolve checkout imports even during an external cold replay."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_7980_v692_evidence_features import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
