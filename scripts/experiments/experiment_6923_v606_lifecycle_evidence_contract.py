#!/usr/bin/env python3
"""Run the REQ-REPORT-6923 V606 lifecycle evidence contract."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "python"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from carnot.experiment_6923_v606_lifecycle_evidence_contract import main


if __name__ == "__main__":
    raise SystemExit(main())
