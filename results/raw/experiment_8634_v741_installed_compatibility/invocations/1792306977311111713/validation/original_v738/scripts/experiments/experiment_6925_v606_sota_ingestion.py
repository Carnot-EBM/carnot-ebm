#!/usr/bin/env python3
"""Run the bounded V606 execution-time SOTA ingestion.

Spec ref: REQ-REPORT-6925.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6925_v606_sota_ingestion import main


if __name__ == "__main__":
    raise SystemExit(main())
