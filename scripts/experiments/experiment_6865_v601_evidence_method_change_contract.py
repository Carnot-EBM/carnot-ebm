#!/usr/bin/env python3
"""Run the V601 evidence and method-change contract.

Spec ref: REQ-REPORT-6865.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6865_v601_evidence_method_change_contract import main


if __name__ == "__main__":
    raise SystemExit(main())
