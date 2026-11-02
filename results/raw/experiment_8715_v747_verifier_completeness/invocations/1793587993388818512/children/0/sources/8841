#!/usr/bin/env python3
"""Run the append-only V603 executable-manifest contract.

Spec ref: REQ-REPORT-6885.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6885_v603_executable_manifest_branch_contract import main


if __name__ == "__main__":  # pragma: no cover - exercised by the required command
    raise SystemExit(main())
