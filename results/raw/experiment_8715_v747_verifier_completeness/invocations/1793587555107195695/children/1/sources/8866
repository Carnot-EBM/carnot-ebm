#!/usr/bin/env python3
"""Run the V608 executable contract and bounded-scope preflight.

Spec ref: REQ-REPORT-6942.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (REPO_ROOT, PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from carnot.experiment_6942_v608_contract_preflight import main


if __name__ == "__main__":  # pragma: no cover - exercised by the required command
    raise SystemExit(main())
