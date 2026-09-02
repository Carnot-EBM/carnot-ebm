#!/usr/bin/env python3
"""Run the append-only V602 evidence contract.

Spec ref: REQ-REPORT-6874.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6874_v602_evidence_substrate_manifest_contract import main


if __name__ == "__main__":  # pragma: no cover - exercised by the required command
    raise SystemExit(main())
