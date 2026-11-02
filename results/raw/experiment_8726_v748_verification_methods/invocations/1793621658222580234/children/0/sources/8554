#!/usr/bin/env python3
"""Run Exp 4094 precision-calibration SOTA ingestion.

Spec refs: REQ-REPORT-4094, SCENARIO-REPORT-4094.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.sota_ingestion_precision_calibration_4094 import write_outputs  # noqa: E402


def main() -> int:
    receipt = write_outputs(
        note_path=REPO_ROOT
        / "docs"
        / "research-notes"
        / "sota-ingestion-precision-calibration-2026-06-12.md",
        receipt_path=REPO_ROOT
        / "results"
        / "experiment_4094_sota_ingestion_precision_calibration_receipt.json",
        studying_path=REPO_ROOT / "research-studying.md",
    )
    print(receipt["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
