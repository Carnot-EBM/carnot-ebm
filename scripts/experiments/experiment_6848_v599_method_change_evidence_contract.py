#!/usr/bin/env python3
"""Run Exp6848 V599 method-change evidence contract.

Spec refs: REQ-REPORT-6848 and SCENARIO-REPORT-6848-*.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for root in (REPO_ROOT, PYTHON_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from carnot.experiment_6848_v599_method_change_evidence_contract import main as package_main


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the tested package implementation."""

    return package_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
