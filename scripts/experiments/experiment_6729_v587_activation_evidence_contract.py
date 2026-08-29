#!/usr/bin/env python3
"""Run Exp6729 V587 activation evidence contract.

Spec refs: REQ-REPORT-6729, SCENARIO-REPORT-6729-ATOMIC.
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

from carnot.experiment_6729_v587_activation_evidence_contract import main as package_main  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    return package_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
