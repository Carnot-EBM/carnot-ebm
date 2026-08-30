#!/usr/bin/env python3
"""Run the V589 branch-disposition capstone.

Spec refs: REQ-REPORT-6767 and SCENARIO-REPORT-6767-*.
"""

from __future__ import annotations

from pathlib import Path
import sys
from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for candidate in (REPO_ROOT, PYTHON_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from carnot.experiment_6767_v589_branch_disposition import main as package_main


def main(argv: Sequence[str] | None = None) -> int:
    return package_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
