#!/usr/bin/env python3
"""Run Exp6754 V588 branch disposition.

Spec refs: REQ-REPORT-6754 and SCENARIO-REPORT-6754-NO-POOLED-CLAIM.
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

from carnot.experiment_6754_v588_branch_disposition import main as package_main  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    return package_main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
