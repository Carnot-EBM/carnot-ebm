#!/usr/bin/env python3
"""Run Exp6847 V598 independent capstone.

Spec refs: REQ-RESEARCH-6847 and SCENARIO-RESEARCH-6847-*.
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

from carnot.experiment_6847_v598_independent_capstone import main as package_main


def main(argv: Sequence[str] | None = None) -> int:
    return package_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
