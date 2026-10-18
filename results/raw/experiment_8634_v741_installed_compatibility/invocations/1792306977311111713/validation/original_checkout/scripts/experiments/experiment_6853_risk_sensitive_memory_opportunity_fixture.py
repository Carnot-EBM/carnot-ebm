#!/usr/bin/env python3
"""Run the tested risk-sensitive memory fixture builder.

Spec refs: REQ-CL-6853 and SCENARIO-CL-6853-*.
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

from carnot.experiment_6853_risk_sensitive_memory_opportunity_fixture import main as package_main


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate all construction and validation to the tested package."""

    return package_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
