#!/usr/bin/env python3
"""Run the tested sealed risk-sensitive learning audit.

Spec refs: REQ-CL-6856 and SCENARIO-CL-6856-*.
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

from carnot.experiment_6856_sealed_risk_sensitive_learning_audit import (
    main as package_main,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate all validation and reduction to the tested package module."""

    return package_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
