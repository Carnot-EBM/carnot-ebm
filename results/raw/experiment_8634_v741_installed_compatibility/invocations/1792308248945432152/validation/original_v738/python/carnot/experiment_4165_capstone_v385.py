"""Run Exp 4165 capstone v385 aggregation.

Spec refs: REQ-CAPSTONE-4165, SCENARIO-CAPSTONE-4165.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.capstone_v385_4165 import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
