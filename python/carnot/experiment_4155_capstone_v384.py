"""Run Exp 4155 capstone v384 aggregation.

Spec refs: REQ-CAPSTONE-4155, SCENARIO-CAPSTONE-4155.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.capstone_v384_4155 import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
