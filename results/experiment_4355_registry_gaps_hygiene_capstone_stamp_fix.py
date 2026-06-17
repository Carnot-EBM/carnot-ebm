"""Run Exp 4355 registry/gaps hygiene and capstone stamp fix.

Spec refs: REQ-VERIFY-4355, SCENARIO-VERIFY-4355.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting import (  # noqa: E402,I001
    verifier_registry_gaps_hygiene_capstone_stamp_fix_4355 as exp4355,
)


if __name__ == "__main__":  # pragma: no cover
    exp4355.main()
