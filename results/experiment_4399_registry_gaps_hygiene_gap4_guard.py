"""Run Exp 4399 registry/gaps hygiene, GAP-4 guard, and stamp durability.

Spec refs: REQ-VERIFY-4399, SCENARIO-VERIFY-4399.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting import verifier_registry_gaps_hygiene_gap4_guard_4399 as exp4399  # noqa: E402,I001


if __name__ == "__main__":  # pragma: no cover
    exp4399.main()
