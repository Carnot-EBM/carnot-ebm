"""Run Exp 4421 registry/gaps hygiene and GAP-4 guard.

Spec refs: REQ-VERIFY-4421, SCENARIO-VERIFY-4421.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4421_registry_gaps_hygiene_gap4_guard as exp4421  # noqa: E402,I001


if __name__ == "__main__":  # pragma: no cover
    exp4421.main()
