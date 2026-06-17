"""Run Exp 4321 verifier registry/gaps hygiene from the results entrypoint.

Spec refs: REQ-VERIFY-4321, SCENARIO-VERIFY-4321.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting import verifier_registry_gaps_hygiene_4321 as exp4321  # noqa: E402,I001


if __name__ == "__main__":  # pragma: no cover
    exp4321.main()
