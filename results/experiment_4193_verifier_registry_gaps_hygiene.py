"""Run Exp 4193 verifier registry/gaps hygiene from the results entrypoint.

Spec refs: REQ-VERIFY-4193, SCENARIO-VERIFY-4193.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.reporting import verifier_registry_gaps_hygiene_4193 as exp4193  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    exp4193.main()
