"""Run Exp 4051 verifier registry and gaps hygiene.

Spec refs: REQ-VERIFY-4051, SCENARIO-VERIFY-4051.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.verifier_registry_and_gaps_hygiene_4051 import main  # noqa: E402


if __name__ == "__main__":
    main()
