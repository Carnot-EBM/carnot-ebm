"""Run Exp 4095 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4095, SCENARIO-VERIFY-4095.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.verifier_registry_gaps_hygiene_4095 import main  # noqa: E402


if __name__ == "__main__":
    main()
