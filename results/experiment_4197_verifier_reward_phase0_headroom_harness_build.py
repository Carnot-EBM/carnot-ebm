"""Entrypoint for Exp 4197 verifier-reward operating point build.

Spec refs: REQ-CODE-4197, SCENARIO-CODE-4197-PHASE0,
SCENARIO-CODE-4197-HARNESS.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4197_verifier_reward_phase0_headroom import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
