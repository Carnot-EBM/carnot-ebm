"""Entrypoint for Exp 4211 synchronous verifier-as-reward finish.

Spec refs: REQ-CODE-4211, SCENARIO-CODE-4211-BLOCKED-PRECONDITION,
SCENARIO-CODE-4211-SYNC-ACCUMULATE, SCENARIO-CODE-4211-VERDICT-GATES.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4211_verifier_as_reward_finish_synchronous import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
