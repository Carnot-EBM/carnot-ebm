"""Entrypoint for Exp 4198 verifier-reward 3-arm LoRA-RFT launch.

Spec refs: REQ-CODE-4198, SCENARIO-CODE-4198-GATED-LAUNCH,
SCENARIO-CODE-4198-HONEST-DEFERRAL.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_4198_verifier_reward_3arm_rft_launch import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
