#!/usr/bin/env python3
"""Run Exp 4247 offline reward-weighted verifier-reward harness smoke."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4247_verifier_reward_offline_harness_retire_livelora as exp4247  # noqa: E402,I001

if __name__ == "__main__":
    raise SystemExit(exp4247.main(sys.argv[1:]))
