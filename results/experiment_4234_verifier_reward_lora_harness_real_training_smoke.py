#!/usr/bin/env python3
"""Run Exp 4234 verifier-reward LoRA real-training smoke."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4234_verifier_reward_lora_harness_real_training_smoke as exp4234  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(exp4234.main(sys.argv[1:]))
