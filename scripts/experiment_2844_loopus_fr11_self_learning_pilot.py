#!/usr/bin/env python3
"""Run Exp 2844 LoopUS-style FR-11 self-learning pilot."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

os.environ.setdefault("CARNOT_REPO_ROOT", str(REPO_ROOT))

from carnot.eval.loopus_fr11_self_learning_pilot import ExperimentConfig, run_experiment


def main() -> int:
    artifact = run_experiment(ExperimentConfig(repo_root=REPO_ROOT))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
