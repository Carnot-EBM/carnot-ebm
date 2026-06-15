#!/usr/bin/env python3
"""Run Exp 4263 verifier-as-reward out-of-band prep or retirement."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot import experiment_4263_verifier_as_reward_out_of_band_or_retire as exp4263  # noqa: E402,I001


if __name__ == "__main__":
    raise SystemExit(exp4263.main(sys.argv[1:]))
