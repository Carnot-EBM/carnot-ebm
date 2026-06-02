#!/usr/bin/env python3
"""Run Exp 3727 matched-compute FLOP accounting harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.phase3.matched_compute_eval_harness import run_experiment  # noqa: E402


def main() -> int:
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
