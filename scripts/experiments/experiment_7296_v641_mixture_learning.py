#!/usr/bin/env python3
"""Run the reusable prospective fixed-share learning experiment."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from carnot.experiment_7296_v641_mixture_learning import main


if __name__ == "__main__":
    raise SystemExit(main())
