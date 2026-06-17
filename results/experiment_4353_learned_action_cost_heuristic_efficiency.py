#!/usr/bin/env python3
"""Entrypoint for Exp 4353 learned ARC action-cost heuristic efficiency."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4353_learned_action_cost_heuristic_efficiency import main  # noqa: E402


if __name__ == "__main__":
    main()
