#!/usr/bin/env python3
"""Entrypoint for Exp 4370 LLM-generated ARC action-cost heuristics."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4370_llm_generated_action_cost_heuristics import main  # noqa: E402


if __name__ == "__main__":
    main()
