#!/usr/bin/env python3
"""Entrypoint for Exp 4202 ARC-AGI-3 live solver-vs-floor probe."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4202_arc_live_env_solver_vs_floor import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    main()
