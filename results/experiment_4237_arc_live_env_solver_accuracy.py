#!/usr/bin/env python3
"""Entrypoint for Exp 4237 ARC-AGI-3 live margin-triggered solver accuracy probe."""
# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_4237_arc_live_env_solver_accuracy import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    main()
