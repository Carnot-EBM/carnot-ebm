#!/usr/bin/env python3
"""Entry point wrapper for Exp 4352."""
# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.experiment_4352_e3_explore_verify_plan_tr87_ft09 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
