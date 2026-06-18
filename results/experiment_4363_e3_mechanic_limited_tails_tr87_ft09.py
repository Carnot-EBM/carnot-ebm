#!/usr/bin/env python3
"""Entry point wrapper for Exp 4363."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.experiment_4363_e3_mechanic_limited_tails_tr87_ft09 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
