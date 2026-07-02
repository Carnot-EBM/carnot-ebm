#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Entrypoint for Exp 5151 ARC Set-Encoder oracle-distinct hardening."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.arc_oracle_distinct_hardening_5151 import main  # noqa: E402


if __name__ == "__main__":
    main()
