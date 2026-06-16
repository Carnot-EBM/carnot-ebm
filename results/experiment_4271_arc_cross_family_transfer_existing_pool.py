#!/usr/bin/env python3
"""Entrypoint for Exp 4271 ARC cross-family transfer on the existing pool."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.arc_cross_family_transfer_existing_pool_4271 import main  # noqa: E402


if __name__ == "__main__":
    main()
