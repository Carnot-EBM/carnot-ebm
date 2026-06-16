#!/usr/bin/env python3
"""Entrypoint for Exp 4291 ARC-GEN cross-generator non-degenerate replication."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.arcgen_cross_generator_nondegenerate_4291 import main  # noqa: E402


if __name__ == "__main__":
    main()
