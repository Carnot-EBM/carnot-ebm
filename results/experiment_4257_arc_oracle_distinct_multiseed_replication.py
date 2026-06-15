#!/usr/bin/env python3
"""Entrypoint for Exp 4257 ARC oracle-distinct multi-seed replication."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.arc_oracle_distinct_multiseed_replication_4257 import main  # noqa: E402


if __name__ == "__main__":
    main()
