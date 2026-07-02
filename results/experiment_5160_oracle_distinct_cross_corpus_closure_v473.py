#!/usr/bin/env python3
"""Entrypoint for Exp 5160 oracle-distinct cross-corpus closure."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.reporting.oracle_distinct_cross_corpus_closure_5160 import main  # noqa: E402


if __name__ == "__main__":
    main()
