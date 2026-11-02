#!/usr/bin/env python3
"""Compatibility wrapper for the Exp 3873 planned FoVer corpus builder."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.in_distribution_error_rich_corpus import cli_main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        cli_main(
            compatibility_label="experiment_3873_in_distribution_error_rich_corpus.json"
        )
    )
