#!/usr/bin/env python3
"""Run Exp 3928 independent-corpus moat scissor replication."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.moat_scissor_replication_3928 import cli_main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main(["--repo-root", str(REPO_ROOT)]))
