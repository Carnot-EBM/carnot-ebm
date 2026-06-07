#!/usr/bin/env python3
"""Exp 3895 tested-harness in-distribution moat scissor runner."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.moat_scissor_tested_harness import cli_main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main(["--repo-root", str(REPO_ROOT)]))
