#!/usr/bin/env python3
"""Run the environment-indexed runtime proof grammar audit."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_6769_environment_indexed_proof_grammar_v2 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
