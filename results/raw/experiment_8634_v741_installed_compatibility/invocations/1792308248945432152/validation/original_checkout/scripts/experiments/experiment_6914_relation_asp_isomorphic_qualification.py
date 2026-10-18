#!/usr/bin/env python3
"""Run Exp6914 relation ASP and isomorphic qualification."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6914_relation_asp_isomorphic_qualification import main


if __name__ == "__main__":  # pragma: no cover - this file is the command entrypoint.
    raise SystemExit(main())
