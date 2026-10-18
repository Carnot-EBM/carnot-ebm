#!/usr/bin/env python3
"""Run the Exp7107 fresh-process continual-memory cold audit."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7107_v623_continual_memory_cold_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
