"""Entrypoint for Exp 4262 ARC-AGI-3 scored-only live accuracy probe."""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.experiment_4262_arc_live_env_accuracy_probe import main  # noqa: E402


if __name__ == "__main__":
    main()
