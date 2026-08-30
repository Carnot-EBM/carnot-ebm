#!/usr/bin/env python3
"""Repository entry point for REQ-CL-6791."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_6791_compositional_online_constraint_routing_ab import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
