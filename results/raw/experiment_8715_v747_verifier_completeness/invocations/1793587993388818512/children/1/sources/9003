#!/usr/bin/env python3
"""Run the Exp7182 independent audit in its own Python process."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7182_v633_grounding_energy_independent_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
