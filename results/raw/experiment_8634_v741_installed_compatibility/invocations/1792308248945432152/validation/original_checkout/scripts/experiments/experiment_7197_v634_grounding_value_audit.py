#!/usr/bin/env python3
"""Run the V634 typed-grounding value and semantic audit."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7197_v634_grounding_value_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
