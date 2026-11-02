#!/usr/bin/env python3
"""Run the Exp7196 separated Qwen3.8 atomic capture."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7196_v634_qwen_atomic_capture import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
