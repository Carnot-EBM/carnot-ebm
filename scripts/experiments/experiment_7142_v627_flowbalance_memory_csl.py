#!/usr/bin/env python3
"""Run the V627 frozen-Qwen verifier-balanced memory comparison."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from carnot.experiment_7142_v627_flowbalance_memory_csl import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
