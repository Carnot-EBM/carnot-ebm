#!/usr/bin/env python3
"""Run the REQ-KAN-7506 fixture qualification from the repository root."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from carnot.experiment_7506_v657_causal_prototype import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
