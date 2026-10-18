#!/usr/bin/env python3
"""Run Exp 3351 GateMate latency benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_3351_gatemate_latency_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
