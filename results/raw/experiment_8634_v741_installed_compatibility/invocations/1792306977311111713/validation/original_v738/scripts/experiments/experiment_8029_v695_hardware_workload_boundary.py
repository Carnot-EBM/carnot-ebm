#!/usr/bin/env python3
"""REQ-REPORT-8029: route the runnable script to the measured package code."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.experiment_8029_v695_hardware_workload_boundary import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
