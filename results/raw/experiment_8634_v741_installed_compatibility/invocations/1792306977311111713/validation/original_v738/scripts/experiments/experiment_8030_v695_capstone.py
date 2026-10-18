#!/usr/bin/env python3
"""REQ-REPORT-8030: run the durable capstone reducer from any working directory."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting.v695_capstone import main  # noqa: E402

MODEL_SPECS: list[str] = []

if __name__ == "__main__":
    raise SystemExit(main())
