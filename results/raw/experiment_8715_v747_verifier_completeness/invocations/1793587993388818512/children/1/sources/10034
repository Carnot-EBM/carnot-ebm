#!/usr/bin/env python3
"""REQ-VERIFY-8693: run the bounded CPU artifact mapper from any directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting.hardware_boundary_runner_8693 import main

if __name__ == "__main__":
    raise SystemExit(main())
