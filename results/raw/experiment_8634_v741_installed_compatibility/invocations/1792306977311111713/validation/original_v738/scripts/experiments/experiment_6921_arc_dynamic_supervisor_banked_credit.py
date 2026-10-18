#!/usr/bin/env python3
"""Run the deterministic REQ-ARC-WMTE-6921 receipt audit."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.experiment_6921_arc_dynamic_supervisor_banked_credit import main


if __name__ == "__main__":
    raise SystemExit(main())
