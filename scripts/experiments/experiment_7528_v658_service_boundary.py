#!/usr/bin/env python3
"""Run the V658 count-memory service-boundary experiment."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "python", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from carnot.experiment_7528_v658_service_boundary import main


if __name__ == "__main__":
    raise SystemExit(main())
