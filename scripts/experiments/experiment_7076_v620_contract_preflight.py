#!/usr/bin/env python3
"""Run the V620 active-roadmap contract preflight from the repository."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from carnot.experiment_7076_v620_contract_preflight import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
