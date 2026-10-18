#!/usr/bin/env python3
"""Run the Exp6996 V613 source and task-contract preflight."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from carnot.experiment_6996_v613_source_contract_preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
