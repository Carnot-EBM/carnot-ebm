#!/usr/bin/env python3
"""Run Exp7010 from a repository checkout (REQ-ARC-7010)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from carnot.experiment_7010_arc_eval_provenance_contract import main


if __name__ == "__main__":
    raise SystemExit(main())
