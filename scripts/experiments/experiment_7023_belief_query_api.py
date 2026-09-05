#!/usr/bin/env python3
"""Run the deterministic bounded ARC belief-query experiment."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from carnot.agentic.arc_belief_query import main


if __name__ == "__main__":
    raise SystemExit(main())
