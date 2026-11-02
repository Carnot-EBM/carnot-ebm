#!/usr/bin/env python3
"""REQ-VERIFY-8516: direct invocation uses the tested bounded publication runner."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.objective_publication_runner_8516 import main

if __name__ == "__main__":
    raise SystemExit(main())
