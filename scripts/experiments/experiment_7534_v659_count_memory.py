#!/usr/bin/env python3
"""Run the reusable REQ-CL-7534 count-memory qualification."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from carnot.experiment_7534_v659_count_memory import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
