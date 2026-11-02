#!/usr/bin/env python3
"""Run the V627 authentic chronological event-stream producer."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from carnot.experiment_7141_v627_csl_event_stream import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
