#!/usr/bin/env python3
"""CLI wrapper for the unattended tool-gap refinement step (REQ-ARC-WMTE-6770).

WHY a wrapper: the logic lives in the importable package module so tests and
the conductor can call it directly; this file only makes it invocable as
`.venv/bin/python scripts/arc_tool_gap_refine.py [inputs ...]`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from carnot.agentic.arc_tool_gap_refinement import main

if __name__ == "__main__":
    raise SystemExit(main())
