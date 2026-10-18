#!/usr/bin/env python3
"""Run the task-owned 32K ARC code-carrying tool preflight."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.experiment_6752_arc_code_carrying_tool_preflight import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
