#!/usr/bin/env python3
"""Run the Exp6915 deterministic qualified relation event-bank merge."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6915_qualified_relation_event_bank import main


if __name__ == "__main__":  # pragma: no cover - this file is the command entrypoint.
    raise SystemExit(main())
