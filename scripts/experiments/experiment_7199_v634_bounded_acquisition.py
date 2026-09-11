#!/usr/bin/env python3
"""Run the V634 bounded exact version-space acquisition experiment."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7199_v634_bounded_acquisition import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
