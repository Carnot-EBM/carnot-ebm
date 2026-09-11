#!/usr/bin/env python3
"""Run the V635 refinement cold and causal audit."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7214_v635_refinement_cold_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
