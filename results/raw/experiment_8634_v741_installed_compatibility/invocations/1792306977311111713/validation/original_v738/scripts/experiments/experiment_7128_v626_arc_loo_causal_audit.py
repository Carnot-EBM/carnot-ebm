#!/usr/bin/env python3
"""Run the REQ-ARC-7128 independent ARC raw-trace audit."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_7128_v626_arc_loo_causal_audit import main


if __name__ == "__main__":  # pragma: no cover - required command surface.
    raise SystemExit(main())
