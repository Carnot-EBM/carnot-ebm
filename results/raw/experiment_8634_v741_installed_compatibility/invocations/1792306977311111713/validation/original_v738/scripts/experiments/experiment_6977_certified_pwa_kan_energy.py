#!/usr/bin/env python3
"""Run calibration-only PWA-KAN residual certification.

Spec ref: REQ-KAN-6977.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6977_certified_pwa_kan_energy import main


if __name__ == "__main__":  # pragma: no cover - required command boundary.
    raise SystemExit(main())
