#!/usr/bin/env python3
"""Repository entry point for REQ-CL-6798."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_6798_csl_causal_safety_byte_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
