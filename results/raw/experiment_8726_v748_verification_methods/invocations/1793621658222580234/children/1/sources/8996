#!/usr/bin/env python3
"""Run the V626 verifier-committed uncertainty routing comparison."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from carnot.experiment_7130_v626_verifier_committed_routing import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
