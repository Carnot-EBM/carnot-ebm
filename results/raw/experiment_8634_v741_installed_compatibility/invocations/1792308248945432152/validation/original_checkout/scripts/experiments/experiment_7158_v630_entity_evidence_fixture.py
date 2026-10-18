#!/usr/bin/env python3
"""Run the V630 exact entity-to-evidence fixture builder from a checkout."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_7158_v630_entity_evidence_fixture import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
