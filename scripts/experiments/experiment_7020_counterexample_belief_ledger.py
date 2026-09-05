#!/usr/bin/env python3
"""Run the transactional ARC belief ledger (REQ-CSL-7020)."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.agentic.arc_belief_ledger import main


if __name__ == "__main__":  # pragma: no cover - required command surface.
    raise SystemExit(main())
