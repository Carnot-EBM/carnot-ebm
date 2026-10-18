#!/usr/bin/env python3
"""Run REQ-INFERENCE-6868 from the repository checkout."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_6868_three_family_semantic_scoring_stream_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
