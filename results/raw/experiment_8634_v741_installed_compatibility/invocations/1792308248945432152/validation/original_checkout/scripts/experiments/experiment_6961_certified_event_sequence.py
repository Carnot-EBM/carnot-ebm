#!/usr/bin/env python3
"""Build the sealed prospective exact-certificate sequence.

Spec ref: REQ-LEARN-6961.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6961_certified_event_sequence import main


if __name__ == "__main__":  # pragma: no cover - required command surface.
    raise SystemExit(main())
