#!/usr/bin/env python3
"""Run the queue-regulated exact-certificate comparison.

Spec ref: REQ-LEARN-6962.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from carnot.experiment_6962_queue_regulated_self_learning import main


if __name__ == "__main__":  # pragma: no cover - required command surface.
    raise SystemExit(main())
