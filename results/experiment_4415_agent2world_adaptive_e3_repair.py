"""Run Exp 4415 Agent2World adaptive E3 repair."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from carnot.experiment_4415_agent2world_adaptive_e3_repair import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
