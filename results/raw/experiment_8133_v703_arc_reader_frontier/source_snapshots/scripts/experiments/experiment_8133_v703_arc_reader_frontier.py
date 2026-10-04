#!/usr/bin/env python3
"""Qualify current supervisor behavior. Spec: REQ-REPORT-8133.

Explicit checkout paths let the script replay from private working directories
without an ambient PYTHONPATH. No model or game is needed to read evidence.
"""

import os
from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
os.environ["PYTHONUNBUFFERED"] = "1"

from carnot.reporting.arc_supervisor_v703_frontier import scope  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Reuse private control routes so current evidence has one publication path."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
