#!/usr/bin/env python3
"""Inspect authenticated supervisor outcomes without models. REQ-REPORT-8202.

Absolute checkout paths permit private CLI checks without PYTHONPATH. The
qualified runner seals evidence and publishes the same validated bytes.
"""

import os
from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
os.environ["PYTHONUNBUFFERED"] = "1"

from carnot.reporting.arc_supervisor_v708_frontier import scope  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Keep private reductions, cold replay and production on the qualified routes."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
