#!/usr/bin/env python3
"""Run independently qualified reader custody. Spec: REQ-REPORT-8147.

Explicit checkout paths allow private cold replay without PYTHONPATH. The shared
runner owns validation and atomic publication, keeping this CLI small.
"""

import os
from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
os.environ["PYTHONUNBUFFERED"] = "1"

from carnot.reporting.arc_supervisor_v704_renewal import scope  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Use the same reader for private controls and the current receipt."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
