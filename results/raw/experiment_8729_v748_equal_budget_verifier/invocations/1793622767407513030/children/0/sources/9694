#!/usr/bin/env python3
"""Inspect only new supervisor outcomes. Spec: REQ-REPORT-8120.

Script-path execution adds the checkout explicitly so cold replay can start
from a private working directory without borrowing an ambient PYTHONPATH.
"""

import os
from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
os.environ["PYTHONUNBUFFERED"] = "1"

from carnot.reporting.arc_supervisor_v702_delta import scope  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Reuse the qualified private-control and authenticated-evidence routes."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
