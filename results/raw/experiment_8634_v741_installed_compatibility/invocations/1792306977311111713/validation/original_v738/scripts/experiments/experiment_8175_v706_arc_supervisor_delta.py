#!/usr/bin/env python3
"""Read new live supervisor outcomes with no model. Spec: REQ-REPORT-8175.

Explicit checkout paths make the same small CLI usable from a private working
directory. The qualified runner keeps evidence and checked publication together.
"""

import os
from pathlib import Path
import sys

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]
os.environ["PYTHONUNBUFFERED"] = "1"

from carnot.reporting.arc_supervisor_v706_delta import scope  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Use the existing CLI routes so private controls share the live reader."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
