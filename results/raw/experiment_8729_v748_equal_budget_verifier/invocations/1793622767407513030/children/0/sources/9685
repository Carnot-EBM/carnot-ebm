#!/usr/bin/env python3
"""Read only qualified new supervisor outcomes. Spec: REQ-REPORT-8107."""

import os
from pathlib import Path
import sys

# External replay must import the same code without an ambient checkout cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["PYTHONUNBUFFERED"] = "1"

from carnot.reporting.arc_supervisor_v701_evidence import scope  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """The shared CLI separates private controls from authenticated live evidence."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
