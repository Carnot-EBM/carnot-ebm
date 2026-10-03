#!/usr/bin/env python3
"""Read the supervisor ledger without new game work. Spec: REQ-REPORT-8080."""

import os
from pathlib import Path
import sys

# Private replay must find the same reader when started outside the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["PYTHONUNBUFFERED"] = "1"

from carnot.reporting.arc_supervisor_v699_frontier import scope  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """The shared CLI separates private controls from authenticated live evidence."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
