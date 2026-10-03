#!/usr/bin/env python3
"""Audit receipt deltas without starting games. Spec: REQ-REPORT-7988."""

from pathlib import Path
import sys

# The script must find repository readers when cold replay starts elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting import arc_supervisor_v692_delta as task  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Reuse bounded private routes so fixture results cannot become live evidence."""
    return run(argv, scope=task)


if __name__ == "__main__":
    raise SystemExit(main())
