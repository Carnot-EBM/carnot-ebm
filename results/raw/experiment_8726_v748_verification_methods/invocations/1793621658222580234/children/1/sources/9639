#!/usr/bin/env python3
"""Read new authenticated outcomes without game work. REQ-REPORT-8041."""

from pathlib import Path
import sys

# Cold replay starts in private scratch, so resolve the shared CLI explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting.arc_supervisor_v696_delta import scope  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """The qualified shared CLI keeps private fixtures separate from live evidence."""
    return run(argv, scope=scope)


if __name__ == "__main__":
    raise SystemExit(main())
