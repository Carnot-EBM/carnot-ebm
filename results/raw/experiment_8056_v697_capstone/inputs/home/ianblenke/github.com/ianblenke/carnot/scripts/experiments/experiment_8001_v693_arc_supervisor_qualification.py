#!/usr/bin/env python3
"""Qualify supervisor evidence without model calls. REQ-REPORT-8001."""

from pathlib import Path
import sys

# Cold replay starts in private scratch, so expose the repository CLI package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.reporting import arc_supervisor_qualification as task  # noqa: E402
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Use the tested private routes and current frozen qualification scope."""
    return run(argv, scope=task)


if __name__ == "__main__":
    raise SystemExit(main())
