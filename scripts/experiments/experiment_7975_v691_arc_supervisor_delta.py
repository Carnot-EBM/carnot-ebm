#!/usr/bin/env python3
"""Parameterize the qualified receipt CLI. Spec: REQ-REPORT-7975."""

from carnot.reporting import arc_supervisor_v691_delta as task
from scripts.experiments.experiment_7962_v690_arc_supervisor_delta import main as run


def main(argv: list[str] | None = None) -> int:
    """Reuse private reducer and replay routes without another dispatcher."""
    return run(argv, scope=task)


if __name__ == "__main__":
    raise SystemExit(main())
