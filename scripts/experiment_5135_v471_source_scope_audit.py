#!/usr/bin/env python3
"""CLI entrypoint for Exp 5135 V471 source/scope audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.experiment_5135_v471_source_scope_audit import DEFAULT_TESTS_RUN, REPO_ROOT, run


def main(
    *,
    root: Path = REPO_ROOT,
    output: Path | None = None,
    date: str = "20260702",
    duration_s: float | None = None,
    tests_run: list[str] | None = None,
) -> Path:
    """Run the tested Exp 5135 audit implementation and return the artifact path."""

    return run(
        root=root,
        artifact_path=output,
        run_date=date,
        duration_s=duration_s,
        tests_run=tests_run if tests_run is not None else DEFAULT_TESTS_RUN,
    )


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260702", help="Run label date, formatted YYYYMMDD.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root to inspect.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    main(root=args.root, output=args.output, date=args.date)
