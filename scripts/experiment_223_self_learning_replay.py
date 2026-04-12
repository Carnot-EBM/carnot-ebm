#!/usr/bin/env python3
"""Experiment 223: held-out live self-learning replay benchmark.

Writes:
- ``results/experiment_223_results.json``

Spec: REQ-VERIFY-033, REQ-VERIFY-034, REQ-VERIFY-035,
SCENARIO-VERIFY-033, SCENARIO-VERIFY-034, SCENARIO-VERIFY-035
"""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.pipeline.self_learning_replay import RESULT_OUTPUT, run_experiment
from carnot.pipeline.self_learning_replay import get_repo_root as _get_repo_root


def get_repo_root() -> Path:
    return _get_repo_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Exp 223 held-out live self-learning replay artifact."
    )
    parser.add_argument(
        "--output",
        default=str(RESULT_OUTPUT),
        help="Relative output path for results/experiment_223_results.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_experiment(
        repo_root=get_repo_root(),
        result_path=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    main()
