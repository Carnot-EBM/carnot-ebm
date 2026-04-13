#!/usr/bin/env python3
"""Experiment 241: held-out live self-learning replay benchmark v2.

Writes:
- ``results/experiment_241_results.json``

Spec: REQ-VERIFY-054, REQ-VERIFY-055,
SCENARIO-VERIFY-060, SCENARIO-VERIFY-061, SCENARIO-VERIFY-062
"""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.pipeline.self_learning_replay import RESULT_OUTPUT_V2, run_experiment_v2
from carnot.pipeline.self_learning_replay import get_repo_root as _get_repo_root


def get_repo_root() -> Path:
    return _get_repo_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Exp 241 held-out live self-learning replay v2 artifact."
    )
    parser.add_argument(
        "--output",
        default=str(RESULT_OUTPUT_V2),
        help="Relative output path for results/experiment_241_results.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_experiment_v2(
        repo_root=get_repo_root(),
        result_path=Path(args.output),
    )
    return 0


if __name__ == "__main__":
    main()
