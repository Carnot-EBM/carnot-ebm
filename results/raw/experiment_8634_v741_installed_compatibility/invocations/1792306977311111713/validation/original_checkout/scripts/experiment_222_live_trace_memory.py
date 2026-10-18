#!/usr/bin/env python3
"""Experiment 222: live trace memory and repair guidance.

Writes:
- ``results/experiment_222_results.json``
- ``results/constraint_memory_live_222.json``

Spec: REQ-VERIFY-030, REQ-VERIFY-031, REQ-VERIFY-032,
SCENARIO-VERIFY-030, SCENARIO-VERIFY-031, SCENARIO-VERIFY-032
"""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.pipeline.live_trace_memory import (
    MEMORY_OUTPUT,
    RESULT_OUTPUT,
    run_experiment,
)
from carnot.pipeline.live_trace_memory import (
    get_repo_root as _get_repo_root,
)


def get_repo_root() -> Path:
    return _get_repo_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Exp 222 live trace memory and repair-guidance artifacts."
    )
    parser.add_argument(
        "--output",
        default=str(RESULT_OUTPUT),
        help="Relative output path for results/experiment_222_results.json",
    )
    parser.add_argument(
        "--memory-output",
        default=str(MEMORY_OUTPUT),
        help="Relative output path for results/constraint_memory_live_222.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = get_repo_root()
    run_experiment(
        repo_root=repo_root,
        result_path=Path(args.output),
        memory_path=Path(args.memory_output),
    )
    return 0


if __name__ == "__main__":
    main()
