#!/usr/bin/env python3
"""Experiment 243: sampler-backed repair reranking replay benchmark.

Writes:
- ``results/experiment_243_results.json``

Spec: REQ-SAMPLE-008,
SCENARIO-SAMPLE-015, SCENARIO-SAMPLE-016, SCENARIO-SAMPLE-017
"""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.inference.repair_reranker import DEFAULT_OUTPUT, get_repo_root, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Exp 243 sampler-backed repair reranking replay artifact.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Relative output path for results/experiment_243_results.json",
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
