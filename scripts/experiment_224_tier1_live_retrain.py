#!/usr/bin/env python3
"""Experiment 224: Tier 1 live-only ConstraintTracker retrain on Exp 219-221 traces.

Trains the Tier 1 online weights (ConstraintTracker) using ONLY the live
traces recorded in Exp 219, 220, and 221.  Evaluates on held-out (last 25%)
and compares to Exp 223 tracker_only baseline.

Writes:
- ``results/experiment_224_results.json`` — full evaluation artifact
- ``results/tier1_live_weights.json`` — trained ConstraintTracker weights

Spec: REQ-VERIFY-033, REQ-VERIFY-034, REQ-LEARN-001,
SCENARIO-VERIFY-033, SCENARIO-LEARN-001
"""

from __future__ import annotations

import argparse
from pathlib import Path

from carnot.pipeline.self_learning_replay import (
    RESULT_OUTPUT_224,
    WEIGHTS_OUTPUT_224,
    get_repo_root,
    run_experiment_224,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain Tier 1 ConstraintTracker on live traces from Exp 219-221 "
            "and write Exp 224 results + weights."
        )
    )
    parser.add_argument(
        "--output",
        default=str(RESULT_OUTPUT_224),
        help="Relative output path for results/experiment_224_results.json",
    )
    parser.add_argument(
        "--weights",
        default=str(WEIGHTS_OUTPUT_224),
        help="Relative output path for results/tier1_live_weights.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)  # validates args; run_experiment_224 uses constants
    run_experiment_224(repo_root=get_repo_root())
    return 0


if __name__ == "__main__":
    main()
