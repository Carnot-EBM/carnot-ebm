"""Exp 4046: closed-loop replanning over the vc33 verified world model.

Spec refs: REQ-PHASE4-040, SCENARIO-PHASE4-040.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_vc33_closed_loop_replan import (  # noqa: E402
    DEFAULT_DIVERGENCE_THRESHOLD,
    DEFAULT_HORIZON,
    DEFAULT_MAX_PLAN_EXPANSIONS,
    DEFAULT_MAX_REAL_STEPS,
    DEFAULT_MAX_BRANCHING,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--max-plan-expansions", type=int, default=DEFAULT_MAX_PLAN_EXPANSIONS)
    parser.add_argument("--max-branching", type=int, default=DEFAULT_MAX_BRANCHING)
    parser.add_argument("--max-real-steps", type=int, default=DEFAULT_MAX_REAL_STEPS)
    parser.add_argument("--divergence-threshold", type=float, default=DEFAULT_DIVERGENCE_THRESHOLD)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(
        repo_root=REPO,
        write=not args.no_write,
        horizon=args.horizon,
        max_plan_expansions=args.max_plan_expansions,
        max_branching=args.max_branching,
        max_real_steps=args.max_real_steps,
        divergence_threshold=args.divergence_threshold,
    )
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - exercised by required experiment command
    main()
