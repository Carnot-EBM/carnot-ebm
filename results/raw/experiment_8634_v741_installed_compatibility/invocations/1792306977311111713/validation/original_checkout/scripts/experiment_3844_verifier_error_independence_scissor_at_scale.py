#!/usr/bin/env python3
"""Run Exp 3844 verifier error-independence scissor at scale.

Spec: REQ-VERIFY-3844, SCENARIO-VERIFY-3844.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.eval.verifier_error_independence_scissor_at_scale import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_ITEMS,
    DEFAULT_RANDOM_SEED,
    ExperimentConfig,
    run_experiment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--max-tokens", type=int, default=10)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    artifact = run_experiment(
        ExperimentConfig(
            repo_root=repo_root,
            output_path=Path(args.output_path) if args.output_path else None,
            n_items=args.n_items,
            random_seed=args.random_seed,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_resamples=args.bootstrap_resamples,
            max_tokens=args.max_tokens,
        ),
        write=True,
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
