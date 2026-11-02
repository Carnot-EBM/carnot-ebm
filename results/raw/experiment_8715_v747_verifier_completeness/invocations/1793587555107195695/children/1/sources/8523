"""CLI entrypoint for Exp 4078 verifier-reward RFT train launch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_exp4078_verifier_reward_rft_train_launch import (  # noqa: E402
    REPO_ROOT,
    RESULT_FILENAME,
    run_experiment,
    run_worker,
)


def parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / RESULT_FILENAME,
        help="Path for the terminal Exp 4078 JSON artifact.",
    )
    parser.add_argument("--worker", action="store_true", help="Run one detached base+arm worker.")
    parser.add_argument("--base-key", default="")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--arm", default="")
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--trust-remote-code", choices=("true", "false"), default="false")
    parser.add_argument("--training-config-json", default="{}")
    return parser.parse_args()


def main() -> int:  # pragma: no cover
    args = parse_args()
    if args.worker:
        run_worker(
            base_key=args.base_key,
            base_model=args.base_model,
            arm=args.arm,
            corpus_path=args.corpus_path,
            checkpoint_path=args.checkpoint_path,
            log_path=args.log_path,
            trust_remote_code=args.trust_remote_code == "true",
            training_config_json=args.training_config_json,
        )
        return 0

    artifact = run_experiment(output_path=args.output)
    print(json.dumps(artifact, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
