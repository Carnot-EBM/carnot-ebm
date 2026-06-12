"""CLI entrypoint for Exp 4077 verifier-reward RFT corpus build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carnot.agentic.arc_exp4077_verifier_reward_rft_corpus_build import (
    REPO_ROOT,
    RESULT_FILENAME,
    run_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / RESULT_FILENAME,
        help="Path for the terminal Exp 4077 JSON artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = run_experiment(output_path=args.output)
    print(json.dumps(artifact, sort_keys=True))


if __name__ == "__main__":
    main()
