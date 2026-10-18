"""CLI entrypoint for Exp 4088 trustworthy verifier-reward RFT corpus build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.agentic.arc_exp4088_verifier_reward_rft_corpus_build import (  # noqa: E402
    RESULT_FILENAME,
    run_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / RESULT_FILENAME),
        help="Path for the terminal Exp 4088 JSON artifact.",
    )
    args = parser.parse_args()
    artifact = run_experiment(repo_root=REPO_ROOT, output_path=args.output)
    print(artifact["honest_verdict"])
    return 0 if str(artifact["honest_verdict"]).startswith(("complete:", "blocked_")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
