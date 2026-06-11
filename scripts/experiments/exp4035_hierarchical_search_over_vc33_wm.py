"""Exp 4035: hierarchical search over the vc33 verified world model.

Spec refs: REQ-PHASE4-037, SCENARIO-PHASE4-037.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_vc33_hierarchical_search import (  # noqa: E402
    DEFAULT_MAX_BRANCHING,
    DEFAULT_MAX_EXPANSIONS,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-expansions", type=int, default=DEFAULT_MAX_EXPANSIONS)
    parser.add_argument("--max-branching", type=int, default=DEFAULT_MAX_BRANCHING)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(
        repo_root=REPO,
        write=not args.no_write,
        max_expansions=args.max_expansions,
        max_branching=args.max_branching,
    )
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - exercised by the required experiment command
    main()
