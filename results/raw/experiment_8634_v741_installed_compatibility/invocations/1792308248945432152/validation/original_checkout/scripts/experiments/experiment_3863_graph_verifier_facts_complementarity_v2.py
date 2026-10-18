#!/usr/bin/env python3
"""Run Exp 3863 graph-verifier facts complementarity from cached scores.

Spec: REQ-VERIFY-3863, SCENARIO-VERIFY-3863.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_repo_root(),
        help="Repository root containing the cached Exp 3862 artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root / "python"))
    from carnot.verify.graph_verifier_facts_complementarity_v2 import write_artifact

    output = write_artifact(root)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
