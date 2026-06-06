#!/usr/bin/env python3
"""Run Exp 3887 facts complementarity from cached Exp 3886 scores.

Spec refs: REQ-VERIFY-3887, SCENARIO-VERIFY-3887.
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
        help="Repository root containing the cached Exp 3886 artifact.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.repo_root.resolve()
    sys.path.insert(0, str(root / "python"))
    from carnot.verify.facts_complementarity_3887 import write_artifact

    output = write_artifact(root)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
