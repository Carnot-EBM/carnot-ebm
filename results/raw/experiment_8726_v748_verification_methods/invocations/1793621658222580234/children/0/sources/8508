"""Exp 4034: induce a vc33 goal predicate over the verified world-model substrate.

Spec refs: REQ-PHASE4-036, SCENARIO-PHASE4-036.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_vc33_goal_predicate_induction import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(repo_root=REPO, write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover - exercised by the required experiment command
    main()
