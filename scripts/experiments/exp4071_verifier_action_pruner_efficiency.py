"""Exp 4071: verifier-as-action-pruner efficiency on solved ARC-AGI-3 traces.

Spec refs: REQ-PHASE4-043, SCENARIO-PHASE4-043.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_exp4071_verifier_action_pruner_efficiency import run_experiment  # noqa: E402


def main() -> int:
    artifact = run_experiment(repo_root=REPO)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
