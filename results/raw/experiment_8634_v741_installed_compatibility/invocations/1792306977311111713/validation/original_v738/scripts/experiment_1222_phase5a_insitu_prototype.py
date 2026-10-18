#!/usr/bin/env python3
"""Exp 1222 — Phase 5-A in-situ training substrate prototype runner.

Spec: openspec/change-proposals/in-situ-training-phase5-derisking.md
      (exp_NEXT_A acceptance gate).

Acceptance gate (per spec): the prototype runs end-to-end on 100 random 5x5
puzzles, produces valid action sequences for ≥50%, and records the
vacuous-anchor distance and conditional-acceptance probability matrices.

This script is a *runner only* — it constructs the prototype, executes
100 random-puzzle queries with no weight updates, and writes the
artifact required by the conductor's reconciliation step to
``results/experiment_1222_phase5a_insitu_prototype.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.phase5.insitu_prototype import (
    build_phase5a_artifact,
    run_phase5a_prototype,
    write_phase5a_artifact,
)

ARTIFACT_PATH = Path("results/experiment_1222_phase5a_insitu_prototype.json")
N_PUZZLES = 100
SEED = 1222


def main() -> int:
    summary = run_phase5a_prototype(n_puzzles=N_PUZZLES, seed=SEED)
    artifact = build_phase5a_artifact(summary, seed=SEED)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_phase5a_artifact(artifact, ARTIFACT_PATH)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["phase5a_prototype_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
