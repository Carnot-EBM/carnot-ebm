#!/usr/bin/env python3
"""Exp 1223 — Phase 5-B in-situ training loop with verifier-ensemble grounding.

Spec: openspec/change-proposals/in-situ-training-phase5-derisking.md
      (exp_NEXT_B acceptance gate) and REQ-KONA-017 +
      SCENARIO-KONA-017 in openspec/capabilities/phase3-kona/spec.md.

Acceptance gate (per spec): the in-situ training loop runs a
1000-query trajectory with k=3 verifier-ensemble grounding, applies
CD-1 updates to the encoder + energy MLP whenever the verifier
ensemble accepts, and reports five Q9 stability gates.  ALL FIVE must
pass for ``phase5b_stability_confirmed = True``.  The runner writes
the artifact regardless of gate outcomes — partial-pass results are
honest, not blocked.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.phase5.insitu_training_loop import (
    DEFAULT_LEARNING_RATE,
    build_phase5b_artifact,
    confirm_phase5a_ready,
    evaluate_phase5b_gates,
    run_phase5b_training_loop,
    write_phase5b_artifact,
)

PHASE5A_ARTIFACT = Path("results/experiment_1222_phase5a_insitu_prototype.json")
ARTIFACT_PATH = Path("results/experiment_1223_phase5b_insitu_training_loop.json")
N_QUERIES = 1000
SEED = 1223
# DEFAULT_LEARNING_RATE = 1e-3 — see insitu_training_loop.py header for
# why the proposal's η=1e-5 is too small to evaluate Gate 1 in 1000 queries.
LEARNING_RATE = DEFAULT_LEARNING_RATE


def main() -> int:
    if not confirm_phase5a_ready(PHASE5A_ARTIFACT):
        artifact = build_phase5b_artifact(
            diagnostics={},
            gates={},
            seed=SEED,
            learning_rate_used=LEARNING_RATE,
            blocked=True,
            blocked_reason="phase5a_prototype_not_ready",
        )
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_phase5b_artifact(artifact, ARTIFACT_PATH)
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 2
    diagnostics = run_phase5b_training_loop(
        n_queries=N_QUERIES, seed=SEED, learning_rate=LEARNING_RATE
    )
    gates = evaluate_phase5b_gates(diagnostics)
    artifact = build_phase5b_artifact(
        diagnostics, gates, seed=SEED, learning_rate_used=LEARNING_RATE
    )
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_phase5b_artifact(artifact, ARTIFACT_PATH)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["phase5b_stability_confirmed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
