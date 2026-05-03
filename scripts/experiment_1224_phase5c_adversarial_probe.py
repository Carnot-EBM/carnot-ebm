#!/usr/bin/env python3
"""Exp 1224 — Phase 5-C adversarial probe runner.

Spec: openspec/change-proposals/in-situ-training-phase5-derisking.md
      (exp_NEXT_C acceptance gate) and REQ-KONA-018 in
      openspec/capabilities/phase3-kona/spec.md.

Runs three attack classes against the Phase 5-A+B prototype and writes
results/experiment_1224_phase5c_adversarial_probe.json.

Prerequisite: results/experiment_1223_phase5b_insitu_training_loop.json
must have phase5b_stability_confirmed = true.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

from carnot.phase5.adversarial_probe import (
    build_phase5c_artifact,
    evaluate_defense_verdict,
    run_attack1_single_verifier_gaming,
    run_attack2_pairwise_correlation,
    run_attack3_joint_nullspace,
    write_phase5c_artifact,
)
from carnot.phase5.insitu_prototype import (
    InSituEnergyMLP,
    InSituEncoder,
    InSituRefiner,
)

PHASE5B_PATH = Path("results/experiment_1223_phase5b_insitu_training_loop.json")
ARTIFACT_PATH = Path("results/experiment_1224_phase5c_adversarial_probe.json")
N_ATTACK_SAMPLES = 200
N_ATTACK3_STARTS = 20
SEED = 1224


def main() -> int:
    start_time = _dt.datetime.now(_dt.timezone.utc)

    # Prerequisite check: Phase 5-B must be confirmed stable.
    with open(PHASE5B_PATH) as fh:
        phase5b = json.load(fh)

    if not phase5b.get("phase5b_stability_confirmed"):
        artifact = {
            "experiment": "1224_phase5c_adversarial_probe",
            "status": "blocked",
            "adversarial_probe_complete": False,
            "honest_verdict": "blocked",
            "blocked_reason": "phase5b_stability_confirmed is False",
        }
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_phase5c_artifact(artifact, ARTIFACT_PATH)
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 1

    # Initialise model components (reproducible seeds matching Phase 5-A).
    encoder = InSituEncoder.init(seed=SEED)
    energy_mlp = InSituEnergyMLP.init(seed=SEED + 1)
    refiner = InSituRefiner.init(seed=SEED + 2)

    # Attack 1 — single-verifier gaming.
    gaming_rate_attack1 = run_attack1_single_verifier_gaming(
        encoder, refiner, n_samples=N_ATTACK_SAMPLES, seed=SEED
    )
    print(f"Attack 1 gaming rate: {gaming_rate_attack1:.4f}")

    # Attack 2 — pairwise correlation exploitation.
    pairwise_max_correlation, cap_matrix = run_attack2_pairwise_correlation(
        encoder, refiner, n_samples=N_ATTACK_SAMPLES, seed=SEED + 1
    )
    print(f"Attack 2 max pairwise correlation: {pairwise_max_correlation:.4f}")

    # Attack 3 — joint null-space gradient attack.
    joint_gaming_rate = run_attack3_joint_nullspace(
        encoder, refiner, energy_mlp,
        n_starts=N_ATTACK3_STARTS, seed=SEED + 2,
    )
    print(f"Attack 3 joint gaming rate: {joint_gaming_rate:.4f}")

    # Defense verdict.
    verdict = evaluate_defense_verdict(
        gaming_rate_attack1, pairwise_max_correlation, joint_gaming_rate, cap_matrix
    )
    print(f"all_attacks_blocked: {verdict['all_attacks_blocked']}")
    print(f"honest_verdict: {verdict['honest_verdict']}")

    # Build and write artifact.
    artifact = build_phase5c_artifact(
        start_time=start_time,
        seed=SEED,
        gaming_rate_attack1=gaming_rate_attack1,
        pairwise_max_correlation=pairwise_max_correlation,
        joint_gaming_rate=joint_gaming_rate,
        conditional_matrix=cap_matrix,
        verdict_dict=verdict,
        phase5b_stability_confirmed=bool(phase5b.get("phase5b_stability_confirmed")),
    )
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_phase5c_artifact(artifact, ARTIFACT_PATH)
    print(f"\nArtifact written to {ARTIFACT_PATH}")
    return 0 if verdict["all_attacks_blocked"] else 1


if __name__ == "__main__":
    sys.exit(main())
