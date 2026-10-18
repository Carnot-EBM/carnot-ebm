"""Phase 5 in-situ training derisking — small-scale prototype seeds.

**Phase 5 North Star:** model weights update DURING inference using the
verifier ensemble as supervision.  Addresses Sakana DGM's reward-hacking
open problem by keeping the verifier frozen while the substrate adapts.

This package contains the small-scale (~50K param) prototype components
that exp_NEXT_A in `openspec/change-proposals/in-situ-training-phase5-derisking.md`
requires before scaling to 1B+ substrate.

Spec: REQ-KONA-008 (snap-to-action reuse), REQ-KONA-012 (active inference),
      REQ-KONA-018 (Phase 5-C adversarial probe),
      change-proposal `in-situ-training-phase5-derisking`.
"""

from carnot.phase5.adversarial_probe import (
    ATTACK1_GAMING_THRESHOLD,
    ATTACK2_CORRELATION_THRESHOLD,
    ATTACK3_GAMING_THRESHOLD,
    build_phase5c_artifact,
    evaluate_defense_verdict,
    run_attack1_single_verifier_gaming,
    run_attack2_pairwise_correlation,
    run_attack3_joint_nullspace,
    write_phase5c_artifact,
)
from carnot.phase5.insitu_prototype import (
    ConditionalAcceptanceProbMatrix,
    InSituEncoder,
    InSituEnergyMLP,
    VacuousAnchorTracker,
    apply_action_sequence,
    build_phase5a_artifact,
    generate_random_5x5_puzzle,
    run_phase5a_prototype,
    snap_to_action,
    verify_action_sequence,
)
from carnot.phase5.insitu_training_loop import (
    build_phase5b_artifact,
    cd1_update,
    confirm_phase5a_ready,
    encoder_forward_with_h,
    encoder_spectral_norm,
    evaluate_oracle,
    evaluate_phase5b_gates,
    run_phase5b_training_loop,
    verifier_ensemble_pass,
    write_phase5b_artifact,
)

__all__ = [
    "ATTACK1_GAMING_THRESHOLD",
    "ATTACK2_CORRELATION_THRESHOLD",
    "ATTACK3_GAMING_THRESHOLD",
    "ConditionalAcceptanceProbMatrix",
    "InSituEncoder",
    "InSituEnergyMLP",
    "VacuousAnchorTracker",
    "apply_action_sequence",
    "build_phase5a_artifact",
    "build_phase5b_artifact",
    "build_phase5c_artifact",
    "cd1_update",
    "confirm_phase5a_ready",
    "encoder_forward_with_h",
    "encoder_spectral_norm",
    "evaluate_defense_verdict",
    "evaluate_oracle",
    "evaluate_phase5b_gates",
    "generate_random_5x5_puzzle",
    "run_attack1_single_verifier_gaming",
    "run_attack2_pairwise_correlation",
    "run_attack3_joint_nullspace",
    "run_phase5a_prototype",
    "run_phase5b_training_loop",
    "snap_to_action",
    "verifier_ensemble_pass",
    "verify_action_sequence",
    "write_phase5b_artifact",
    "write_phase5c_artifact",
]
