"""Phase 5 in-situ training derisking — small-scale prototype seeds.

**Phase 5 North Star:** model weights update DURING inference using the
verifier ensemble as supervision.  Addresses Sakana DGM's reward-hacking
open problem by keeping the verifier frozen while the substrate adapts.

This package contains the small-scale (~50K param) prototype components
that exp_NEXT_A in `openspec/change-proposals/in-situ-training-phase5-derisking.md`
requires before scaling to 1B+ substrate.

Spec: REQ-KONA-008 (snap-to-action reuse), REQ-KONA-012 (active inference),
      change-proposal `in-situ-training-phase5-derisking`.
"""

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

__all__ = [
    "ConditionalAcceptanceProbMatrix",
    "InSituEncoder",
    "InSituEnergyMLP",
    "VacuousAnchorTracker",
    "apply_action_sequence",
    "build_phase5a_artifact",
    "generate_random_5x5_puzzle",
    "run_phase5a_prototype",
    "snap_to_action",
    "verify_action_sequence",
]
