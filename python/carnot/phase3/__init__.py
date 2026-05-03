"""Phase 3 seed modules — bridging discrete Ising to continuous energy landscapes.

**Phase 3 North Star:** functional parity with Kona (Logical Intelligence) —
continuous-latent, non-autoregressive reasoning.  Nothing in this package is
production-ready; these are concrete seeds for that long-horizon goal.

Spec: REQ-KONA-001, REQ-KONA-002, REQ-KONA-003
"""

from carnot.phase3.continuous_ebm import (
    ContinuousEBM,
    build_kona_artifact,
    compare_minima,
    compare_samplers,
    fit_continuous_ebm,
    sample_continuous,
    sample_energy_matching,
    sample_langevin,
)
from carnot.phase3.active_inference_pilot import (
    ARC3Action,
    ARC3PuzzleEnv,
    ActiveInferencePilot,
    BoardState,
    EpisodeResult,
    PuzzleSpec,
    build_default_k5_ensemble_energies,
    build_experiment_artifact as build_active_inference_artifact,
    run_phase4_vs_baseline,
    run_random_baseline_episode,
    write_experiment_artifact as write_active_inference_artifact,
)
from carnot.phase3.nrgpt_energy import (
    NRGPTEnergyBlock,
    build_artifact as build_nrgpt_artifact,
    run_experiment as run_nrgpt_experiment,
    train_and_compare as train_and_compare_nrgpt,
)
from carnot.phase3.snap_validity import (
    SnapSweepConfig,
    build_snap_validity_artifact,
    build_synthetic_action_space,
    infer_latent_dim,
    run_snap_validity_sweep,
    sample_uniform_latents,
    snap_to_action,
    snap_states_to_actions,
    snap_validity_verdict,
    snapped_actions_legal_mask,
)

__all__ = [
    "ARC3Action",
    "ARC3PuzzleEnv",
    "ActiveInferencePilot",
    "BoardState",
    "ContinuousEBM",
    "EpisodeResult",
    "NRGPTEnergyBlock",
    "PuzzleSpec",
    "build_active_inference_artifact",
    "build_default_k5_ensemble_energies",
    "SnapSweepConfig",
    "build_nrgpt_artifact",
    "build_kona_artifact",
    "build_snap_validity_artifact",
    "build_synthetic_action_space",
    "compare_minima",
    "compare_samplers",
    "fit_continuous_ebm",
    "infer_latent_dim",
    "run_phase4_vs_baseline",
    "run_random_baseline_episode",
    "run_nrgpt_experiment",
    "run_snap_validity_sweep",
    "sample_continuous",
    "sample_energy_matching",
    "sample_langevin",
    "sample_uniform_latents",
    "snap_to_action",
    "snap_states_to_actions",
    "snap_validity_verdict",
    "snapped_actions_legal_mask",
    "train_and_compare_nrgpt",
    "write_active_inference_artifact",
]
