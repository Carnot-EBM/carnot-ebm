"""REQ-VICTRIS-9258-V788: offline Victris Arena Gibbs tactical state-energy scorer and bounded RSI archive."""

from __future__ import annotations

from carnot.victris.energy_bot import (
    FEATURE_NAMES,
    UNTRUSTED_SHOUT_KEYS,
    GibbsEnergyParams,
    VictrisEnergyBot,
    decide,
    default_energy_params,
    extract_candidate_features,
    generate_candidate_turns,
    sanitize_observation,
    score_candidate_energy,
    shoot_bot,
    synthetic_tactical_observations,
)
from carnot.victris.rsi_loop import (
    DEV_SEEDS,
    HELDOUT_SEEDS,
    RSIArchiveRunner,
    detect_arena_sim,
    enforce_tick_budget,
    summarize_tick_latencies,
    validate_seed_split,
    wilson_ci95,
)

__all__ = [
    "DEV_SEEDS",
    "FEATURE_NAMES",
    "GibbsEnergyParams",
    "HELDOUT_SEEDS",
    "RSIArchiveRunner",
    "UNTRUSTED_SHOUT_KEYS",
    "VictrisEnergyBot",
    "decide",
    "default_energy_params",
    "detect_arena_sim",
    "enforce_tick_budget",
    "extract_candidate_features",
    "generate_candidate_turns",
    "sanitize_observation",
    "score_candidate_energy",
    "shoot_bot",
    "summarize_tick_latencies",
    "synthetic_tactical_observations",
    "validate_seed_split",
    "wilson_ci95",
]
