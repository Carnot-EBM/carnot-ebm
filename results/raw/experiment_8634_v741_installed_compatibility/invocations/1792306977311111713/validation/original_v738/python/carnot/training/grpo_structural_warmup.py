"""Structural warm-up helpers for Exp 1159 GRPO v4.

This module holds the pure reward-schedule and artifact-schema logic for
Exp 1159 so tests can cover it without loading the live 35B GGUF model.

Spec: REQ-LEARN-1159, SCENARIO-LEARN-1159, SCENARIO-LEARN-1160.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EXP1129_IMPROVEMENT = 0.0851
REFLECTION_WEIGHT_FULL = 0.3
WARMUP_SECONDS = 300
TRAINING_SECONDS = 900
GROUP_SIZE_N = 8

REQUIRED_ARTIFACT_FIELDS = (
    "dualgpu_used",
    "cuda_device_count",
    "warmup_seconds",
    "training_seconds",
    "training_wall_budget_hit",
    "advantage_stdev_warmup",
    "advantage_stdev_full",
    "n_eval_questions",
    "baseline_fraction_correct",
    "trained_fraction_correct",
    "improvement_over_baseline",
    "improvement_vs_exp1129",
    "reflection_weight",
    "structural_warmup_used",
    "grpo_v4_honest_result",
    "honest_verdict",
)


@dataclass(frozen=True)
class PhaseConfig:
    """Reward and rollout configuration for one Exp 1159 training phase."""

    name: str
    wall_budget_s: float
    group_size_n: int
    thinkprm_weight: float
    reflection_weight: float
    diversity_penalty_enabled: bool
    proxy_reuse_enabled: bool


def build_structural_warmup_phase_configs() -> tuple[PhaseConfig, PhaseConfig]:
    """Return the Phase 1 warm-up and Phase 2 full-training configs."""
    return (
        PhaseConfig(
            name="warmup",
            wall_budget_s=float(WARMUP_SECONDS),
            group_size_n=GROUP_SIZE_N,
            thinkprm_weight=0.0,
            reflection_weight=1.0,
            diversity_penalty_enabled=True,
            proxy_reuse_enabled=False,
        ),
        PhaseConfig(
            name="full",
            wall_budget_s=float(TRAINING_SECONDS),
            group_size_n=GROUP_SIZE_N,
            thinkprm_weight=1.0,
            reflection_weight=REFLECTION_WEIGHT_FULL,
            diversity_penalty_enabled=True,
            proxy_reuse_enabled=True,
        ),
    )


def combine_phase_rewards(
    phase: PhaseConfig,
    thinkprm_score: float,
    reflection_reward: float,
) -> float:
    """Return one completion's total reward under a phase schedule."""
    return float(
        round(
            phase.thinkprm_weight * float(thinkprm_score)
            + phase.reflection_weight * float(reflection_reward),
            12,
        )
    )


def combine_phase_reward_groups(
    phase: PhaseConfig,
    thinkprm_scores: list[float],
    reflection_rewards: list[float],
) -> list[float]:
    """Combine aligned ThinkPRM and reflection rewards for one GRPO group."""
    if len(thinkprm_scores) != len(reflection_rewards):
        raise ValueError(
            f"reward group length mismatch: {len(thinkprm_scores)} vs {len(reflection_rewards)}"
        )
    return [
        combine_phase_rewards(phase, t, r)
        for t, r in zip(thinkprm_scores, reflection_rewards, strict=True)
    ]


def improvement_vs_exp1129(
    improvement_over_baseline: float,
    *,
    exp1129_improvement: float = EXP1129_IMPROVEMENT,
) -> float:
    """Return Exp 1159 improvement minus Exp 1129's +0.0851 baseline."""
    return float(round(float(improvement_over_baseline) - float(exp1129_improvement), 12))


def derive_structural_warmup_verdict(
    dualgpu_used: bool,
    improvement_over_baseline: float,
    *,
    exp1129_improvement: float = EXP1129_IMPROVEMENT,
) -> str:
    """Map Exp 1159 outcomes to the canonical honest-verdict labels."""
    improvement = float(improvement_over_baseline)
    if not dualgpu_used:
        return "blocked_no_dualgpu"
    if improvement > float(exp1129_improvement):
        return "structural_warmup_above_0851"
    if improvement > 0.0:
        return "positive_below_exp1129"
    if improvement < 0.0:
        return "negative_regression"
    return "neutral"


def build_structural_warmup_artifact_fields(
    *,
    cuda_device_count: int,
    dualgpu_used: bool,
    training_wall_budget_hit: bool,
    advantage_stdev_warmup: float,
    advantage_stdev_full: float,
    n_eval_questions: int,
    baseline_fraction_correct: float,
    trained_fraction_correct: float,
    improvement_over_baseline: float,
) -> dict[str, Any]:
    """Return the REQ-LEARN-1159 required artifact fields."""
    improvement = float(improvement_over_baseline)
    return {
        "dualgpu_used": bool(dualgpu_used),
        "cuda_device_count": int(cuda_device_count),
        "warmup_seconds": WARMUP_SECONDS,
        "training_seconds": TRAINING_SECONDS,
        "training_wall_budget_hit": bool(training_wall_budget_hit),
        "advantage_stdev_warmup": float(advantage_stdev_warmup),
        "advantage_stdev_full": float(advantage_stdev_full),
        "n_eval_questions": int(n_eval_questions),
        "baseline_fraction_correct": float(baseline_fraction_correct),
        "trained_fraction_correct": float(trained_fraction_correct),
        "improvement_over_baseline": improvement,
        "improvement_vs_exp1129": improvement_vs_exp1129(improvement),
        "reflection_weight": REFLECTION_WEIGHT_FULL,
        "structural_warmup_used": True,
        "grpo_v4_honest_result": True,
        "honest_verdict": derive_structural_warmup_verdict(
            bool(dualgpu_used),
            improvement,
        ),
    }
