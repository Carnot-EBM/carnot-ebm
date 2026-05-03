"""Repair-grounded reward helpers for Exp 1146 GRPO.

The functions in this module keep the new Exp 1146 reward logic independent of
the live-GPU experiment runner. Unit tests can exercise the reward math and
artifact schema without loading Qwen3.6-35B or invoking llama.cpp.

Spec: REQ-LEARN-1146, SCENARIO-LEARN-1146, SCENARIO-LEARN-1147,
      REQ-LEARN-1173, SCENARIO-LEARN-1173, SCENARIO-LEARN-1174,
      SCENARIO-LEARN-1175
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

REFLECTION_WEIGHT = 0.3
N_REPAIR_STEPS_PER_COMPLETION = 1
ALPHA_T_AT_TRAINING = 0.52
EXP1129_IMPROVEMENT = 0.0851
EXP1159_V4_BASELINE = 0.10
FN_ABSTAIN_THRESH_LOW = 0.3
FN_ABSTAIN_THRESH_HIGH = 0.7
V5_WARMUP_SECONDS = 300
V5_TRAINING_SECONDS = 900

REQUIRED_TINYV_ARTIFACT_FIELDS = (
    "improvement_over_baseline",
    "v4_baseline",
    "fn_abstention_rate",
    "fn_threshold_tuned",
    "training_completed",
    "dualgpu_confirmed",
    "grpo_v5_honest_result",
    "honest_verdict",
)


@dataclass(frozen=True)
class ReflectionRewardResult:
    """Result of scoring one completion with the one-step repair reward."""

    energy_before: float
    energy_after: float
    reward: float
    repaired_response: str
    repair_attempted: bool
    clipped: bool


@dataclass(frozen=True)
class TinyVRewardResult:
    """Reward result after optional TinyV uncertainty abstention."""

    emitted_reward: float
    raw_reward: float
    thinkprm_confidence: float
    reflection_reward: float
    abstained: bool


@dataclass(frozen=True)
class TinyVRewardGroupResult:
    """Aligned GRPO group rewards plus TinyV abstention diagnostics."""

    rewards: list[float]
    raw_rewards: list[float]
    abstained: list[bool]
    fn_abstention_rate: float


def _clip_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def normalized_reflection_reward(energy_before: float, energy_after: float) -> float:
    """Return clipped ``(E_before - E_after) / E_before`` or 0 when undefined."""
    if energy_before <= 0.0:
        return 0.0
    return _clip_unit((float(energy_before) - float(energy_after)) / float(energy_before))


def combine_rewards(
    thinkprm_score: float,
    reflection_reward: float,
    *,
    reflection_weight: float = REFLECTION_WEIGHT,
) -> float:
    """Return ``r_thinkprm + reflection_weight * r_reflect``."""
    return float(
        round(float(thinkprm_score) + float(reflection_weight) * float(reflection_reward), 12)
    )


def combine_reward_groups(
    thinkprm_scores: list[float],
    reflection_rewards: list[float],
    *,
    reflection_weight: float = REFLECTION_WEIGHT,
) -> list[float]:
    """Combine aligned ThinkPRM and reflection rewards for a GRPO group."""
    if len(thinkprm_scores) != len(reflection_rewards):
        raise ValueError(
            f"reward group length mismatch: {len(thinkprm_scores)} vs {len(reflection_rewards)}"
        )
    return [
        combine_rewards(t, r, reflection_weight=reflection_weight)
        for t, r in zip(thinkprm_scores, reflection_rewards, strict=True)
    ]


def _validate_tinyv_thresholds(low: float, high: float) -> tuple[float, float]:
    low_f = float(low)
    high_f = float(high)
    if not 0.0 <= low_f <= high_f <= 1.0:
        raise ValueError("TinyV thresholds must satisfy 0.0 <= low <= high <= 1.0")
    return low_f, high_f


def tinyv_confidence_abstains(
    thinkprm_confidence: float,
    *,
    low: float = FN_ABSTAIN_THRESH_LOW,
    high: float = FN_ABSTAIN_THRESH_HIGH,
) -> bool:
    """Return True when ThinkPRM confidence falls in the TinyV uncertainty band."""
    low_f, high_f = _validate_tinyv_thresholds(low, high)
    confidence = float(thinkprm_confidence)
    return low_f <= confidence <= high_f


def combine_rewards_with_tinyv_abstention(
    thinkprm_score: float,
    reflection_reward: float,
    *,
    reflection_weight: float = REFLECTION_WEIGHT,
    fn_abstain_thresh_low: float = FN_ABSTAIN_THRESH_LOW,
    fn_abstain_thresh_high: float = FN_ABSTAIN_THRESH_HIGH,
    abstention_enabled: bool = True,
) -> TinyVRewardResult:
    """Combine rewards, then zero the emitted reward in TinyV's uncertainty band."""
    raw_reward = combine_rewards(
        thinkprm_score,
        reflection_reward,
        reflection_weight=reflection_weight,
    )
    abstained = bool(
        abstention_enabled
        and tinyv_confidence_abstains(
            thinkprm_score,
            low=fn_abstain_thresh_low,
            high=fn_abstain_thresh_high,
        )
    )
    return TinyVRewardResult(
        emitted_reward=0.0 if abstained else raw_reward,
        raw_reward=raw_reward,
        thinkprm_confidence=float(thinkprm_score),
        reflection_reward=float(reflection_reward),
        abstained=abstained,
    )


def combine_reward_groups_with_tinyv_abstention(
    thinkprm_scores: list[float],
    reflection_rewards: list[float],
    *,
    reflection_weight: float = REFLECTION_WEIGHT,
    fn_abstain_thresh_low: float = FN_ABSTAIN_THRESH_LOW,
    fn_abstain_thresh_high: float = FN_ABSTAIN_THRESH_HIGH,
    abstention_enabled: bool = True,
) -> TinyVRewardGroupResult:
    """Combine an aligned GRPO group and report the TinyV abstention rate."""
    if len(thinkprm_scores) != len(reflection_rewards):
        raise ValueError(
            f"reward group length mismatch: {len(thinkprm_scores)} vs {len(reflection_rewards)}"
        )
    results = [
        combine_rewards_with_tinyv_abstention(
            t,
            r,
            reflection_weight=reflection_weight,
            fn_abstain_thresh_low=fn_abstain_thresh_low,
            fn_abstain_thresh_high=fn_abstain_thresh_high,
            abstention_enabled=abstention_enabled,
        )
        for t, r in zip(thinkprm_scores, reflection_rewards, strict=True)
    ]
    abstained = [result.abstained for result in results]
    return TinyVRewardGroupResult(
        rewards=[result.emitted_reward for result in results],
        raw_rewards=[result.raw_reward for result in results],
        abstained=abstained,
        fn_abstention_rate=(sum(1 for flag in abstained if flag) / len(abstained))
        if abstained
        else 0.0,
    )


def build_repair_prompt(question: str, response: str, feedback: str) -> str:
    """Build the same one-step verifier-feedback prompt shape as verify_repair."""
    return (
        f"Question: {question}\n\n"
        f"Your previous answer:\n{response}\n\n"
        f"The following issues were found:\n{feedback}\n\n"
        f"Please provide a corrected answer that fixes these issues."
    )


class ReflectionRewardEvaluator:
    """Compute the one-step repair reward through a Carnot verify-like pipeline."""

    def __init__(
        self,
        pipeline: Any,
        repair_generate_fn: Callable[[str], str],
        *,
        domain: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.repair_generate_fn = repair_generate_fn
        self.domain = domain

    def score(self, question: str, response: str) -> ReflectionRewardResult:
        """Verify, repair once if needed, re-verify, and return ``r_reflect``."""
        before = self.pipeline.verify(question, response, self.domain)
        energy_before = float(getattr(before, "energy", 0.0))
        violations = list(getattr(before, "violations", []) or [])
        verified = bool(getattr(before, "verified", False))

        if verified or not violations:
            return ReflectionRewardResult(
                energy_before=energy_before,
                energy_after=energy_before,
                reward=0.0,
                repaired_response=response,
                repair_attempted=False,
                clipped=False,
            )

        feedback = str(self.pipeline._format_violations(violations))
        repaired = self.repair_generate_fn(build_repair_prompt(question, response, feedback))
        after = self.pipeline.verify(question, repaired, self.domain)
        energy_after = float(getattr(after, "energy", 0.0))
        raw_reward = 0.0 if energy_before <= 0.0 else (energy_before - energy_after) / energy_before
        reward = normalized_reflection_reward(energy_before, energy_after)
        return ReflectionRewardResult(
            energy_before=energy_before,
            energy_after=energy_after,
            reward=reward,
            repaired_response=repaired,
            repair_attempted=True,
            clipped=raw_reward != reward,
        )


def derive_reflection_honest_verdict(
    dualgpu_used: bool,
    improvement_over_baseline: float,
    *,
    exp1129_improvement: float = EXP1129_IMPROVEMENT,
) -> str:
    """Map Exp 1146 outcomes to the canonical honest-verdict labels."""
    improvement = float(improvement_over_baseline)
    if not dualgpu_used:
        return "blocked_no_dualgpu"
    if improvement > float(exp1129_improvement):
        return "reflection_positive_above_0851"
    if improvement > 0.0:
        return "positive_below_exp1129"
    if improvement < 0.0:
        return "negative_regression"
    return "neutral"


def build_reflection_artifact_fields(
    *,
    cuda_device_count: int,
    dualgpu_used: bool,
    n_training_questions: int,
    training_seconds: float,
    training_wall_budget_hit: bool,
    advantage_stdev: float,
    n_eval_questions: int,
    baseline_fraction_correct: float,
    trained_fraction_correct: float,
    improvement_over_baseline: float,
    honest_verdict: str | None = None,
) -> dict[str, Any]:
    """Return the required Exp 1146 artifact fields."""
    verdict = honest_verdict or derive_reflection_honest_verdict(
        dualgpu_used,
        improvement_over_baseline,
    )
    return {
        "dualgpu_used": bool(dualgpu_used),
        "cuda_device_count": int(cuda_device_count),
        "n_training_questions": int(n_training_questions),
        "training_seconds": float(training_seconds),
        "training_wall_budget_hit": bool(training_wall_budget_hit),
        "advantage_stdev": float(advantage_stdev),
        "reflection_weight": REFLECTION_WEIGHT,
        "reflection_reward_integrated": True,
        "n_repair_steps_per_completion": N_REPAIR_STEPS_PER_COMPLETION,
        "n_eval_questions": int(n_eval_questions),
        "baseline_fraction_correct": float(baseline_fraction_correct),
        "trained_fraction_correct": float(trained_fraction_correct),
        "improvement_over_baseline": float(improvement_over_baseline),
        "alpha_t_at_training": ALPHA_T_AT_TRAINING,
        "fr11_self_learning_signal_used": True,
        "grpo_reflection_honest_result": True,
        "honest_verdict": verdict,
    }


def derive_tinyv_honest_verdict(
    training_completed: bool,
    improvement_over_baseline: float,
    *,
    v4_baseline: float = EXP1159_V4_BASELINE,
) -> str:
    """Map Exp 1173 outcomes to the canonical TinyV verdict labels."""
    if not training_completed:
        return "training_wall_hit"
    improvement = float(improvement_over_baseline)
    baseline = float(v4_baseline)
    if improvement > baseline:
        return "tinyv_improves_over_v4"
    if improvement < baseline:
        return "tinyv_degrades_v4"
    return "tinyv_tied_with_v4"


def build_tinyv_artifact_fields(
    *,
    cuda_device_count: int,
    dualgpu_confirmed: bool,
    training_completed: bool,
    training_wall_budget_hit: bool,
    advantage_stdev_warmup: float,
    advantage_stdev_full: float,
    n_eval_questions: int,
    baseline_fraction_correct: float,
    trained_fraction_correct: float,
    improvement_over_baseline: float,
    fn_abstention_rate: float,
    fn_threshold_tuned: float = FN_ABSTAIN_THRESH_LOW,
    fn_abstain_thresh_high: float = FN_ABSTAIN_THRESH_HIGH,
    honest_verdict: str | None = None,
) -> dict[str, Any]:
    """Return the REQ-LEARN-1173 required artifact fields."""
    _validate_tinyv_thresholds(fn_threshold_tuned, fn_abstain_thresh_high)
    completed = bool(training_completed)
    verdict = honest_verdict or derive_tinyv_honest_verdict(
        completed,
        improvement_over_baseline,
    )
    return {
        "dualgpu_confirmed": bool(dualgpu_confirmed),
        "dualgpu_used": bool(dualgpu_confirmed),
        "cuda_device_count": int(cuda_device_count),
        "warmup_seconds": V5_WARMUP_SECONDS,
        "training_seconds": V5_TRAINING_SECONDS,
        "training_wall_budget_hit": bool(training_wall_budget_hit),
        "training_completed": completed,
        "advantage_stdev_warmup": float(advantage_stdev_warmup),
        "advantage_stdev_full": float(advantage_stdev_full),
        "n_eval_questions": int(n_eval_questions),
        "baseline_fraction_correct": float(baseline_fraction_correct),
        "trained_fraction_correct": float(trained_fraction_correct),
        "improvement_over_baseline": float(improvement_over_baseline),
        "v4_baseline": EXP1159_V4_BASELINE,
        "fn_abstention_rate": float(fn_abstention_rate),
        "fn_threshold_tuned": float(fn_threshold_tuned),
        "fn_abstain_thresh_high": float(fn_abstain_thresh_high),
        "reflection_weight": REFLECTION_WEIGHT,
        "structural_warmup_used": True,
        "grpo_v5_honest_result": bool(completed and dualgpu_confirmed),
        "honest_verdict": verdict,
    }
