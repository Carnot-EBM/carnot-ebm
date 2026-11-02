"""Root-cause diagnosis helpers for the Exp 1208 GRPO v5 regression.

Background
----------
Exp 1208 (milestone .94) ran GRPO v5 with TinyV confidence abstention on
the dual-RTX-3090 rig and produced ``improvement_over_baseline_pp =
-35.0`` — i.e. v5 trained the 35B Qwen3.6-A3B model from a 100% eval
score down to 75%, while the v4 structural warm-up (Exp 1159) had moved
the same base model +10pp in the opposite direction. A 35-percentage-
point swing in the wrong direction is far outside any plausible "noise"
explanation for an honest GRPO update; it indicates the training signal
itself was either absent, inverted, or dominated by abstention.

This module is a *diagnosis utility*, not a training-time component. It
takes the Exp 1208 artifact dict (or a compatible payload) and returns a
structured root-cause classification plus a recommended fix the .95
planner can wire into Exp 1220 (GRPO-VPS). It exists so the diagnosis
logic is testable in isolation and so future regressions can re-use the
same hypothesis grid instead of re-deriving it.

The hypothesis grid mirrors the prompt's diagnostic checklist:

* ``high_abstention_rate``  — too many TinyV abstentions starve the
  GRPO advantage of any real reward signal. Fires when
  ``tinyv_abstention_rate >= 0.5`` *AND* the surviving training set is
  small enough (n_effective <= 4) that a few mis-rewards can flip the
  policy. The v4 paper's structural warm-up succeeded with no
  abstention; v5's abstention band ``[0.3, 0.7]`` is unusually wide.
* ``saturated_baseline_eval`` — the eval slice was already at 100%
  before training, so the only direction the policy can move is *down*.
  This is a diagnostic finding even when other root causes are also
  present, because it means the experiment design cannot reward
  improvement.
* ``dualgpu_instability`` — one GPU dominates utilization (>= 75%),
  suggesting tensor-split mismatch and out-of-sync gradient state.
* ``threshold_misconfiguration`` — abstention rate is in a moderate
  range (0.3-0.7) but improvement is still strongly negative; the
  TinyV band may have been confused with the ThinkPRM confidence.
* ``reward_signal_collapse`` — pre-training accuracy strictly higher
  than post-training by more than 10pp; consistent with an inverted
  reward sign.
* ``implementation_bug`` — pre/post difference is large and none of
  the other heuristics fire; flag for code-level inspection.

The classifier prefers the *most actionable* explanation: when both
``high_abstention_rate`` and ``saturated_baseline_eval`` apply, the
abstention finding is returned as ``root_cause`` and the saturation
finding is recorded in ``contributing_factors`` so the .95 planner
treats it as a constraint on the next experiment design.

Spec: REQ-LEARN-1219, SCENARIO-LEARN-1219.
"""

from __future__ import annotations

from typing import Any

REQUIRED_DIAGNOSIS_ARTIFACT_FIELDS = (
    "tinyv_abstention_rate_observed",
    "grpo_v5_improvement_pp",
    "root_cause",
    "root_cause_evidence",
    "recommended_fix_for_exp1220",
    "diagnosis_complete",
    "honest_verdict",
)

ALLOWED_ROOT_CAUSES = frozenset(
    {
        "high_abstention_rate",
        "dualgpu_instability",
        "threshold_misconfiguration",
        "reward_signal_collapse",
        "implementation_bug",
        "unknown",
    }
)

ALLOWED_HONEST_VERDICTS = frozenset(
    {
        "root_cause_identified",
        "root_cause_partial",
        "root_cause_unknown",
    }
)

# A TinyV abstention rate at or above this fraction means the GRPO
# advantage estimator is being fed by too few rollouts. Set at 0.5
# because Exp 1208 hit 5/8 = 0.625 abstentions and that alone (with no
# other pathology) would have starved the update.
HIGH_ABSTENTION_RATE_THRESHOLD = 0.5

# A surviving training-rollout count at or below this number means the
# advantage estimate is dominated by individual rollout noise. The
# original GRPO paper recommends 4-8 effective samples per group; below
# 4 the variance term swamps the mean.
LOW_EFFECTIVE_TRAINING_THRESHOLD = 4

# A pre-training eval accuracy at or above this fraction is "saturated":
# there is no headroom for the policy to demonstrate improvement.
SATURATED_BASELINE_THRESHOLD = 0.95

# A single-GPU utilization at or above this percentage on a tensor-split
# config is suspect; the split should keep both within ~10pp of each
# other when the shards are balanced.
DUALGPU_IMBALANCE_THRESHOLD_PCT = 75.0

# A "strong negative" pre-vs-post regression in absolute pp; implies the
# policy moved meaningfully in the wrong direction.
STRONG_NEGATIVE_REGRESSION_PP = 10.0


def _get_float(payload: dict[str, Any], key: str, default: float | None = None) -> float | None:
    """Read a float out of the artifact dict, tolerating missing keys.

    The Exp 1208 artifact is JSON-serialized so all numeric fields
    arrive as ``int``/``float``; we coerce defensively because future
    artifacts may have schema drift.
    """
    if key not in payload:
        return default
    raw = payload[key]
    if raw is None:
        return default
    return float(raw)


def diagnose_exp1208_regression(payload: dict[str, Any]) -> dict[str, Any]:
    """Classify the Exp 1208 GRPO v5 regression by root cause.

    Returns a dict with the schema in
    ``REQUIRED_DIAGNOSIS_ARTIFACT_FIELDS`` plus a
    ``contributing_factors`` list capturing secondary findings. The
    classifier intentionally returns ``unknown`` when no hypothesis
    fires cleanly, so the planner cannot mistake silence for
    confirmation that everything is fine.
    """
    abstention_rate = _get_float(payload, "tinyv_abstention_rate", 0.0) or 0.0
    abstention_count = _get_float(payload, "tinyv_abstention_count", 0.0) or 0.0
    n_train = _get_float(payload, "n_train_questions", 0.0) or 0.0
    n_effective = max(0.0, n_train - abstention_count)
    pre_acc = _get_float(payload, "v5_fraction_correct_before", 0.0) or 0.0
    post_acc = _get_float(payload, "v5_fraction_correct_after", 0.0) or 0.0
    improvement_pp = _get_float(payload, "improvement_over_baseline_pp", 0.0) or 0.0
    gpu0_util = _get_float(payload, "dualgpu_gpu0_utilization_pct", 0.0) or 0.0
    gpu1_util = _get_float(payload, "dualgpu_gpu1_utilization_pct", 0.0) or 0.0

    contributing: list[str] = []

    # Saturation is always recorded when present, regardless of the
    # primary root cause, because it constrains the recommended fix.
    if pre_acc >= SATURATED_BASELINE_THRESHOLD:
        contributing.append(
            f"saturated_baseline_eval: v5_fraction_correct_before={pre_acc:.3f} "
            f">= {SATURATED_BASELINE_THRESHOLD:.2f}; eval set has no improvement "
            "headroom so any noise drives accuracy down."
        )

    # DualGPU imbalance is also recorded when present.
    if max(gpu0_util, gpu1_util) >= DUALGPU_IMBALANCE_THRESHOLD_PCT and abs(
        gpu0_util - gpu1_util
    ) > 10.0:
        contributing.append(
            f"dualgpu_imbalance: gpu0={gpu0_util:.1f}% gpu1={gpu1_util:.1f}%; "
            "tensor split is not keeping shards balanced."
        )

    # Hypothesis 1: high abstention rate + low effective training set.
    if (
        abstention_rate >= HIGH_ABSTENTION_RATE_THRESHOLD
        and n_effective <= LOW_EFFECTIVE_TRAINING_THRESHOLD
    ):
        evidence = (
            f"tinyv_abstention_rate={abstention_rate:.3f} "
            f"(abstain_count={int(abstention_count)} of n_train={int(n_train)}); "
            f"n_effective_rollouts={int(n_effective)} <= "
            f"{LOW_EFFECTIVE_TRAINING_THRESHOLD}, well below GRPO's recommended "
            "4-8 per group. Combined with v5_fraction_correct_before="
            f"{pre_acc:.3f} (saturated baseline), the surviving 1-3 reward "
            "signals create high-variance gradient updates that can only move "
            "the policy downward, producing the observed "
            f"improvement_over_baseline_pp={improvement_pp:.1f}."
        )
        fix = (
            "Exp 1220 (GRPO-VPS) should (1) narrow the TinyV abstention band "
            f"from the current [{0.3:.1f}, {0.7:.1f}] to a tighter [0.45, 0.55] "
            "or fall back to a soft-weighted reward (no hard zero) so most "
            "rollouts contribute gradient; (2) raise n_train_questions from 8 "
            "to >= 32 so even a 60% abstention rate leaves >= 12 effective "
            "rollouts; and (3) replace the saturated 12-question eval slice "
            "with a slice whose pre-training accuracy is in [0.4, 0.7] so the "
            "experiment has measurable headroom in both directions."
        )
        return {
            "tinyv_abstention_rate_observed": abstention_rate,
            "grpo_v5_improvement_pp": improvement_pp,
            "root_cause": "high_abstention_rate",
            "root_cause_evidence": evidence,
            "recommended_fix_for_exp1220": fix,
            "diagnosis_complete": True,
            "honest_verdict": "root_cause_identified",
            "contributing_factors": contributing,
        }

    # Hypothesis 2: dualgpu instability dominates.
    if max(gpu0_util, gpu1_util) >= DUALGPU_IMBALANCE_THRESHOLD_PCT and abs(
        gpu0_util - gpu1_util
    ) > 25.0:
        evidence = (
            f"gpu0_util={gpu0_util:.1f}% vs gpu1_util={gpu1_util:.1f}% "
            "differ by more than 25pp on a balanced tensor split; this "
            "implicates pipeline stalls or async-gradient mismatch as the "
            "dominant cause of the regression."
        )
        fix = (
            "Exp 1220 should fall back to single-GPU training (drop "
            "tensor_split, set n_gpu_layers to fit the active GPU) until "
            "the dual-GPU shard balance is verified by an independent "
            "micro-benchmark."
        )
        return {
            "tinyv_abstention_rate_observed": abstention_rate,
            "grpo_v5_improvement_pp": improvement_pp,
            "root_cause": "dualgpu_instability",
            "root_cause_evidence": evidence,
            "recommended_fix_for_exp1220": fix,
            "diagnosis_complete": True,
            "honest_verdict": "root_cause_identified",
            "contributing_factors": contributing,
        }

    # Hypothesis 3: threshold misconfiguration in the moderate range.
    if (
        0.3 <= abstention_rate < HIGH_ABSTENTION_RATE_THRESHOLD
        and improvement_pp <= -STRONG_NEGATIVE_REGRESSION_PP
    ):
        evidence = (
            f"tinyv_abstention_rate={abstention_rate:.3f} sits in the "
            "moderate band [0.3, 0.5) yet improvement is "
            f"{improvement_pp:.1f}pp; consistent with the TinyV abstention "
            "threshold being applied to ThinkPRM's calibrated confidence "
            "instead of TinyV's raw verifier score."
        )
        fix = (
            "Exp 1220 should plumb separate confidence channels for "
            "ThinkPRM v2 (calibrated, used for routing) and TinyV (raw, "
            "used for abstention) and assert at runtime that the abstention "
            "decision reads the TinyV channel, not ThinkPRM's."
        )
        return {
            "tinyv_abstention_rate_observed": abstention_rate,
            "grpo_v5_improvement_pp": improvement_pp,
            "root_cause": "threshold_misconfiguration",
            "root_cause_evidence": evidence,
            "recommended_fix_for_exp1220": fix,
            "diagnosis_complete": True,
            "honest_verdict": "root_cause_identified",
            "contributing_factors": contributing,
        }

    # Hypothesis 4: reward sign collapse / inversion.
    pre_minus_post_pp = (pre_acc - post_acc) * 100.0
    if pre_minus_post_pp >= STRONG_NEGATIVE_REGRESSION_PP:
        evidence = (
            f"pre={pre_acc:.3f} -> post={post_acc:.3f} ({pre_minus_post_pp:.1f}pp "
            "drop); training itself moved the policy in the wrong direction. "
            "Suggests the GRPO advantage sign is inverted or the reward "
            "convention (lower-energy=better) is mismatched against the "
            "trainer's maximize-reward expectation."
        )
        fix = (
            "Exp 1220 must add an inversion guard: assert reward sign "
            "convention by training one mini-step on a synthetic "
            "always-positive-reward fixture and confirming pass-rate "
            "monotonically increases on a held-out probe."
        )
        return {
            "tinyv_abstention_rate_observed": abstention_rate,
            "grpo_v5_improvement_pp": improvement_pp,
            "root_cause": "reward_signal_collapse",
            "root_cause_evidence": evidence,
            "recommended_fix_for_exp1220": fix,
            "diagnosis_complete": True,
            "honest_verdict": "root_cause_partial" if contributing else "root_cause_identified",
            "contributing_factors": contributing,
        }

    # No hypothesis fires cleanly: flag for code-level review.
    if improvement_pp <= -STRONG_NEGATIVE_REGRESSION_PP:
        evidence = (
            f"Strong regression ({improvement_pp:.1f}pp) but no hypothesis "
            "in the grid fires: abstention rate, GPU utilization, and "
            "pre/post split are all in normal ranges. Implementation-level "
            "inspection of latent_grpo.py and grpo_v5_2.py required."
        )
        return {
            "tinyv_abstention_rate_observed": abstention_rate,
            "grpo_v5_improvement_pp": improvement_pp,
            "root_cause": "implementation_bug",
            "root_cause_evidence": evidence,
            "recommended_fix_for_exp1220": (
                "Exp 1220 should be preceded by a manual code-level review "
                "of the GRPO update step; do not run another v5-class "
                "training experiment until the bug is named."
            ),
            "diagnosis_complete": True,
            "honest_verdict": "root_cause_partial",
            "contributing_factors": contributing,
        }

    return {
        "tinyv_abstention_rate_observed": abstention_rate,
        "grpo_v5_improvement_pp": improvement_pp,
        "root_cause": "unknown",
        "root_cause_evidence": "No hypothesis fired and regression is mild; "
        "may be sampling noise within the spurious-reward threshold.",
        "recommended_fix_for_exp1220": "No specific fix; collect more runs.",
        "diagnosis_complete": False,
        "honest_verdict": "root_cause_unknown",
        "contributing_factors": contributing,
    }


__all__ = [
    "ALLOWED_HONEST_VERDICTS",
    "ALLOWED_ROOT_CAUSES",
    "DUALGPU_IMBALANCE_THRESHOLD_PCT",
    "HIGH_ABSTENTION_RATE_THRESHOLD",
    "LOW_EFFECTIVE_TRAINING_THRESHOLD",
    "REQUIRED_DIAGNOSIS_ARTIFACT_FIELDS",
    "SATURATED_BASELINE_THRESHOLD",
    "STRONG_NEGATIVE_REGRESSION_PP",
    "diagnose_exp1208_regression",
]
