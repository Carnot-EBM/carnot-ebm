"""GRPO-VPS full training-run helpers (Exp 1220).

This module is the small, well-tested seam between Exp 1209's step-level
reward primitives in :mod:`carnot.training.grpo_vps` and the live training
script :file:`scripts/experiment_1220_grpo_vps_full_training.py`. The
script orchestrates a Phase A reflection-only warm-up followed by a
Phase B mix in which the VPS step-level reward (per arXiv 2604.20659)
is blended with the structural reflection reward and a small correctness
bonus.

The helpers below intentionally do NOT touch the LLM, llama_cpp, or any
GPU runtime — they are pure-Python so the unit tests can cover the
reward-shaping logic deterministically without spinning up a 35B model.
The script wraps these helpers around the live rollout loop.

Why these helpers exist as a separate module rather than inside
:mod:`grpo_vps`:
    Exp 1219's regression diagnosis (root cause = high abstention rate
    + saturated baseline + n_train too small) prescribed three fixes
    that are *not* part of the underlying step-reward signal — they are
    rollout-pool adjustments. Keeping them in a sibling module makes
    the diff between "the verifier-derived reward" (REQ-LEARN-1209) and
    "the Exp 1220 training-run shape" (REQ-LEARN-1220) easy to audit.

Spec: REQ-LEARN-1220, SCENARIO-LEARN-1222, SCENARIO-LEARN-1223,
      SCENARIO-LEARN-1224, SCENARIO-LEARN-1225.
"""

from __future__ import annotations

from typing import Sequence

# Hard-coded v4 baseline improvement (Exp 1159's headline result).
# REQ-LEARN-1220-6 forbids this from being relaxed by future v4 gains
# without an explicit code change.
V4_BASELINE_IMPROVEMENT_PP: float = 10.0

# Required artifact field names (REQ-LEARN-1220-5).  The experiment
# script asserts these at exit so a partial artifact never silently
# ships.
REQUIRED_GRPO_VPS_TRAINING_ARTIFACT_FIELDS: tuple[str, ...] = (
    "llama_cpp_gpu_offload",
    "cuda_device_count",
    "model_used",
    "exp1219_fix_applied",
    "training_completed",
    "n_training_questions",
    "n_eval_questions",
    "grpo_vps_fraction_correct_before",
    "grpo_vps_fraction_correct_after",
    "grpo_vps_improvement_pp",
    "v4_baseline_improvement_pp",
    "beats_v4_floor",
    "grpo_vps_training_completed",
    "honest_verdict",
)

ALLOWED_GRPO_VPS_VERDICTS: frozenset[str] = frozenset(
    {
        "vps_training_beats_v4",
        "vps_training_matches_v4",
        "vps_training_below_v4",
        "training_wall_hit",
        "blocked_no_gpu",
    }
)


def compute_vps_aggregate_reward(response: str, decay: float = 0.9) -> float:
    """Compute the aggregate VPS reward for one CoT response.

    Splits the response into reasoning steps using the existing
    SymCodeVerifier segmenter (the same one Exp 1209 uses), computes
    each step's reward via :func:`carnot.training.grpo_vps.segment_reward`,
    and returns the geometrically-decayed sum
    ``sum(decay**k * step_reward[k] for k in range(n_steps))``.

    Earlier steps weigh more heavily because an error in the first
    step corrupts every subsequent step — a wrong premise propagates
    forward, but a wrong final summation does not propagate backward.

    Args:
        response: Full chain-of-thought response text.
        decay:    Geometric decay factor in [0, 1].  ``1.0`` weights all
                  steps equally; ``0.0`` keeps only the first step.

    Returns:
        Aggregate float reward.  Returns ``0.0`` when the segmenter
        produces zero steps (empty or whitespace-only response).

    Spec: REQ-LEARN-1220-1, SCENARIO-LEARN-1222.
    """
    from carnot.training.grpo_vps import compute_step_rewards_for_response  # noqa: PLC0415

    per_step = compute_step_rewards_for_response(response)
    if not per_step:
        return 0.0
    return float(sum(r * (decay**k) for k, r in enumerate(per_step)))


def soft_confidence_weight(
    rewards: Sequence[float],
    confidences: Sequence[float],
) -> list[float]:
    """Soft-weight rewards by verifier confidence (Exp 1219 fix #1).

    Exp 1208's TinyV-style abstention zeroed out any reward whose
    verifier confidence fell inside the [0.3, 0.7] uncertain band.
    Exp 1219 diagnosed that this threw away ~62.5% of the rollouts and
    starved GRPO's group-size requirement, producing the -35pp
    regression vs the v4 floor. The recommended fix is to multiply the
    reward by the confidence rather than zeroing it: high-confidence
    rewards pass through nearly unchanged, low-confidence rewards are
    attenuated, but no rollout is fully discarded.

    Args:
        rewards:     Per-rollout reward values.
        confidences: Per-rollout verifier confidences in [0.0, 1.0].

    Returns:
        ``[rewards[i] * confidences[i] for i in range(len(rewards))]``.

    Raises:
        ValueError: when ``rewards`` and ``confidences`` differ in
            length — silent length-mismatch was the kind of bug that
            slipped past Exp 1208's abstention path.

    Spec: REQ-LEARN-1220-2, SCENARIO-LEARN-1223.
    """
    if len(rewards) != len(confidences):
        raise ValueError(
            f"rewards/confidences length mismatch: "
            f"{len(rewards)} vs {len(confidences)}"
        )
    return [float(r) * float(c) for r, c in zip(rewards, confidences, strict=True)]


def mix_phase_b_reward(
    r_vps: float,
    r_reflect: float,
    r_correctness: float,
    *,
    w_vps: float = 0.5,
    w_reflect: float = 0.3,
    w_correctness: float = 0.2,
) -> float:
    """Combine three reward channels into one Phase-B training signal.

    Phase B of the Exp 1220 schedule mixes the VPS step-level reward
    with the structural reflection reward (preserved from Exp 1159's
    warm-up) and a small correctness bonus that pulls rollouts toward
    the gold answer when both verifiers agree it's right. The default
    weights match the task specification (0.5 / 0.3 / 0.2).

    Args:
        r_vps:           Aggregate VPS reward from
                         :func:`compute_vps_aggregate_reward`.
        r_reflect:       Energy-drop reflection reward (E_before - E_after).
        r_correctness:   Binary or soft correctness signal in [0, 1].
        w_vps:           Weight on the VPS channel.
        w_reflect:       Weight on the reflection channel.
        w_correctness:   Weight on the correctness channel.

    Returns:
        Convex combination ``w_vps * r_vps + w_reflect * r_reflect +
        w_correctness * r_correctness``.

    Raises:
        ValueError: when the three weights do not sum to ``1.0`` within
            ``1e-6`` — catching a future tweak that accidentally
            unbalances the mix.

    Spec: REQ-LEARN-1220-3, SCENARIO-LEARN-1224.
    """
    weight_sum = w_vps + w_reflect + w_correctness
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"phase-B weights must sum to 1.0, got {weight_sum} "
            f"(w_vps={w_vps}, w_reflect={w_reflect}, w_correctness={w_correctness})"
        )
    return float(
        w_vps * r_vps + w_reflect * r_reflect + w_correctness * r_correctness
    )


def derive_grpo_vps_honest_verdict(
    improvement_pp: float,
    *,
    training_completed: bool,
    prereq_ok: bool,
) -> str:
    """Map an Exp 1220 outcome to the canonical verdict string.

    The decision tree is intentionally simple so the resulting verdict
    is auditable from the artifact alone:

    1. If the prereq gate (GPU offload + >=2 CUDA devices) failed,
       the verdict is ``blocked_no_gpu`` regardless of any other
       field — there cannot have been a meaningful training run.
    2. If the prereq gate passed but training did not complete (wall
       budget exhausted before the eval cycle), the verdict is
       ``training_wall_hit``.
    3. Otherwise the verdict is determined by ``improvement_pp`` vs
       the v4 floor of +10pp:
         - ``> 10.0`` -> ``vps_training_beats_v4``
         - ``[0.0, 10.0]`` -> ``vps_training_matches_v4``
         - ``< 0.0`` -> ``vps_training_below_v4``

    Args:
        improvement_pp:     Eval improvement in percentage points
                            (``after - before`` * 100).
        training_completed: Whether the full training cycle finished
                            inside the wall budget.
        prereq_ok:          Whether the pre-flight GPU + GGUF gates
                            both passed.

    Returns:
        One of the five tokens in :data:`ALLOWED_GRPO_VPS_VERDICTS`.

    Spec: REQ-LEARN-1220-4, SCENARIO-LEARN-1225.
    """
    if not prereq_ok:
        return "blocked_no_gpu"
    if not training_completed:
        return "training_wall_hit"
    if improvement_pp > V4_BASELINE_IMPROVEMENT_PP:
        return "vps_training_beats_v4"
    if improvement_pp >= 0.0:
        return "vps_training_matches_v4"
    return "vps_training_below_v4"


def build_grpo_vps_training_artifact_fields(
    *,
    llama_cpp_gpu_offload: bool,
    cuda_device_count: int,
    model_used: str,
    exp1219_fix_applied: str,
    training_completed: bool,
    n_training_questions: int,
    n_eval_questions: int,
    grpo_vps_fraction_correct_before: float,
    grpo_vps_fraction_correct_after: float,
) -> dict[str, object]:
    """Build the full required-field block for the Exp 1220 artifact.

    Centralises the field-name set and the derived-metric arithmetic
    so the experiment script and tests cannot drift apart. Returns a
    plain ``dict`` (not a dataclass) because the artifact is JSON-
    serialised at exit and a dict trips zero JSON-encoding edge cases.

    The derived fields are:
        - ``grpo_vps_improvement_pp`` = 100 *
          ``(after - before)``.
        - ``v4_baseline_improvement_pp`` = ``V4_BASELINE_IMPROVEMENT_PP``
          (10.0).
        - ``beats_v4_floor`` = ``grpo_vps_improvement_pp >
          V4_BASELINE_IMPROVEMENT_PP``.
        - ``grpo_vps_training_completed`` mirrors ``training_completed``
          for downstream reconcilers that prefer the verbose name.
        - ``honest_verdict`` from :func:`derive_grpo_vps_honest_verdict`.

    Args:
        llama_cpp_gpu_offload: Result of
            ``llama_cpp.llama_supports_gpu_offload()``.
        cuda_device_count:     Result of ``torch.cuda.device_count()``.
        model_used:            HF id or path of the GGUF used for live
                               inference, or a short fallback string
                               when the prereq gate failed.
        exp1219_fix_applied:   Free-form description of which fixes
                               from Exp 1219 are active in this run.
        training_completed:    True iff the full Phase A + Phase B +
                               eval cycle finished inside the wall
                               budget.
        n_training_questions:  Configured training-set size (target).
        n_eval_questions:      Configured holdout-eval size (target).
        grpo_vps_fraction_correct_before: Pre-training pass-rate.
        grpo_vps_fraction_correct_after:  Post-training pass-rate.

    Returns:
        Dict containing every entry in
        :data:`REQUIRED_GRPO_VPS_TRAINING_ARTIFACT_FIELDS`.

    Spec: REQ-LEARN-1220-5, REQ-LEARN-1220-6.
    """
    improvement_pp = round(
        100.0 * (grpo_vps_fraction_correct_after - grpo_vps_fraction_correct_before),
        4,
    )
    prereq_ok = bool(llama_cpp_gpu_offload) and int(cuda_device_count) >= 2
    verdict = derive_grpo_vps_honest_verdict(
        improvement_pp,
        training_completed=bool(training_completed),
        prereq_ok=prereq_ok,
    )
    return {
        "llama_cpp_gpu_offload": bool(llama_cpp_gpu_offload),
        "cuda_device_count": int(cuda_device_count),
        "model_used": str(model_used),
        "exp1219_fix_applied": str(exp1219_fix_applied),
        "training_completed": bool(training_completed),
        "n_training_questions": int(n_training_questions),
        "n_eval_questions": int(n_eval_questions),
        "grpo_vps_fraction_correct_before": float(grpo_vps_fraction_correct_before),
        "grpo_vps_fraction_correct_after": float(grpo_vps_fraction_correct_after),
        "grpo_vps_improvement_pp": float(improvement_pp),
        "v4_baseline_improvement_pp": float(V4_BASELINE_IMPROVEMENT_PP),
        "beats_v4_floor": bool(improvement_pp > V4_BASELINE_IMPROVEMENT_PP),
        "grpo_vps_training_completed": bool(training_completed),
        "honest_verdict": verdict,
    }
