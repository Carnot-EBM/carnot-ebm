"""GRPO-VPS: step-level process supervision using Carnot's step-level verifiers.

arXiv 2604.20659 (GRPO-VPS) extends GRPO with per-step rewards computed as the
change in the model's belief in the correct answer at each step boundary.  This
module implements Carnot's variant: instead of belief-change rewards, it derives
per-step rewards from two orthogonal structural verifiers that the pipeline
already computes at inference time.

Why this approach rather than belief-change as in the paper:
    Belief-change rewards require running the model forward twice per step
    boundary, which doubles inference cost.  Carnot's verifiers (causal
    reasoning and symbolic arithmetic) are O(1) per step and are independent
    of the model's internal probabilities — they check structural correctness
    directly, which is more robust to reward hacking on the model's own outputs.

Spec: REQ-LEARN-1209, SCENARIO-LEARN-1211, SCENARIO-LEARN-1212,
      SCENARIO-LEARN-1213, SCENARIO-LEARN-1214
"""

from __future__ import annotations

from typing import TYPE_CHECKING


def segment_reward(
    step_text: str,
    step_index: int,
    prior_step: str | None = None,
) -> float:
    """Compute a per-step process supervision reward for one CoT step.

    Combines two orthogonal structural verifiers:
    - CausalReasoningVerifier: checks that this step's numeric premise
      agrees with the previous step's numeric conclusion.
    - Z3MathVerifier: checks that arithmetic equations within this step
      are correct.

    Each verifier returns a violation probability in [0.0, 1.0].  We
    convert these to step quality scores by complementing them (1 - prob),
    then average the two scores.  A reward near 1.0 means both verifiers
    found no problems; near 0.0 means at least one verifier detected a
    clear violation.

    Args:
        step_text:   Text of the current reasoning step.
        step_index:  Zero-based index of this step within the full response.
                     Used for logging but not for the reward computation itself.
        prior_step:  Text of the immediately preceding step, or None for
                     the first step.  Passed to CausalReasoningVerifier.

    Returns:
        Float in [0.0, 1.0].

    Spec: REQ-LEARN-1209-3, SCENARIO-LEARN-1213
    """
    from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier  # noqa: PLC0415
    from carnot.verify.z3_math_verifier import Z3MathVerifier  # noqa: PLC0415

    causal_score = CausalReasoningVerifier().verify_step(step_text, prior_step)
    z3_score = Z3MathVerifier().verify_step(step_text)
    reward = 0.5 * (1.0 - causal_score) + 0.5 * (1.0 - z3_score)
    return float(max(0.0, min(1.0, reward)))


def aggregate_step_rewards(
    per_step_rewards: list[float],
    gamma: float = 0.9,
) -> float:
    """Aggregate per-step rewards into a single scalar using geometric decay.

    Earlier steps are weighted more heavily under the assumption that
    errors early in a chain-of-thought are more fundamental — a wrong
    premise in step 1 corrupts every subsequent step.

    The formula is the discounted sum: sum(r_i * gamma^i for i, r_i in ...)

    Args:
        per_step_rewards: List of per-step reward floats, ordered from
                          first step to last.
        gamma:            Decay factor.  gamma=1.0 gives equal weighting;
                          gamma=0.0 weights only the first step.
                          Defaults to 0.9 (mild front-loading).

    Returns:
        Discounted sum as a float.  Returns 0.0 for an empty list.

    Spec: REQ-LEARN-1209-4, SCENARIO-LEARN-1214
    """
    if not per_step_rewards:
        return 0.0
    return sum(r * (gamma**i) for i, r in enumerate(per_step_rewards))


def compute_step_rewards_for_response(
    response: str,
) -> list[float]:
    """Split a CoT response into steps and compute per-step rewards.

    Convenience wrapper used by the experiment script.  Segments the
    response using CausalReasoningVerifier's underlying SymCodeVerifier
    step segmenter (paragraph/sentence splits) and calls segment_reward
    on each consecutive pair.

    Args:
        response: Full chain-of-thought response text.

    Returns:
        List of per-step rewards, one per segmented step.
    """
    from carnot.pipeline.symcode_verifier import SymCodeVerifier  # noqa: PLC0415

    steps = SymCodeVerifier().segment_steps(response)
    if not steps:
        return []
    rewards = []
    for i, step in enumerate(steps):
        prior = steps[i - 1] if i > 0 else None
        rewards.append(segment_reward(step, i, prior))
    return rewards
