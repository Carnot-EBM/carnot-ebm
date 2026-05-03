"""GRPO-v6: FSPO Per-Token Factuality Weighting combined with VPS Step Supervision.

arXiv 2505.24630 (FSPO) introduces per-token advantage weighting where each token
in a reasoning step inherits the factuality score of its parent step.  This is
strictly more granular than VPS (step-level only): if a step is factually shaky,
EVERY token in that step gets a downweighted gradient signal, while tokens in
verified-correct steps receive full signal.

Why this improves on plain VPS:
    VPS assigns one reward scalar per step, but GRPO's policy gradient update
    processes individual tokens.  Without FSPO weighting, a step with reward 0.8
    contributes the same gradient per token regardless of whether the step has 3
    tokens or 300 tokens.  FSPO reweights so that the GRPO loss for each token T
    in step S is multiplied by factuality_score[S], giving shorter factually-
    verified steps disproportionately more influence per-token than long uncertain
    steps.

Implementation contract (REQ-LEARN-1221):
    1. compute_fspo_vps_advantage: builds per-token advantage tensor from step rewards
       and factuality scores.
    2. select_best_completion: picks the completion with the highest sum of token
       advantages (GRPO selection under FSPO weighting).
    3. derive_fspo_honest_verdict: maps fspo_delta_pp onto an honest verdict string.

Spec: REQ-LEARN-1221, REQ-LEARN-1221-1, REQ-LEARN-1221-2, REQ-LEARN-1221-3,
      REQ-LEARN-1221-4, SCENARIO-LEARN-1226, SCENARIO-LEARN-1227, SCENARIO-LEARN-1228
"""

from __future__ import annotations


def compute_fspo_vps_advantage(
    step_rewards: list[float],
    factuality_scores: list[float],
    tokens_per_step: list[int],
) -> list[float]:
    """Compute per-token FSPO-VPS advantages for one completion.

    For each step S the group-normalised step reward is multiplied by the
    factuality score for that step, and the result is broadcast to every
    token in S.

    Group normalisation here means z-score across the steps of THIS completion
    (not across a group of completions — the caller is responsible for
    normalising across the GRPO group before calling select_best_completion).
    If all step rewards are identical (zero variance) the raw rewards are used
    unchanged so we never divide by zero.

    Args:
        step_rewards:      Per-step quality scores in [0, 1], one per step.
        factuality_scores: Per-step factuality scores in [0, 1], same length
                           as step_rewards.  Higher means more factually
                           verified.
        tokens_per_step:   Number of tokens in each step, same length as
                           step_rewards.

    Returns:
        Flat list of per-token advantage floats.  Length equals sum(tokens_per_step).

    Spec: REQ-LEARN-1221-1, SCENARIO-LEARN-1226
    """
    if not step_rewards:
        return []
    if len(step_rewards) != len(factuality_scores) or len(step_rewards) != len(
        tokens_per_step
    ):
        raise ValueError(
            "step_rewards, factuality_scores, and tokens_per_step must have the same length"
        )

    # Group-normalise the step rewards within this completion.
    n = len(step_rewards)
    mean = sum(step_rewards) / n
    variance = sum((r - mean) ** 2 for r in step_rewards) / n
    std = variance**0.5
    if std < 1e-9:
        # All steps identical — use raw rewards (no normalisation possible).
        normalised = list(step_rewards)
    else:
        normalised = [(r - mean) / std for r in step_rewards]

    # Broadcast: each token in step S gets normalised[S] * factuality[S].
    token_advantages: list[float] = []
    for s_idx, n_tokens in enumerate(tokens_per_step):
        advantage = normalised[s_idx] * factuality_scores[s_idx]
        token_advantages.extend([advantage] * n_tokens)
    return token_advantages


def select_best_completion(
    completions: list[str],
    advantages: list[list[float]],
) -> str:
    """Return the completion whose sum of per-token advantages is highest.

    Ties are broken by returning the first maximally-advantaged completion
    in the input list.

    Args:
        completions: Candidate completion strings, one per element.
        advantages:  Per-token advantage lists, parallel to completions.
                     Each inner list can be different length.

    Returns:
        The best completion string.

    Spec: REQ-LEARN-1221-2, SCENARIO-LEARN-1227
    """
    if not completions:
        raise ValueError("completions list is empty")
    if len(completions) != len(advantages):
        raise ValueError("completions and advantages must have the same length")

    best_idx = 0
    best_score = sum(advantages[0])
    for i in range(1, len(completions)):
        score = sum(advantages[i])
        if score > best_score:
            best_score = score
            best_idx = i
    return completions[best_idx]


def derive_fspo_honest_verdict(fspo_delta_pp: float) -> str:
    """Map the FSPO-vs-VPS accuracy delta to a standard verdict string.

    Args:
        fspo_delta_pp: Percentage-point difference (fspo_accuracy - vps_accuracy)
                       multiplied by 100.  Positive means FSPO improved over VPS.

    Returns:
        One of: "fspo_improves_over_vps", "fspo_matches_vps",
        "fspo_degrades_vps".

    Spec: REQ-LEARN-1221-3, SCENARIO-LEARN-1228
    """
    if fspo_delta_pp > 0:
        return "fspo_improves_over_vps"
    if fspo_delta_pp < 0:
        return "fspo_degrades_vps"
    return "fspo_matches_vps"
