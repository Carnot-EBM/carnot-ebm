"""Tests for ActFocus token-energy weighting.

Spec: REQ-LEARN-2242, SCENARIO-LEARN-2242.
"""

from __future__ import annotations

from carnot.training.actfocus import (
    actfocus_fast_update_score,
    build_token_energy_trace,
    compute_actfocus_weights,
    energy_variance_by_role,
)


def test_req_learn_2242_action_tokens_receive_larger_weights() -> None:
    """REQ-LEARN-2242: action-token variance upweights fast-update evidence."""

    trace = build_token_energy_trace(
        "Step 1: decompose the operands and check the carry. Final: answer = 42, corrected.",
        base_energy=1.3,
    )

    variances = energy_variance_by_role(trace)
    weights = compute_actfocus_weights(trace)

    action_weights = [row.weight for row in weights if row.role == "action"]
    reasoning_weights = [row.weight for row in weights if row.role == "reasoning"]

    assert variances["action"] > variances["reasoning"]
    assert sum(action_weights) / len(action_weights) > sum(reasoning_weights) / len(
        reasoning_weights
    )
    assert actfocus_fast_update_score(trace) > 0.0


def test_req_learn_2242_reasoning_only_trace_has_no_action_update_score() -> None:
    """REQ-LEARN-2242: no answer-action tokens means no ActFocus retention value."""

    trace = build_token_energy_trace(
        "Step 1: decompose the operands. Step 2: compare the intermediate totals.",
        base_energy=0.8,
    )

    variances = energy_variance_by_role(trace)

    assert variances["action"] == 0.0
    assert actfocus_fast_update_score(trace) == 0.0
