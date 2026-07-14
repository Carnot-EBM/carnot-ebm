"""Tests for the ARC counterexample-patched executable transition model.

Spec refs: REQ-ARC-WMTE-5641,
SCENARIO-ARC-WMTE-5641-COUNTEREXAMPLE-PATCH-REPLAY,
SCENARIO-ARC-WMTE-5641-CONTROLS-AND-ABSTENTION.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_counterexample_executable_model import (
    ABSTAIN,
    ExecutableTransitionHypothesisPatcher,
    TransitionClause,
    TransitionReceipt,
    _component_at_click,
    _effect_key,
    _normal_interval,
    _usable,
    effect_signature,
    context_features,
    make_contradictory_receipt,
    make_unsupported_receipt,
    run_chronological_evaluation,
)


def _receipt(
    step: int,
    *,
    color: int,
    trace_id: str = "toy",
    x: int = 2,
    y: int = 2,
    effect_color: int = 9,
    reward: int = 0,
    terminal: bool = False,
) -> TransitionReceipt:
    before = np.zeros((8, 8), dtype=np.int16)
    before[y : y + 2, x : x + 2] = color
    after = before.copy()
    after[y : y + 2, x : x + 2] = effect_color
    return TransitionReceipt(
        trace_id=trace_id,
        episode="ep-0",
        step=step,
        state=before,
        action=6,
        data={"x": x, "y": y},
        successor=after,
        reward=reward,
        terminal=terminal,
    )


def test_scenario_5641_counterexample_patch_replay_and_contradiction_rejection() -> None:
    """SCENARIO-ARC-WMTE-5641-COUNTEREXAMPLE-PATCH-REPLAY: patches are triggered by
    falsifying receipts and must replay all accumulated receipts before acceptance."""

    patcher = ExecutableTransitionHypothesisPatcher(relax_support=1)
    first = _receipt(0, color=2)
    same_topology_new_identity = _receipt(1, color=3, x=4, y=2)
    future_same_topology = _receipt(2, color=4, x=2, y=4)

    assert patcher.predict(first).decision == ABSTAIN
    first_decision = patcher.observe(first)
    relax_decision = patcher.observe(same_topology_new_identity)

    assert first_decision.patch_accepted is True
    assert first_decision.patch_operator == "add"
    assert relax_decision.patch_accepted is True
    assert relax_decision.patch_operator == "relax"
    assert patcher.predict(future_same_topology).effect == future_same_topology.effect_signature
    assert patcher.all_receipt_replay_pass() is True

    contradictory = make_contradictory_receipt(first, effect_color=7)
    control = patcher.evaluate_patch_control(contradictory, "contradictory_patch")

    assert control["accepted"] is False
    assert control["rejected"] is True
    assert control["reason"] == "active_clause_contradiction"
    assert patcher.diagnostics()["unsafe_patch_accept_count"] == 0
    assert patcher.all_receipt_replay_pass() is True


def test_scenario_5641_controls_and_unsupported_abstention() -> None:
    """SCENARIO-ARC-WMTE-5641-CONTROLS-AND-ABSTENTION: unsupported objects abstain,
    informative mechanism questions beat irrelevant questions, and patched prediction
    improves over the unpatched arm on chronological receipts."""

    rows = [
        _receipt(0, color=2),
        _receipt(1, color=3, x=4, y=2),
        _receipt(2, color=4, x=2, y=4),
        _receipt(3, color=5, x=4, y=4),
        _receipt(4, color=6, x=1, y=1),
    ]
    unsupported = make_unsupported_receipt(rows[0])

    result = run_chronological_evaluation(
        {"toy": rows},
        unsupported_receipts=[unsupported],
        random_seed=5641,
    )

    assert result.heldout_transition_error_by_arm["patched"] < result.heldout_transition_error_by_arm[
        "unpatched"
    ]
    assert result.abstention_calibration["unsupported_abstention_rate"] == 1.0
    assert result.mechanism_question_controls["informative"]["score"] > result.mechanism_question_controls[
        "irrelevant"
    ]["score"]
    assert result.unsafe_patch_accept_count == 0
    assert result.all_receipt_replay_pass is True


def test_req_5641_defensive_branches_fail_closed() -> None:
    """REQ-ARC-WMTE-5641: malformed, unsupported, no-change, and contradictory
    edge cases remain abstaining or rejected instead of becoming accepted patches."""

    row = _receipt(0, color=2)
    unsupported = make_unsupported_receipt(row)
    patcher = ExecutableTransitionHypothesisPatcher()
    unsupported_decision = patcher.observe(unsupported)

    assert unsupported_decision.reason == "unsupported_object_abstained"
    assert patcher.predict(unsupported).decision == ABSTAIN
    assert _usable(unsupported) is False
    assert context_features(unsupported).reason == "unsupported_object"
    assert _effect_key(None) == ABSTAIN
    assert _normal_interval([])["n"] == 0
    assert _normal_interval([0.25]) == {"mean": 0.25, "lower": 0.25, "upper": 0.25, "n": 1}

    no_change = np.zeros((2, 2), dtype=np.int16)
    assert effect_signature(no_change, no_change)["changed_bbox_shape"] == [0, 0]
    assert effect_signature(no_change, np.zeros((3, 3), dtype=np.int16))["changed_count"] == -1
    assert _component_at_click(no_change, None) is None
    assert _component_at_click(no_change.reshape(1, 2, 2), {"x": 0, "y": 0}) is None
    assert _component_at_click(no_change, {"x": "bad", "y": 0}) is None
    assert _component_at_click(no_change, {"x": 0, "y": 0}) is None
    large = np.ones((24, 24), dtype=np.int16)
    assert _component_at_click(large, {"x": 1, "y": 1}, max_cells=8) is None

    frame_like = type("FrameLike", (), {"frame": row.state})()
    framed = TransitionReceipt(
        trace_id="framed",
        episode="ep-0",
        step=0,
        state=frame_like,
        action=row.action,
        data=row.data,
        successor=row.successor,
    )
    assert context_features(framed).supported is True

    no_click_contradiction = make_contradictory_receipt(
        TransitionReceipt(
            trace_id="toy",
            episode="ep-0",
            step=9,
            state=row.state,
            action=row.action,
            data=None,
            successor=row.successor,
        )
    )
    assert no_click_contradiction.successor.shape == row.successor.shape

    empty_component = make_contradictory_receipt(
        TransitionReceipt(
            trace_id="toy",
            episode="ep-0",
            step=10,
            state=no_change,
            action=6,
            data={"x": 0, "y": 0},
            successor=no_change.copy(),
        )
    )
    assert int(empty_component.successor[0, 0]) == 7

    action_clause = TransitionClause(
        clause_id="action-clause",
        selector_kind="action",
        selector_value="*",
        action=6,
        effect=row.effect_signature,
        support=2,
        created_by_patch="add",
    )
    unknown_clause = TransitionClause(
        clause_id="unknown-clause",
        selector_kind="unknown",
        selector_value="*",
        action=6,
        effect=row.effect_signature,
        support=2,
        created_by_patch="add",
    )
    assert action_clause.matches(context_features(row)) is True
    assert unknown_clause.specificity == 0
    assert unknown_clause.matches(context_features(row)) is False
