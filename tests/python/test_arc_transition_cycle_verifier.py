"""Tests for the generic ARC forward/inverse transition-cycle verifier.

Spec refs: REQ-ARC-WMTE-5619,
SCENARIO-ARC-WMTE-5619-CYCLE-ADMISSION,
SCENARIO-ARC-WMTE-5619-CORRUPTION-REJECTION.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.agentic.arc_transition_cycle_verifier import (
    ObservedTransition,
    TransitionCycleVerifier,
    make_corrupted_transition,
)


def _click_transition(
    index: int, *, action: int = 6, x: int | None = None, y: int | None = None
) -> ObservedTransition:
    x = 1 + (index % 5) if x is None else x
    y = 1 + ((index * 2) % 5) if y is None else y
    before = np.zeros((8, 8), dtype=np.int16)
    before[y, x] = 1
    after = before.copy()
    after[y, x] = 5
    return ObservedTransition(
        game="toy",
        episode=f"ep-{index // 8}",
        step=index,
        state=before,
        action=action,
        data={"x": x, "y": y},
        successor=after,
    )


def _key_transition(index: int, *, action: int = 1) -> ObservedTransition:
    before = np.zeros((8, 8), dtype=np.int16)
    row = 1 + (index % 5)
    before[row, 1] = 2
    after = before.copy()
    after[row, 2] = 2
    after[row, 1] = 0
    return ObservedTransition(
        game="toy",
        episode=f"key-{index // 8}",
        step=index,
        state=before,
        action=action,
        data=None,
        successor=after,
    )


def test_scenario_arc_wmte_5619_cycle_admits_valid_transition_and_records_receipt() -> None:
    """SCENARIO-ARC-WMTE-5619-CYCLE-ADMISSION: forward replay plus inverse recovery must
    agree before a transition can become an update receipt."""

    calibration = [_click_transition(i) for i in range(24)] + [
        _key_transition(i) for i in range(24)
    ]
    verifier = TransitionCycleVerifier(min_support=3).fit(calibration)

    valid = _click_transition(99, x=4, y=4)
    decision = verifier.evaluate(valid)

    assert decision.admitted is True
    assert decision.successor_plausible is True
    assert decision.inverse_action_matches is True
    assert decision.inverse_action == 6
    assert decision.forward_replay_error == 0.0
    assert decision.update_receipt is not None
    assert decision.update_receipt["immutable"] is True
    assert decision.update_receipt["action"] == 6


def test_scenario_arc_wmte_5619_corruptions_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-5619-CORRUPTION-REJECTION: permuted actions, mismatched
    successors, no-op substitutions, and wrong-object changes must not be admitted."""

    calibration = [_click_transition(i) for i in range(24)] + [
        _key_transition(i) for i in range(24)
    ]
    verifier = TransitionCycleVerifier(min_support=3).fit(calibration)
    valid = _click_transition(101, x=3, y=5)
    other = _key_transition(102)

    corruptions = [
        make_corrupted_transition(valid, "permuted_action", replacement_action=1),
        make_corrupted_transition(
            valid, "mismatched_successor", replacement_successor=other.successor
        ),
        make_corrupted_transition(valid, "noop_substitution"),
        make_corrupted_transition(valid, "wrong_object_change"),
    ]

    for row in corruptions:
        decision = verifier.evaluate(row)
        assert decision.admitted is False, row.condition
        assert decision.update_receipt is None

    assert verifier.diagnostics()["unsafe_transition_accept_count"] == 0


class _FakeFrame:
    """Minimal stand-in for an arcengine frame: only .frame is read by grid_of."""

    def __init__(self, grid: np.ndarray) -> None:
        self.frame = grid
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.available_actions = [1, 6]


class _SpyCycleVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, int, Any, Any]] = []

    def observe_transition(self, before: Any, action: int, data: Any, after: Any) -> Any:
        self.calls.append((before, action, data, after))
        return SimpleNamespace(
            admitted=True,
            abstained=False,
            rejected=False,
            update_receipt={
                "receipt_id": "cycle:test",
                "immutable": True,
                "action": int(action),
            },
        )

    def diagnostics(self) -> dict[str, Any]:
        return {"calls": len(self.calls)}


def test_req_arc_wmte_5619_stepwise_explorer_feeds_live_observations_to_cycle_gate() -> None:
    """REQ-ARC-WMTE-5619: StepwiseExplorer exposes the verifier on the live observation
    path, so it can gate world-model update admission without a per-game adapter."""

    from carnot.agentic import arc_competition_agent as comp

    spy = _SpyCycleVerifier()
    explorer = comp.StepwiseExplorer(transition_cycle_verifier=spy)
    grid0 = np.zeros((3, 3), dtype=np.int16)
    explorer._ingest(_FakeFrame(grid0.copy()))
    origin = explorer.cur

    explorer.awaiting = {
        "origin": origin,
        "action": 6,
        "data": {"x": 1, "y": 1},
        "grid": _FakeFrame(grid0.copy()),
        "level_before": int(explorer.best_level),
        "previous_frame": _FakeFrame(grid0.copy()),
    }

    grid1 = grid0.copy()
    grid1[1, 1] = 5
    explorer._ingest(_FakeFrame(grid1))

    assert len(spy.calls) == 1
    before, action, data, after = spy.calls[0]
    assert np.array_equal(before.frame, grid0)
    assert action == 6
    assert data == {"x": 1, "y": 1}
    assert np.array_equal(after.frame, grid1)
    diagnostics = explorer.transition_cycle_diagnostics()
    assert diagnostics["enabled"] is True
    assert diagnostics["admitted_update_count"] == 1
    assert diagnostics["immutable_update_receipts"][0]["receipt_id"] == "cycle:test"
