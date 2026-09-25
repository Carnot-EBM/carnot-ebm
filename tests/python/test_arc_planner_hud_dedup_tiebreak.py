"""REQ-ARC-WMTE-10013 planner HUD-deduplication and tie-break tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_executable_world_model as e3


def _candidate(action: int) -> dict[str, object]:
    return {"action": action, "data": None}


def _transition(before: np.ndarray, after: np.ndarray) -> e3.Transition:
    return e3.Transition(before, 1, None, after, 0, 0)


def test_flags_off_preserve_planner_result_and_diagnostics_bytes(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-FLAGS-OFF-IDENTITY preserves fixture bytes."""
    monkeypatch.delenv("CARNOT_ARC_PLAN_HUD_DEDUP", raising=False)
    monkeypatch.delenv("CARNOT_ARC_PLAN_GOAL_TIEBREAK", raising=False)
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate(1), _candidate(2)])

    def engine(grid, action, _data):
        out = np.asarray(grid).copy()
        out[0, 0] += action
        return out

    def run_once() -> bytes:
        diagnostics: dict[str, object] = {}
        plan = e3.plan_in_model(
            engine,
            lambda grid: int(grid[0, 0]) >= 3,
            np.zeros((1, 2), dtype=np.int16),
            max_nodes=20,
            max_depth=4,
            goal_energy=lambda _grid: 1.0,
            diagnostics=diagnostics,
        )
        return json.dumps(
            {"plan": plan, "diagnostics": diagnostics},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    baseline = run_once()
    replay = run_once()
    assert replay == baseline
    decoded = json.loads(baseline)
    assert decoded["plan"] == [_candidate(1), _candidate(2)]
    assert decoded["diagnostics"]["nodes_expanded"] == 4
    assert not any("hud_dedup" in key for key in decoded["diagnostics"])
    assert not any("tiebreak" in key for key in decoded["diagnostics"])


def test_mask_merges_only_mask_cell_variants(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-HUD-DEDUP-SAFETY merges HUD-only variants."""
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate(1), _candidate(2)])
    root = np.zeros((1, 2), dtype=np.int16)

    def engine(grid, action, _data):
        out = np.asarray(grid).copy()
        out[0, 0] = 1
        out[0, 1] = action
        return out

    unmasked_diag: dict[str, object] = {}
    masked_diag: dict[str, object] = {}
    assert (
        e3.plan_in_model(
            engine,
            lambda _grid: False,
            root,
            max_nodes=20,
            max_depth=3,
            diagnostics=unmasked_diag,
        )
        is None
    )
    assert (
        e3.plan_in_model(
            engine,
            lambda _grid: False,
            root,
            max_nodes=20,
            max_depth=3,
            diagnostics=masked_diag,
            dedup_mask=np.array([[False, True]]),
        )
        is None
    )
    assert masked_diag["hud_dedup_states_merged"] == 1
    assert masked_diag["nodes_expanded"] < unmasked_diag["nodes_expanded"]
    assert masked_diag["hud_dedup_mask_status"] == "applied"
    assert masked_diag["hud_dedup_mask_reason"] == "accepted"


def test_shape_incompatible_mask_is_truthfully_diagnosed(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-AM1-TRUTHFUL-STATUS rejects wrong shapes."""
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [])
    diagnostics: dict[str, object] = {}
    e3.plan_in_model(
        lambda grid, _action, _data: grid,
        lambda _grid: False,
        np.zeros((1, 2), dtype=np.int16),
        diagnostics=diagnostics,
        dedup_mask=np.zeros((2, 2), dtype=bool),
    )
    assert diagnostics["hud_dedup_mask_status"] == "not_used"
    assert diagnostics["hud_dedup_mask_reason"] == "shape_mismatch"


def test_goal_and_engine_still_receive_full_masked_cells(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-HUD-DEDUP-SAFETY keeps full-grid semantics."""
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate(1)])
    engine_inputs: list[np.ndarray] = []
    goal_inputs: list[np.ndarray] = []

    def engine(grid, _action, _data):
        engine_inputs.append(np.asarray(grid).copy())
        return np.array([[1, 7]], dtype=np.int16)

    def goal(grid):
        goal_inputs.append(np.asarray(grid).copy())
        return int(grid[0, 0]) == 1 and int(grid[0, 1]) == 7

    plan = e3.plan_in_model(
        engine,
        goal,
        np.array([[0, 3]], dtype=np.int16),
        max_nodes=5,
        dedup_mask=np.array([[False, True]]),
    )
    assert plan == [_candidate(1)]
    assert int(engine_inputs[0][0, 1]) == 3
    assert int(goal_inputs[0][0, 1]) == 7


def test_masked_duplicate_goal_is_checked_before_skip(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-AM1-GOAL-BEFORE-DEDUP keeps HUD goals."""
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate(1)])

    def engine(grid, _action, _data):
        out = np.asarray(grid).copy()
        out[0, 0] = (int(out[0, 0]) + 1) % 3
        out[0, 1] += 1
        return out

    goal = lambda grid: int(grid[0, 1]) == 3
    root = np.zeros((1, 2), dtype=np.int16)
    expected = [_candidate(1)] * 3
    unmasked = e3.plan_in_model(engine, goal, root, goal_energy=lambda _grid: 1.0)
    diagnostics: dict[str, object] = {}
    masked = e3.plan_in_model(
        engine,
        goal,
        root,
        goal_energy=lambda _grid: 1.0,
        diagnostics=diagnostics,
        dedup_mask=np.array([[False, True]]),
    )
    assert unmasked == expected
    assert masked == expected
    assert diagnostics["termination_reason"] == "plan_found"


def test_novelty_tiebreak_prefers_more_changed_non_hud_cells(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-NOVELTY-ORDER prefers greater novelty."""
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate(1), _candidate(2)])

    def engine(grid, action, _data):
        state = tuple(int(value) for value in np.asarray(grid).flat)
        if state == (0, 0, 0, 0):
            return np.array([[1, 0, 7, 7] if action == 1 else [1, 1, 0, 0]])
        if state == (1, 0, 7, 7):
            return np.array([[9, 0, 7, 7]])
        if state == (1, 1, 0, 0):
            return np.array([[9, 1, 0, 0]])
        return np.array([[2, 0, 0, 0]])

    fifo_plan = e3.plan_in_model(
        engine,
        lambda grid: int(grid[0, 0]) == 9,
        np.zeros((1, 4), dtype=np.int16),
        max_nodes=20,
        goal_energy=lambda _grid: 1.0,
        dedup_mask=np.array([[False, False, True, True]]),
    )
    assert fifo_plan == [_candidate(1), _candidate(1)]

    diagnostics: dict[str, object] = {}
    plan = e3.plan_in_model(
        engine,
        lambda grid: int(grid[0, 0]) == 9,
        np.zeros((1, 4), dtype=np.int16),
        max_nodes=20,
        goal_energy=lambda _grid: 1.0,
        diagnostics=diagnostics,
        dedup_mask=np.array([[False, False, True, True]]),
        goal_tiebreak="novelty",
    )
    assert plan == [_candidate(2), _candidate(1)]
    assert diagnostics["goal_tiebreak_mode"] == "novelty"
    assert diagnostics["goal_tiebreak_states_scored"] >= 2


def test_novelty_tiebreak_uses_insertion_order_as_tertiary_key(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-NOVELTY-ORDER is deterministic on equal novelty."""
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate(1), _candidate(2)])

    def engine(grid, action, _data):
        state = tuple(int(value) for value in np.asarray(grid).flat)
        if state == (0, 0):
            return np.array([[1, 0] if action == 1 else [0, 1]])
        if state == (1, 0):
            return np.array([[9, 0]])
        return np.array([[0, 2]])

    plans = [
        e3.plan_in_model(
            engine,
            lambda grid: int(grid[0, 0]) == 9,
            np.zeros((1, 2), dtype=np.int16),
            max_nodes=20,
            goal_energy=lambda _grid: 1.0,
            goal_tiebreak="novelty",
        )
        for _ in range(3)
    ]
    assert plans == [[_candidate(1), _candidate(1)]] * 3


def test_tiebreak_flag_does_not_change_blind_fifo_branch(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-NOVELTY-ORDER leaves no-energy FIFO unchanged."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_GOAL_TIEBREAK", "novelty")
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate(1), _candidate(2)])

    def engine(grid, action, _data):
        out = np.asarray(grid).copy()
        out[0, 0] += action
        return out

    diagnostics: dict[str, object] = {}
    plan = e3.plan_in_model(
        engine,
        lambda grid: int(grid[0, 0]) >= 3,
        np.zeros((1, 1), dtype=np.int16),
        max_nodes=20,
        diagnostics=diagnostics,
        goal_tiebreak="novelty",
    )
    assert plan == [_candidate(1), _candidate(2)]
    assert diagnostics["used_goal_energy_search"] is False
    assert "goal_tiebreak_mode" not in diagnostics


def test_direct_caller_default_ignores_tiebreak_environment(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-AM1-TIEBREAK-SCOPE keeps direct defaults."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_GOAL_TIEBREAK", "novelty")
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate(1), _candidate(2)])

    def engine(grid, action, _data):
        state = tuple(int(value) for value in np.asarray(grid).flat)
        if state == (0, 0, 0, 0):
            return np.array([[1, 0, 7, 7] if action == 1 else [1, 1, 0, 0]])
        if state == (1, 0, 7, 7):
            return np.array([[9, 0, 7, 7]])
        if state == (1, 1, 0, 0):
            return np.array([[9, 1, 0, 0]])
        return np.array([[2, 0, 0, 0]])

    plan = e3.plan_in_model(
        engine,
        lambda grid: int(grid[0, 0]) == 9,
        np.zeros((1, 4), dtype=np.int16),
        max_nodes=20,
        goal_energy=lambda _grid: 1.0,
        dedup_mask=np.array([[False, False, True, True]]),
    )
    assert plan == [_candidate(1), _candidate(1)]


def _bare_policy(mask: np.ndarray, transitions: list[e3.Transition]):
    policy = object.__new__(agent.E3AgentPolicy)
    policy.two_sided_goal_contract = None
    policy.explorer = SimpleNamespace(hud_mask=mask)
    policy.cell = 1
    policy.transitions = transitions
    policy._episode_transition_start = 0
    return policy


def test_wrapper_does_not_read_or_pass_mask_when_flag_is_off(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-FLAGS-OFF-IDENTITY avoids wrapper side effects."""
    monkeypatch.delenv("CARNOT_ARC_PLAN_HUD_DEDUP", raising=False)

    class ExplodingExplorer:
        @property
        def hud_mask(self):
            raise AssertionError("disabled wrapper read the explorer mask")

    policy = _bare_policy(np.zeros((2, 3), dtype=bool), [])
    policy.explorer = ExplodingExplorer()
    captured: dict[str, object] = {}

    def planner(_engine, _goal, _grid, **kwargs):
        captured.update(kwargs)
        return []

    assert (
        policy._call_plan_in_model(
            planner,
            object(),
            lambda _grid: False,
            np.zeros((2, 3)),
            goal_energy_override=lambda _grid: 1.0,
        )
        == []
    )
    assert "dedup_mask" not in captured
    assert "goal_tiebreak" not in captured


def test_wrapper_passes_only_a_swallow_clean_mask(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-LIVE-WRAPPER passes only the clean mask."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "1")
    before = np.zeros((2, 3), dtype=np.int16)
    after = before.copy()
    after[0, 0:2] = 4
    clean_mask = np.zeros((2, 3), dtype=bool)
    clean_mask[1, :] = True
    policy = _bare_policy(clean_mask, [_transition(before, after)])
    diagnostics: dict[str, object] = {}
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [])

    policy._call_plan_in_model(
        e3.plan_in_model,
        lambda grid, _action, _data: grid,
        lambda _grid: False,
        before,
        diagnostics=diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert diagnostics["planner_hud_dedup_mask_status"] == "applied"
    assert diagnostics["planner_hud_dedup_planner_reason"] == "accepted"
    assert diagnostics["planner_hud_dedup_swallow"]["reason"] == "ok"


def test_wrapper_copies_shape_mismatch_from_planner(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-AM1-TRUTHFUL-STATUS copies shape refusal."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "1")
    before = np.zeros((2, 3), dtype=np.int16)
    after = before.copy()
    after[0, 0] = 4
    clean_mask = np.zeros((2, 3), dtype=bool)
    clean_mask[1, :] = True
    policy = _bare_policy(clean_mask, [_transition(before, after)])
    diagnostics: dict[str, object] = {}
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [])

    policy._call_plan_in_model(
        e3.plan_in_model,
        lambda grid, _action, _data: grid,
        lambda _grid: False,
        np.zeros((1, 3), dtype=np.int16),
        diagnostics=diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert diagnostics["planner_hud_dedup_mask_status"] == "not_used"
    assert diagnostics["planner_hud_dedup_planner_reason"] == "shape_mismatch"


def test_wrapper_reports_mask_keyword_not_accepted(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-AM1-TRUTHFUL-STATUS reports old callees."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "1")
    before = np.zeros((2, 3), dtype=np.int16)
    after = before.copy()
    after[0, 0] = 4
    clean_mask = np.zeros((2, 3), dtype=bool)
    clean_mask[1, :] = True
    policy = _bare_policy(clean_mask, [_transition(before, after)])
    diagnostics: dict[str, object] = {}

    def planner(_engine, _goal, _grid, *, diagnostics=None, goal_energy=None):
        return []

    policy._call_plan_in_model(
        planner,
        object(),
        lambda _grid: False,
        before,
        diagnostics=diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert diagnostics["planner_hud_dedup_mask_status"] == "not_used"
    assert diagnostics["planner_hud_dedup_planner_reason"] == "callable_signature_rejected"


def test_wrapper_passes_tiebreak_explicitly(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-AM1-TIEBREAK-SCOPE passes novelty explicitly."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_GOAL_TIEBREAK", "novelty")
    monkeypatch.delenv("CARNOT_ARC_PLAN_HUD_DEDUP", raising=False)
    policy = _bare_policy(np.zeros((1, 1), dtype=bool), [])
    captured: dict[str, object] = {}

    def planner(_engine, _goal, _grid, **kwargs):
        captured.update(kwargs)
        return []

    policy._call_plan_in_model(
        planner,
        object(),
        lambda _grid: False,
        np.zeros((1, 1), dtype=np.int16),
        goal_energy_override=lambda _grid: 1.0,
    )
    assert captured["goal_tiebreak"] == "novelty"


def test_wrapper_refuses_a_mask_that_swallows_game_cells(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10013-HUD-DEDUP-SAFETY refuses a swallowing mask."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "1")
    before = np.zeros((2, 3), dtype=np.int16)
    after = before.copy()
    after[0, 0] = 4
    swallowing = np.ones((2, 3), dtype=bool)
    policy = _bare_policy(swallowing, [_transition(before, after)])
    captured: dict[str, object] = {}
    diagnostics: dict[str, object] = {}

    def planner(_engine, _goal, _grid, **kwargs):
        captured.update(kwargs)
        return []

    policy._call_plan_in_model(
        planner,
        object(),
        lambda _grid: False,
        before,
        diagnostics=diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert "dedup_mask" not in captured
    assert diagnostics["planner_hud_dedup_mask_status"] == "refused"
    assert diagnostics["planner_hud_dedup_swallow"]["swallows"] is True
