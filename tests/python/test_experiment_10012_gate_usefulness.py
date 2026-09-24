"""Small-fake tests for the Experiment 10012 gate-usefulness harness."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_10012_gate_usefulness as exp


def _pair(name: str, *, accepted: bool, useful: bool, faithful: bool) -> dict:
    return {
        "pair_id": name,
        "window_status": "label_informative",
        "gate_decisions": {"g": accepted},
        "labels": {"USEFUL": useful, "FAITHFUL": faithful},
    }


def test_label_logic_requires_level_up_and_three_matching_changed_steps() -> None:
    """REQ-ARC-WMTE-10012: USEFUL and FAITHFUL use their pre-registered definitions."""
    labels = exp.execution_labels(
        {
            "real_level_up": True,
            "matched_steps_before_divergence": 3,
            "real_state_changed": True,
        }
    )
    assert labels == {"USEFUL": True, "FAITHFUL": True}
    assert exp.execution_labels(
        {
            "real_level_up": False,
            "matched_steps_before_divergence": 2,
            "real_state_changed": True,
        }
    ) == {"USEFUL": False, "FAITHFUL": False}


def test_gate_table_arithmetic_counts_all_four_cells() -> None:
    """SCENARIO-ARC-WMTE-10012-GATE-TABLES: fixed gates reduce to four counts."""
    rows = [
        _pair("au", accepted=True, useful=True, faithful=False),
        _pair("an", accepted=True, useful=False, faithful=True),
        _pair("ru", accepted=False, useful=True, faithful=True),
        _pair("rn", accepted=False, useful=False, faithful=False),
    ]
    assert exp.build_gate_tables(rows, ["g"], label="USEFUL")["g"] == {
        "accepted_and_positive": 1,
        "accepted_and_negative": 1,
        "rejected_positive": 1,
        "rejected_negative": 1,
        "n_pairs": 4,
    }
    assert exp.build_gate_tables(rows, ["g"], label="FAITHFUL")["g"] == {
        "accepted_and_positive": 1,
        "accepted_and_negative": 1,
        "rejected_positive": 1,
        "rejected_negative": 1,
        "n_pairs": 4,
    }


class _FakeEnv:
    def __init__(self) -> None:
        self.value = 0

    def reset(self) -> SimpleNamespace:
        self.value = 1
        return SimpleNamespace(frame=[np.array([[self.value]])], levels_completed=0)

    def step(self, action: object, data: object = None) -> SimpleNamespace:
        del action, data
        self.value += 1
        return SimpleNamespace(frame=[np.array([[self.value]])], levels_completed=0)


def test_state_rebuild_checks_window_last_grid_and_uses_tmp_path(tmp_path) -> None:
    """SCENARIO-ARC-WMTE-10012-STATE-REBUILD: replay must match the final window grid."""
    actions = [
        {"action_index": 1, "action": "RESET", "data": {}},
        {"action_index": 2, "action": "ACTION1", "data": {}},
        {"action_index": 3, "action": "ACTION2", "data": {}},
    ]
    evidence = tmp_path / "state.json"
    rebuilt = exp.rebuild_state_from_actions(
        env=_FakeEnv(),
        action_rows=actions,
        induction_action_index=3,
        expected_grid=np.array([[2]]),
        action_resolver=lambda name: name,
        grid_reader=lambda frame: np.asarray(frame.frame[-1]),
        evidence_path=evidence,
    )
    assert rebuilt.recoverable is True
    assert rebuilt.level == 0
    assert np.array_equal(rebuilt.grid, np.array([[2]]))
    assert evidence.exists()

    mismatch = exp.rebuild_state_from_actions(
        env=_FakeEnv(),
        action_rows=actions,
        induction_action_index=3,
        expected_grid=np.array([[99]]),
        action_resolver=lambda name: name,
        grid_reader=lambda frame: np.asarray(frame.frame[-1]),
        evidence_path=tmp_path / "mismatch.json",
    )
    assert mismatch.recoverable is False
    assert mismatch.reason == "rebuilt_grid_mismatch"


def test_identity_useful_stops_the_run() -> None:
    """SCENARIO-ARC-WMTE-10012-LABEL-GUARDS: useful IDENTITY is a harness bug."""
    rows = [
        {"engine_family": "EXPERT", "labels": {"USEFUL": True}},
        {"engine_family": "IDENTITY", "labels": {"USEFUL": True}},
    ]
    with pytest.raises(exp.IdentityUsefulHarnessBug):
        exp.classify_window_labels(rows)


def test_expert_without_level_up_marks_window_uninformative() -> None:
    """SCENARIO-ARC-WMTE-10012-LABEL-GUARDS: failed EXPERT excludes the window."""
    rows = [
        {"engine_family": "EXPERT", "labels": {"USEFUL": False}},
        {"engine_family": "THINK", "labels": {"USEFUL": True}},
    ]
    decision = exp.classify_window_labels(rows)
    assert decision == {
        "status": "label_uninformative",
        "reason": "expert_no_real_level_up_within_live_planning_budget",
    }


def test_gate_tables_exclude_unrecoverable_and_uninformative_pairs() -> None:
    """REQ-ARC-WMTE-10012: gate tables contain only informative recovered windows."""
    rows = [
        _pair("kept", accepted=True, useful=True, faithful=True),
        {
            **_pair("unrecoverable", accepted=True, useful=False, faithful=False),
            "window_status": "state_unrecoverable",
        },
        {
            **_pair("uninformative", accepted=False, useful=True, faithful=True),
            "window_status": "label_uninformative",
        },
    ]
    table = exp.build_gate_tables(rows, ["g"], label="USEFUL")["g"]
    assert table["n_pairs"] == 1
    assert table["accepted_and_positive"] == 1


class _PlanEnv:
    def __init__(self) -> None:
        self.value = 0
        self.reset_calls = 0

    def reset(self) -> SimpleNamespace:
        self.reset_calls += 1
        self.value = 0
        return SimpleNamespace(frame=[np.array([[0]])], levels_completed=0)

    def step(self, action: object, data: object = None) -> SimpleNamespace:
        del action, data
        self.value += 1
        return SimpleNamespace(
            frame=[np.array([[self.value]])],
            levels_completed=int(self.value >= 3),
        )


def _candidate_with_wrong_dynamics() -> exp.Candidate:
    source = """
import numpy as np
def engine(grid, action, data=None):
    out = np.asarray(grid).copy()
    out[0, 0] += 2
    return out
def is_level_complete(grid):
    return bool(np.asarray(grid)[0, 0] >= 6)
"""
    return exp.Candidate("fake", "fake", "THINK", "pilot", source, None, None, "cached")


def _rebuilt(env: _PlanEnv) -> exp.RebuiltState:
    return exp.RebuiltState(
        recoverable=True,
        reason=None,
        grid=np.array([[0]]),
        level=0,
        frame=SimpleNamespace(frame=[np.array([[0]])], levels_completed=0),
        env=env,
        actions_replayed=1,
        expected_sha256="expected",
        observed_sha256="observed",
    )


def _three_step_plan(*args: object, **kwargs: object) -> list[dict]:
    del args, kwargs
    return [{"action": 1, "data": None} for _ in range(3)]


def test_live_scored_executes_after_divergence_and_counts_waste(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10012-A1-DIVERGENCE-WASTE: live replay does not halt."""
    monkeypatch.setattr(exp.e3, "plan_in_model", _three_step_plan)
    monkeypatch.setattr(exp.e3, "to_logical", lambda grid, cell: np.asarray(grid))
    result = exp.plan_and_execute_from_state(
        _candidate_with_wrong_dynamics(),
        _rebuilt(_PlanEnv()),
        arm=exp.LIVE_SCORED,
    )
    assert result["real_level_up"] is True
    assert result["real_actions_used"] == 3
    assert result["first_divergence_index"] == 0
    assert result["real_actions_spent_after_first_divergence"] == 2
    assert result["halted_on_divergence"] is False


def test_offline_twin_halts_at_first_divergence(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-10012-A1-EXECUTION-ARMS: the retained twin arm still halts."""
    monkeypatch.setattr(exp.e3, "plan_in_model", _three_step_plan)
    monkeypatch.setattr(exp.e3, "to_logical", lambda grid, cell: np.asarray(grid))
    result = exp.plan_and_execute_from_state(
        _candidate_with_wrong_dynamics(),
        _rebuilt(_PlanEnv()),
        arm=exp.OFFLINE_TWIN_HALT,
    )
    assert result["real_level_up"] is False
    assert result["real_actions_used"] == 1
    assert result["first_divergence_index"] == 0
    assert result["real_actions_spent_after_first_divergence"] == 0
    assert result["halted_on_divergence"] is True


def test_start_state_choice_matches_scored_induction_reason() -> None:
    """SCENARIO-ARC-WMTE-10012-A1-EXECUTION-ARMS: stalls reset; level-up resumes."""
    assert exp.execute_plan_from_current("stall") is False
    assert exp.execute_plan_from_current("renewed_stall_reinduction") is False
    assert exp.execute_plan_from_current("level_up_reinduction") is True
    assert exp.plan_start_state("stall") == "root_after_reset"
    assert exp.plan_start_state("level_up_reinduction") == "induction_state"


def test_reset_state_calls_reset_and_uses_root_grid(tmp_path) -> None:
    """SCENARIO-ARC-WMTE-10012-A1-EXECUTION-ARMS: root execution performs RESET."""
    env = _PlanEnv()
    env.value = 99
    rebuilt = exp.rebuild_reset_state_from_env(
        env=env,
        expected_level=0,
        grid_reader=lambda frame: np.asarray(frame.frame[-1]),
        evidence_path=tmp_path / "root.json",
    )
    assert env.reset_calls == 1
    assert rebuilt.recoverable is True
    assert rebuilt.actions_replayed == 1
    assert np.array_equal(rebuilt.grid, np.array([[0]]))


def test_identity_supplies_goal_when_expert_predicate_is_missing() -> None:
    """SCENARIO-ARC-WMTE-10012-A1-CONTROLS-AND-TABLES: IDENTITY owns its false goal."""
    source = exp.identity_source_from_expert("def helper():\n    return 1\n")
    candidate = exp.Candidate("id", "g", "IDENTITY", "negative", source, None, None, "derived")
    namespace, error = exp.load_engine_namespace(candidate)
    assert error is None
    assert namespace is not None
    assert np.array_equal(namespace["engine"](np.array([[4]]), 1), np.array([[4]]))
    assert namespace["is_level_complete"](np.array([[4]])) is False


def test_gate_table_can_remove_expert_controls() -> None:
    """SCENARIO-ARC-WMTE-10012-A1-CONTROLS-AND-TABLES: candidate-only counts are explicit."""
    rows = [
        {**_pair("expert", accepted=True, useful=True, faithful=True), "engine_family": "EXPERT"},
        {
            **_pair("identity", accepted=True, useful=False, faithful=False),
            "engine_family": "IDENTITY",
        },
        {**_pair("think", accepted=False, useful=True, faithful=True), "engine_family": "THINK"},
    ]
    table = exp.build_gate_tables(
        rows,
        ["g"],
        label="USEFUL",
        candidate_only=True,
    )["g"]
    assert table == {
        "accepted_and_positive": 0,
        "accepted_and_negative": 0,
        "rejected_positive": 1,
        "rejected_negative": 0,
        "n_pairs": 1,
    }
