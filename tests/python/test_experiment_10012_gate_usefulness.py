"""Small-fake tests for the Experiment 10012 gate-usefulness harness."""

from __future__ import annotations

from pathlib import Path
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
        "unmeasured": 0,
        "n_pairs": 4,
        "n_measured_pairs": 4,
    }
    assert exp.build_gate_tables(rows, ["g"], label="FAITHFUL")["g"] == {
        "accepted_and_positive": 1,
        "accepted_and_negative": 1,
        "rejected_positive": 1,
        "rejected_negative": 1,
        "unmeasured": 0,
        "n_pairs": 4,
        "n_measured_pairs": 4,
    }


def test_zero_scorable_rows_are_unmeasured_not_rejected() -> None:
    """SCENARIO-ARC-WMTE-10012-A3-UNMEASURED-GATES: null is not rejection."""
    decisions = exp.candidate_gate_decisions(
        {
            "live_unmasked_exact_accuracy": None,
            "live_unmasked_n": 0,
            "masked_exact_accuracy": None,
            "masked_n": 0,
            "masked_change_fidelity": None,
            "cell_recall": None,
            "no_op_rate_for_gate": 0.0,
        }
    )
    assert all(decision is None for decision in decisions.values())
    row = _pair("zero", accepted=False, useful=True, faithful=False)
    row["gate_decisions"] = {"g": None}
    table = exp.build_gate_tables([row], ["g"], label="USEFUL")["g"]
    assert table["unmeasured"] == 1
    assert table["n_measured_pairs"] == 0
    assert table["rejected_positive"] == 0

    levelup = exp.e3.Transition(np.array([[0]]), 1, None, np.array([[1]]), 0, 1)
    spec = exp.exp10.WindowSpec(
        game="fake",
        index=0,
        window_file=Path("fake.json"),
        window_sha256="fake",
        rows=[levelup],
        n_prefix=0,
        heldout_indices=(0,),
        excluded_indices=(),
        mask_rows=(),
        cell=1,
        reported_digest={},
        report_split_text="fake",
    )
    candidate = exp.Candidate(
        "fake",
        "fake",
        "THINK",
        "fake",
        "def engine(grid, action, data=None): return grid\n",
        None,
        None,
        "cached",
    )
    metrics = exp.score_gate_metrics(candidate, spec)
    assert metrics["live_selector"]["accepted"] is None
    assert metrics["live_selector"]["measurement_status"] == "unmeasured"
    assert metrics["existing_change_gate"]["counterfactual_enabled_decision"] is None


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


def test_queue_exhaustion_is_not_mislabeled_planner_budget() -> None:
    """SCENARIO-ARC-WMTE-10012-A3-PROVENANCE-AND-LIMITS: vc33 reason is precise."""
    false_labels = {arm.name: {"USEFUL": False} for arm in exp.EXECUTION_ARMS}
    false_labels[exp.OFFLINE_TWIN_HALT.name] = {"USEFUL": True}
    rows = [
        {
            "engine_family": "EXPERT",
            "goal_predicate_status": "callable",
            "labels_by_arm": false_labels,
            "executions": {
                exp.LIVE_SCORED.name: {
                    "planner_diagnostics": {
                        "termination_reason": "queue_exhausted",
                        "nodes_expanded": 814,
                    }
                },
                exp.OFFLINE_TWIN_HALT.name: {"planner_diagnostics": {}},
                exp.BUDGET_150K.name: {
                    "planner_diagnostics": {
                        "termination_reason": "queue_exhausted",
                        "nodes_expanded": 814,
                    }
                },
            },
        },
        {
            "engine_family": "IDENTITY",
            "labels_by_arm": {arm.name: {"USEFUL": False} for arm in exp.EXECUTION_ARMS},
        },
    ]
    result = exp.classify_window_arms(rows)
    assert result[exp.LIVE_SCORED.name]["reason"] == (
        "expert_dynamics_or_goal_gap_queue_exhausted_814_nodes"
    )
    assert result[exp.BUDGET_150K.name]["reason"] != "planner_budget"


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
        "unmeasured": 0,
        "n_pairs": 1,
        "n_measured_pairs": 1,
    }


def test_completion_predicate_validation_checks_boundary_and_window() -> None:
    """SCENARIO-ARC-WMTE-10012-A2-COMPLETION-PREDICATES: all checks are explicit."""
    row = exp.e3.Transition(np.array([[0]]), 1, None, np.array([[1]]), 0, 0)
    result = exp.validate_completion_predicate(
        lambda grid: bool(grid[0, 0] == 9),
        [np.array([[0]]), np.array([[1]]), np.array([[9]])],
        [row],
    )
    assert result["valid"] is True
    assert result["recorded_window_true_count"] == 0


def test_transition_rebuild_proof_hashes_all_fields() -> None:
    """SCENARIO-ARC-WMTE-10012-A2-PAIR-QUALIFICATION: rebuild proof is row exact."""
    row = exp.e3.Transition(np.array([[0]]), 6, {"x": 1}, np.array([[2]]), 0, 0)
    same = exp.e3.Transition(np.array([[0]]), 6, {"x": 1}, np.array([[2]]), 0, 0)
    changed = exp.e3.Transition(np.array([[0]]), 6, {"x": 2}, np.array([[2]]), 0, 0)
    assert exp.transition_rows_equal([row], [same])
    assert exp.canonical_transition_hash([row]) == exp.canonical_transition_hash([same])
    assert not exp.transition_rows_equal([row], [changed])
    assert exp.canonical_transition_hash([row]) != exp.canonical_transition_hash([changed])


def test_stored_pair_qualification_requires_every_recorded_field() -> None:
    """SCENARIO-ARC-WMTE-10012-A2-PAIR-QUALIFICATION: one failure rejects the pair."""
    source = b"def engine(grid, action, data=None): return grid\n"
    row = exp.e3.Transition(np.array([[0]]), 1, None, np.array([[1]]), 0, 0)
    qualified = exp.qualify_stored_pair(
        source=source,
        recorded_sha256=exp.sha256_bytes(source),
        rebuilt_rows=[row],
        comparison_rows=[row],
        reset_matches=True,
        induction_reason="offline_stall_window",
        model="Qwen3.8-27B",
        think_mode="natural_inline_reasoning",
        token_budget=102400,
        field_provenance={
            "window_row_match": "historical_rows",
            "stall_start_recorded": "historical_record",
            "model_recorded": "historical_record",
            "think_mode_recorded": "historical_record",
            "token_budget_recorded": "historical_record",
        },
    )
    assert qualified["qualified"] is True
    constant_claims = exp.qualify_stored_pair(
        source=source,
        recorded_sha256=exp.sha256_bytes(source),
        rebuilt_rows=[row],
        comparison_rows=[row],
        reset_matches=True,
        induction_reason="offline_stall_window",
        model="Qwen3.8-27B",
        think_mode="natural_inline_reasoning",
        token_budget=102400,
        field_provenance={
            "window_row_match": "fresh_rebuild_compared_to_fresh_rebuild",
            "stall_start_recorded": "constant",
            "model_recorded": "constant",
            "think_mode_recorded": "constant",
            "token_budget_recorded": "constant",
        },
    )
    assert constant_claims["qualified"] is False
    assert set(constant_claims["rejected_reasons"]) == {
        "window_row_match",
        "stall_start_recorded",
        "model_recorded",
        "think_mode_recorded",
        "token_budget_recorded",
    }
    rejected = exp.qualify_stored_pair(
        source=source,
        recorded_sha256="wrong",
        rebuilt_rows=[row],
        comparison_rows=[],
        reset_matches=False,
        induction_reason="unknown",
        model="",
        think_mode="",
        token_budget=None,
    )
    assert rejected["qualified"] is False
    assert set(rejected["rejected_reasons"]) == set(rejected["checks"])


def test_control_categories_are_never_silently_pooled() -> None:
    """SCENARIO-ARC-WMTE-10012-A2-CONTROL-CATEGORIES: solver is a separate category."""
    assert exp.control_category(has_expert=True) == "expert_live_planner"
    assert exp.control_category(has_expert=False) == "informative_by_registry_solver"


def test_required_gate_table_cohorts_filter_independently() -> None:
    """SCENARIO-ARC-WMTE-10012-A2-COHORT-TABLES: all four views have fixed filters."""
    base = _pair("row", accepted=True, useful=True, faithful=True)
    rows = [
        {
            **base,
            "pair_id": "qwen",
            "engine_family": "QWEN38",
            "model_family": "qwen3.8-27b",
            "control_category": "informative_by_registry_solver",
        },
        {
            **base,
            "pair_id": "expert",
            "engine_family": "EXPERT",
            "model_family": "source-derived-expert",
            "is_control": True,
            "control_category": "expert_live_planner",
        },
        {
            **base,
            "pair_id": "other",
            "engine_family": "THINK",
            "model_family": "qwen3.5-9b-mtp",
            "control_category": "expert_live_planner",
        },
    ]
    assert exp.build_gate_tables(rows, ["g"], label="USEFUL")["g"]["n_pairs"] == 3
    assert (
        exp.build_gate_tables(rows, ["g"], label="USEFUL", cohort="candidates_only")["g"]["n_pairs"]
        == 2
    )
    assert (
        exp.build_gate_tables(rows, ["g"], label="USEFUL", cohort="qwen38_only")["g"]["n_pairs"]
        == 1
    )
    assert (
        exp.build_gate_tables(rows, ["g"], label="USEFUL", cohort="expert_controlled_windows")["g"][
            "n_pairs"
        ]
        == 2
    )


def test_amendment3_main_and_appendix_cohorts_do_not_overlap() -> None:
    """SCENARIO-ARC-WMTE-10012-A3-COHORT-SPLIT: h2h replay stays appendix-only."""
    base = _pair("row", accepted=True, useful=True, faithful=True)
    rows = [
        {
            **base,
            "pair_id": "stall-candidate",
            "engine_family": "THINK",
            "provenance_cohort": "stall_window",
            "control_category": "expert_live_planner",
        },
        {
            **base,
            "pair_id": "stall-control",
            "engine_family": "EXPERT",
            "is_control": True,
            "provenance_cohort": "stall_window",
            "control_category": "expert_live_planner",
        },
        {
            **base,
            "pair_id": "h2h-expert",
            "engine_family": "QWEN38",
            "provenance_cohort": "h2h_replay_counterfactual",
            "control_category": "expert_live_planner",
        },
        {
            **base,
            "pair_id": "h2h-registry",
            "engine_family": "QWEN38",
            "provenance_cohort": "h2h_replay_counterfactual",
            "control_category": "informative_by_registry_solver",
        },
    ]
    kwargs = {"rows": rows, "gate_names": ["g"], "label": "USEFUL"}
    main = exp.build_gate_tables(**kwargs, cohort="main_with_controls")
    candidates = exp.build_gate_tables(**kwargs, cohort="main_without_controls")
    replay = exp.build_gate_tables(**kwargs, cohort="h2h_replay_counterfactual")
    registry = exp.build_gate_tables(**kwargs, cohort="registry_solver_controlled")
    assert main["g"]["n_pairs"] == 2
    assert candidates["g"]["n_pairs"] == 1
    assert replay["g"]["n_pairs"] == 2
    assert registry["g"]["n_pairs"] == 1


def test_repaired_predicates_validate_on_real_registered_trajectories() -> None:
    """SCENARIO-ARC-WMTE-10012-A2-COMPLETION-PREDICATES: validate real boundaries."""
    repo_root = Path(__file__).resolve().parents[2]
    if not exp.resolve_environment_files(repo_root).is_dir():
        pytest.skip("public environment_files are unavailable")
    paths = exp.exp10.EvidencePaths.under(repo_root)
    reports = exp.exp10.load_control_reports(paths)
    specs = {
        game: exp.exp10.load_window(game, index, reports[game], paths)
        for index, game in enumerate(exp.WINDOWS)
    }
    records = exp.validate_repaired_experts(repo_root, specs)
    assert set(records) == {"sp80", "dc22", "wa30", "sb26"}
    assert all(row["valid"] for row in records.values())


def test_stored_windows_double_rebuild_and_match_historical_counts() -> None:
    """SCENARIO-ARC-WMTE-10012-A2-PAIR-QUALIFICATION: replay every added window twice."""
    repo_root = Path(__file__).resolve().parents[2]
    if not exp.resolve_environment_files(repo_root).is_dir():
        pytest.skip("public environment_files are unavailable")
    windows = {
        game: exp.rebuild_registered_window(repo_root, game, index)
        for index, game in enumerate(exp.ADDED_WINDOWS)
    }
    assert all(window.exact_match for window in windows.values())
    assert all(window.historical_count_match for window in windows.values())
    assert all(window.window_sha256 == window.second_rebuild_sha256 for window in windows.values())
