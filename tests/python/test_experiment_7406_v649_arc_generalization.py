"""Spec-linked tests for the bounded live ARC generalization panel.

The tests use small durable journals. They do not load a model or run a game.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7406_v649_arc_generalization as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


def _registry() -> dict:
    return {
        "games": [
            {
                "game": game,
                "levels_reproduced": level,
                "reproducibility": "reproduced",
                "mechanic_class": f"mechanic-{game}",
            }
            for game, level in zip(exp.TARGET_GAMES, (9, 6, 6), strict=True)
        ]
    }


def _references() -> dict[str, dict]:
    return {
        game: {
            "game": game,
            "comparator_class": "hand_tuned_historical_noncausal",
            "actions": actions,
            "levels": levels,
            "solve_provenance": "development_proxy",
        }
        for game, actions, levels in (
            ("bp35", 57, 2),
            ("cn04", 63, 3),
            ("dc22", 132, 3),
        )
    }


def _schedule() -> list[dict]:
    selection = exp.freeze_panel(_registry(), adaptered_games=set(exp.TARGET_GAMES))
    return exp.build_schedule(selection["games"])


def _append_complete_episode(
    checkpoint_dir: Path,
    episode: dict,
    *,
    levels: tuple[int, ...] = (0, 0),
    requests: int = 1,
    terminal: bool = True,
) -> None:
    exp.append_episode_event(
        checkpoint_dir,
        episode,
        "policy_entered",
        {
            "factory": "make_carnot_agent",
            "policy_class": "E3AgentPolicy",
            "denied_paths": list(exp.WITHHELD_INPUTS),
            "elapsed_s": 0.1,
        },
    )
    for call_index in range(requests):
        exp.append_episode_event(
            checkpoint_dir,
            episode,
            "request_attempted",
            {
                "call_index": call_index,
                "requested_max_tokens": exp.MAX_NEW_TOKENS,
                "request_sha256": f"sha256:{call_index:064x}",
                "elapsed_s": 1.0 + call_index,
            },
        )
        exp.append_episode_event(
            checkpoint_dir,
            episode,
            "generation_completed",
            {
                "call_index": call_index,
                "completion_tokens": 4,
                "transport_completed": True,
                "usable_answer": True,
                "raw_generation": "tool call",
                "elapsed_s": 1.5 + call_index,
            },
        )
    for action_index, level in enumerate(levels, start=1):
        exp.append_episode_event(
            checkpoint_dir,
            episode,
            "environment_action_completed",
            {
                "action_index": action_index,
                "action": 1,
                "data": None,
                "level": level,
                "elapsed_s": 2.0 + action_index,
            },
        )
    exp.append_episode_event(
        checkpoint_dir,
        episode,
        "induction_attempt",
        {
            "attempt_index": 0,
            "induction_id": f"sha256:{episode['execution_order'] + 100:064x}",
            "engaged": True,
            "source_authenticated": True,
            "tool_firing_observed": False,
            "elapsed_s": 4.5,
        },
    )
    if terminal:
        exp.append_episode_event(
            checkpoint_dir,
            episode,
            "episode_terminal",
            {
                "disposition": "complete",
                "error": None,
                "cpu_time_s": 0.5,
                "gpu_time_s": None,
                "gpu_time_measurement": "unavailable",
                "trajectory_supervisor": {
                    "mode": "applied",
                    "arms_enabled": list(exp.CURATED_ARMS),
                    "redirects": [],
                },
                "tool_dispatch_result_rows": [],
                "resumed_policy_actions": [],
                "elapsed_s": 5.0,
            },
        )


def _passing_receipts() -> list[dict]:
    names = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        *exp.REQUIRED_E2E_NAMES,
        *exp.REQUIRED_TERMINAL_NAMES,
    )
    return [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "command_argv": [name],
            "scope": "test",
            "duration_s": 0.1,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": f"sha256:{index:064x}",
        }
        for index, name in enumerate(names, start=1)
    ]


def _runtime_receipt() -> dict:
    return {
        "task_linked_cuda_execution": True,
        "server_pid": 123,
        "server_pid_start_tick": 456,
        "offload_layers_requested": 999,
        "offload_layers_observed": 65,
        "lease_owner": {"task_id": exp.EXPERIMENT_ID},
        "lease_release": {"released": True},
        "gpu_uuid": "GPU-test",
    }


def _artifact(
    tmp_path: Path, *, progress: bool = False, receipts: list[dict] | None = None
) -> dict:
    schedule = _schedule()
    checkpoints = tmp_path / "checkpoints"
    for index, episode in enumerate(schedule):
        levels = (0, 1) if progress and index == 0 else (0, 0)
        _append_complete_episode(checkpoints, episode, levels=levels)
    panel = exp.reduce_durable_panel(schedule, checkpoints, _references())
    cumulative = exp.build_cumulative_induction_ledger(
        {
            "historical_ledger_authenticated": True,
            "rows": [
                {
                    "induction_id": f"sha256:{index:064x}",
                    "source_authenticated": True,
                    "engaged": True,
                }
                for index in range(10)
            ],
        },
        panel["per_game_results"],
    )
    return exp.build_terminal_artifact(
        started_at_utc="2026-09-19T00:00:00+00:00",
        ended_at_utc="2026-09-19T00:10:00+00:00",
        duration_s=600.0,
        preconditions=[
            exp.gate_row(
                "dependency",
                "precondition",
                1,
                1,
                upstream=exp.EXP7398_PATH.as_posix(),
                artifact_field="arc_checkpoint_ready_score",
            )
        ],
        source_hashes={"input": {"sha256": f"sha256:{1:064x}"}},
        selection={"passed": True, "games": list(exp.TARGET_GAMES)},
        panel=panel,
        cumulative_ledger=cumulative,
        model_specs=[{"hf_id": exp.MODEL_ID, "sha256": f"sha256:{2:064x}"}],
        runtime_receipt=_runtime_receipt(),
        invocation_counts={
            **exp.ZERO_INVOCATION_COUNTS,
            "model_loads_attempted": 1,
            "model_loads_completed": 1,
            "generation_calls_attempted": 6,
            "generation_calls_completed": 6,
            "usable_answers": 6,
        },
        validation_receipts=_passing_receipts() if receipts is None else receipts,
        phase_spans=[
            {
                "phase": "live_panel",
                "started_elapsed_s": 10.0,
                "ended_elapsed_s": 500.0,
                "duration_s": 490.0,
                "completed_units": 6,
                "checkpoint_at_utc": "2026-09-19T00:09:00+00:00",
            }
        ],
    )


def test_req_7406_dependency_authenticates_fields_and_recorded_code_hashes(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: dependency fields and code bytes must match."""

    paths = {}
    recorded = {}
    for index, relative in enumerate(exp.EXP7398_CODE_PATHS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}", encoding="utf-8")
        paths[relative.as_posix()] = path
        recorded[relative.as_posix()] = {"sha256": exp.sha256_file(path)}
    upstream = {
        "experiment_id": "exp7398-arc-checkpoint",
        "arc_checkpoint_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "source_artifact_hashes": recorded,
    }
    result = exp.check_checkpoint_dependency(upstream, paths)
    assert result["passed"] is True
    assert all(row["passed"] for row in result["checks"])

    paths[exp.EXP7398_CODE_PATHS[0].as_posix()].write_text("changed", encoding="utf-8")
    changed = exp.check_checkpoint_dependency(upstream, paths)
    assert changed["passed"] is False
    assert changed["gate_check_summary"]["first_failure"]["artifact_field"].endswith(".sha256")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("arc_checkpoint_ready_score", 0),
        ("verdict_class", "disqualified"),
        ("flagged_adversarial", True),
    ),
)
def test_scenario_7406_dependency_rejects_each_structured_gate(
    tmp_path: Path, field: str, value: object
) -> None:
    """SCENARIO-ARC-WMTE-7406-DEPENDENCY: each unsafe field blocks."""

    upstream = {
        "experiment_id": "exp7398-arc-checkpoint",
        "arc_checkpoint_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "source_artifact_hashes": {},
    }
    upstream[field] = value
    result = exp.check_checkpoint_dependency(upstream, {})
    assert result["passed"] is False
    assert result["gate_check_summary"]["first_failure"] is not None


def test_req_7406_freezes_six_label_blind_units_with_fixed_budgets() -> None:
    """REQ-ARC-WMTE-7406: three games and two seeds are sealed before outcomes."""

    selection = exp.freeze_panel(_registry(), adaptered_games=set(exp.TARGET_GAMES))
    schedule = exp.build_schedule(selection["games"])
    assert selection["passed"] is True
    assert selection["selection_used_current_outcomes"] is False
    assert selection["game_source_read"] is False
    assert len(schedule) == 6
    assert schedule[0]["sentinel"] is True
    assert sum(row["sentinel"] for row in schedule) == 1
    assert {row["game"] for row in schedule} == set(exp.TARGET_GAMES)
    assert {row["seed"] for row in schedule} == set(exp.EPISODE_SEEDS)
    assert all(row["action_limit"] == 128 for row in schedule)
    assert all(row["completion_limit"] == 2 for row in schedule)
    assert all(row["max_new_tokens_per_call"] == 256 for row in schedule)
    assert all(row["episode_work_limit_s"] == 240 for row in schedule)
    assert all(set(row["withheld_inputs"]) == set(exp.WITHHELD_INPUTS) for row in schedule)


def test_scenario_7406_sentinel_failure_censors_first_and_leaves_rest_unstarted(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7406-SENTINEL: no request stops the panel."""

    schedule = _schedule()
    first = schedule[0]
    exp.append_episode_event(
        tmp_path,
        first,
        "policy_entered",
        {
            "factory": "make_carnot_agent",
            "policy_class": "E3AgentPolicy",
            "denied_paths": list(exp.WITHHELD_INPUTS),
            "elapsed_s": 0.1,
        },
    )
    panel = exp.reduce_durable_panel(schedule, tmp_path, _references())
    assert panel["sentinel_passed"] is False
    assert panel["per_game_results"][0]["disposition"] == "censored_sentinel"
    assert all(row["disposition"] == "unstarted" for row in panel["per_game_results"][1:])
    assert panel["attempted_units"] == 1
    assert panel["censored_units"] == 1
    assert panel["unstarted_units"] == 5
    assert panel["arc_generalization_capture_complete_score"] == 1


def test_scenario_7406_durable_panel_recovers_complete_and_inflight_rows(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7406-DURABLE-PANEL: journals survive a missing summary."""

    schedule = _schedule()
    for episode in schedule[:-1]:
        _append_complete_episode(tmp_path, episode)
    _append_complete_episode(tmp_path, schedule[-1], terminal=False)
    panel = exp.reduce_durable_panel(schedule, tmp_path, _references())
    assert panel["sentinel_passed"] is True
    assert panel["planned_units"] == 6
    assert panel["attempted_units"] == 6
    assert panel["completed_units"] == 5
    assert panel["censored_units"] == 1
    assert panel["unstarted_units"] == 0
    assert panel["per_game_results"][-1]["disposition"] == "censored_timeout"
    assert panel["arc_generalization_capture_complete_score"] == 1


def test_scenario_7406_reduction_recomputes_progress_costs_and_event_equalities(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7406-REDUCTION: raw events control all row counts."""

    schedule = _schedule()
    for index, episode in enumerate(schedule):
        _append_complete_episode(tmp_path, episode, levels=(0, 1) if index == 0 else (0, 0))
    panel = exp.reduce_durable_panel(schedule, tmp_path, _references())
    first = panel["per_game_results"][0]
    assert first["actions_to_progress"] == 2
    assert first["actions_to_progress_censored"] is False
    assert first["elapsed_s_to_progress"] == 4.0
    assert panel["per_game_results"][1]["actions_to_progress_censored"] is True
    assert panel["per_game_results"][1]["actions_to_progress_upper_bound"] == 2
    assert first["generation_calls_attempted"] == 1
    assert first["generation_calls_completed"] == 1
    assert first["raw_generation_receipts"][0]["raw_generation"] == "tool call"
    assert first["historical_reference"]["comparator_class"].endswith("noncausal")
    assert first["new_solve_credit"] is False
    assert first["solve_provenance"] == "live_agent_self_discovery"
    assert not panel["accounting_failures"]


def test_scenario_7406_reduction_detects_budget_and_post_sentinel_activity(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7406-REDUCTION: event contradictions disqualify capture."""

    schedule = _schedule()
    exp.append_episode_event(
        tmp_path,
        schedule[0],
        "policy_entered",
        {"factory": "wrong", "policy_class": "wrong", "denied_paths": [], "elapsed_s": 0.1},
    )
    _append_complete_episode(tmp_path, schedule[1], requests=2)
    panel = exp.reduce_durable_panel(schedule, tmp_path, _references())
    assert panel["arc_generalization_capture_complete_score"] == 0
    assert "later_episode_started_after_failed_sentinel" in panel["accounting_failures"]
    assert panel["authenticity_failures"]


def test_scenario_7406_reduction_rejects_tampered_journal_and_bad_schedule(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7406-DURABLE-PANEL: journal and roster drift are visible."""

    schedule = _schedule()
    _append_complete_episode(tmp_path, schedule[0])
    path = exp._checkpoint_path(tmp_path, schedule[0])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["state_hash"] = f"sha256:{0:064x}"
    path.write_text(json.dumps(value), encoding="utf-8")
    panel = exp.reduce_durable_panel(schedule[:1], tmp_path, _references())
    assert panel["arc_generalization_capture_complete_score"] == 0
    assert any("state hash mismatch" in row for row in panel["accounting_failures"])
    assert "six_episode_schedule_identity" in panel["accounting_failures"]


def test_scenario_7406_reduction_defends_both_disposition_equalities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7406-REDUCTION: contradictory row flags remain terminal."""

    original = exp._episode_row

    def contradictory(*args: object, **kwargs: object) -> dict:
        row = original(*args, **kwargs)
        row.update({"completed": True, "censored": True})
        return row

    monkeypatch.setattr(exp, "_episode_row", contradictory)
    panel = exp.reduce_durable_panel(_schedule(), tmp_path, _references())
    assert "planned_disposition_equality" in panel["accounting_failures"]
    assert "attempted_disposition_equality" in panel["accounting_failures"]


def test_req_7406_tool_rows_keep_no_firing_distinct_from_no_demand() -> None:
    """REQ-ARC-WMTE-7406: a no-fire row does not establish absent demand."""

    rows = exp.project_tool_dispatch_results(
        [
            {
                "attempt_index": 0,
                "tool_gap": {"tool_calls_total": 0, "terminated_by": "final_answer"},
            },
            {
                "attempt_index": 1,
                "tool_loop": {
                    "event_rows": [{"parsed_tool": "diff_grids", "dispatch_result": {"ok": True}}]
                },
            },
        ]
    )
    assert rows[0]["tool_demand_interpretation"] == "unknown_no_firing_observed"
    assert rows[0]["invented_attempt"] is False
    assert rows[1]["tool_firing_observed"] is True


def test_req_7406_cumulative_ledger_separates_historical_current_and_duplicates() -> None:
    """REQ-ARC-WMTE-7406: only authentic distinct current inductions extend history."""

    duplicate = f"sha256:{1:064x}"
    historical = {
        "historical_ledger_authenticated": True,
        "rows": [
            {"induction_id": duplicate, "source_authenticated": True, "engaged": True},
            {"induction_id": f"sha256:{2:064x}", "source_authenticated": True, "engaged": True},
        ],
    }
    episodes = [
        {
            "episode_id": "bp35:seed-1",
            "induction_attempt_rows": [
                {
                    "induction_id": duplicate,
                    "source_authenticated": True,
                    "engaged": True,
                },
                {
                    "induction_id": f"sha256:{3:064x}",
                    "source_authenticated": True,
                    "engaged": True,
                },
                {"engaged": False},
            ],
        }
    ]
    ledger = exp.build_cumulative_induction_ledger(historical, episodes)
    assert ledger["historical_authenticated_count"] == 2
    assert ledger["current_authenticated_count"] == 2
    assert ledger["current_observed_attempt_count"] == 3
    assert ledger["current_new_count"] == 1
    assert ledger["current_duplicate_count"] == 1
    assert ledger["current_censored_count"] == 1
    assert ledger["cumulative_total"] == 3

    censored = exp.build_cumulative_induction_ledger(
        {"historical_ledger_authenticated": False, "rows": historical["rows"]},
        [{"episode_id": "x", "induction_attempt_rows": [None]}],
    )
    assert censored["historical_authenticated_count"] == 0
    assert censored["current_censored_count"] == 1


def test_req_7406_complete_no_progress_is_terminal_null(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: a valid no-progress panel is complete null evidence."""

    artifact = _artifact(tmp_path)
    assert artifact["arc_generalization_capture_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_bounded_panel_no_progress"
    assert artifact["promotion_score"] == 0
    assert artifact["official_score"] is None
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate_class"] == "model_bounded_generation"
    assert artifact["execution_venue"] == "host"
    assert artifact["MODEL_SPECS"] == [exp.MODEL_ID]
    assert exp.validate_artifact(artifact) == []


def test_req_7406_progress_remains_descriptive_null_without_promotion(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: observed progress does not authorize a causal claim."""

    artifact = _artifact(tmp_path, progress=True)
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_bounded_generalization_observation"
    assert artifact["redirect_progress_interpretation"] == "descriptive_association_only"
    assert artifact["supervisor_ordering_changed"] is False
    assert artifact["new_solve_claimed"] is False
    assert exp.validate_artifact(artifact) == []


def test_req_7406_missing_terminal_receipts_builds_disqualified_candidate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: pending or failed terminal readers disqualify readiness."""

    receipts = [
        row for row in _passing_receipts() if row["name"] not in exp.REQUIRED_TERMINAL_NAMES
    ]
    artifact = _artifact(tmp_path, receipts=receipts)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["arc_generalization_capture_complete_score"] == 0
    assert artifact["gate_check_summary"]["all_passed"] is False
    assert exp.validate_artifact(artifact) == []


def test_req_7406_blocked_artifact_has_exact_gate_summary_and_no_invocation() -> None:
    """REQ-ARC-WMTE-7406: failed external gates publish terminal blocked evidence."""

    checks = [
        exp.gate_row(
            "checkpoint_ready",
            "precondition",
            1,
            None,
            upstream=exp.EXP7398_PATH.as_posix(),
            artifact_field="arc_checkpoint_ready_score",
        )
    ]
    artifact = exp.build_blocked_artifact(
        preconditions=checks,
        source_hashes={},
        started_at_utc="2026-09-19T00:00:00+00:00",
        ended_at_utc="2026-09-19T00:00:01+00:00",
        duration_s=1.0,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["first_failure"]["observed"] is None
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert exp.validate_artifact(artifact) == []


def test_req_7406_validate_artifact_rejects_checksum_budget_and_unsafe_credit(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-7406: cold validation rejects altered terminal claims."""

    artifact = _artifact(tmp_path)
    bad = deepcopy(artifact)
    bad["promotion_score"] = 1
    bad["new_solve_claimed"] = True
    bad["sample_size_budget"]["planned_units"] = 7
    errors = exp.validate_artifact(bad)
    assert "promotion_score" in errors
    assert "new_solve_claimed" in errors
    assert "sample_size_budget" in errors
    assert "reproducibility_checksum" in errors


def test_req_7406_validate_artifact_exercises_closed_failure_boundaries(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: every closed identity and safety boundary fails cold."""

    artifact = _artifact(tmp_path)

    def errors_for(**updates: object) -> list[str]:
        changed = deepcopy(artifact)
        changed.update(updates)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return exp.validate_artifact(changed)

    assert exp.validate_artifact(None) == ["artifact_not_object"]
    assert "identity" in errors_for(schema="wrong")
    assert "run_date" in errors_for(run_date="20260920")
    assert "verdict_class" in errors_for(verdict_class="unknown")
    assert "honest_verdict" in errors_for(honest_verdict="wrong")
    assert "model_invoked" in errors_for(model_invoked=False)
    assert "MODEL_SPECS" in errors_for(MODEL_SPECS=[])
    assert "inference_substrate_class" in errors_for(inference_substrate_class="wrong")
    assert "execution_venue" in errors_for(execution_venue="host_cpu")
    assert "duration_s" in errors_for(duration_s=1.0)
    assert "rows" in errors_for(rows=[])
    assert "validation_receipts" in errors_for(validation_receipts=[])
    assert "official_score" in errors_for(official_score=1)
    assert "supervisor_ordering_changed" in errors_for(supervisor_ordering_changed=True)
    assert "field_principles" in errors_for(field_principles={})

    load_only = deepcopy(artifact)
    load_only["invocation_counts"]["generation_calls_attempted"] = 0
    load_only["invocation_counts"]["generation_calls_completed"] = 0
    load_only["invocation_counts"]["usable_answers"] = 0
    load_only["inference_substrate_class"] = "model_load_no_generation"
    load_only["duration_s"] = 1.0
    load_only["reproducibility_checksum"] = exp.artifact_checksum(load_only)
    assert "duration_s" in exp.validate_artifact(load_only)

    blocked = exp.build_blocked_artifact(
        preconditions=[
            exp.gate_row("missing", "precondition", 1, None, upstream="x", artifact_field="score")
        ],
        source_hashes={},
        started_at_utc="2026-09-19T00:00:00+00:00",
        ended_at_utc="2026-09-19T00:00:01+00:00",
        duration_s=1.0,
    )
    blocked["rows"] = [{}]
    blocked["gate_check_summary"]["first_failure"] = None
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked)
    assert "blocked_invocations_or_rows" in blocked_errors
    assert "blocked_gate_check_summary" in blocked_errors


def test_req_7406_row_event_reader_detects_each_counter_drift(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: durable events, not summary counters, are authoritative."""

    row = _artifact(tmp_path)["rows"][0]
    mutations = {
        "durable_event_count": row["durable_event_count"] + 1,
        "generation_calls_attempted": row["generation_calls_attempted"] + 1,
        "generation_calls_completed": row["generation_calls_completed"] + 1,
        "actions_observed": row["actions_observed"] + 1,
        "induction_attempt_count": row["induction_attempt_count"] + 1,
    }
    for key, value in mutations.items():
        changed = {**row, key: value}
        assert any(error.endswith(key) for error in exp._row_event_errors(changed))


def test_req_7406_independent_reducer_matches_declared_artifact(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-7406: cold reload recomputes the capture score."""

    artifact = _artifact(tmp_path)
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    reduced = exp.independent_reduce_file(path)
    assert reduced["matches_declared"] is True
    assert reduced["reduced_arc_generalization_capture_complete_score"] == 1


def test_req_7406_validation_plan_is_exact_exp7358_scope(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: Exp7358 freezes the exact Exp7303 affected commands."""

    commands = exp.build_validation_plan(exp.REPO_ROOT, tmp_path)
    assert [row.name for row in commands] == list(validation_scope.REQUIRED_CHECK_NAMES)
    assert exp.validate_validation_plan(exp.REPO_ROOT, commands) == []
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert "-n" in focused.argv and "0" in focused.argv
    assert "--no-cov" in focused.argv
    assert "tests/python" not in focused.argv


def test_req_7406_e2e_and_terminal_commands_use_private_scopes(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: E2E and cold readers have fixed bounded commands."""

    e2e = exp.e2e_command_specs(exp.REPO_ROOT, tmp_path / "e2e")
    assert [row.name for row in e2e] == list(exp.REQUIRED_E2E_NAMES)
    assert any("CARNOT_ARC_DISABLE_INDUCTION=1" in arg for arg in e2e[-1].argv)
    terminal = exp.terminal_command_specs(exp.REPO_ROOT, tmp_path / "candidate.json")
    assert [row.name for row in terminal] == list(exp.REQUIRED_TERMINAL_NAMES)
    assert "--strict" in terminal[-1].argv


def test_req_7406_session_environment_keeps_fixed_policy_and_output_budgets(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: the live environment changes no supervisor arms."""

    env = exp.session_environment(
        {},
        arm="curated_supervisor",
        episode_dir=tmp_path / f"bp35__seed-{exp.EPISODE_SEEDS[0]}",
        gpu_index=0,
        port=8080,
    )
    assert env["CARNOT_FORCE_LIVE"] == "1"
    assert env["CARNOT_ARC_INDUCE_TOOL_TURNS"] == "2"
    assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "256"
    assert env["CARNOT_7406_EPISODE_ID"].startswith("bp35:")
    assert "CARNOT_ARC_SUPERVISOR_ORDER" not in env
    with pytest.raises(ValueError, match="unknown live arm"):
        exp.session_environment(
            {},
            arm="changed_arm",
            episode_dir=tmp_path,
            gpu_index=0,
            port=8080,
        )


def test_scenario_7406_session_environment_enforces_failed_sentinel_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ARC-WMTE-7406-SENTINEL: later environments cannot start."""

    state = exp._LiveState(_schedule(), tmp_path)
    state.abort_after_sentinel = True
    monkeypatch.setattr(exp, "_LIVE_STATE", state)
    with pytest.raises(exp.SentinelAbort, match="sentinel failed"):
        exp.session_environment(
            {},
            arm="curated_supervisor",
            episode_dir=tmp_path / "cn04__seed-1",
            gpu_index=0,
            port=8080,
        )


def test_req_7406_parser_accepts_only_fixed_date_and_roles() -> None:
    """REQ-ARC-WMTE-7406: the thin entrypoint fixes date and child roles."""

    args = exp.parse_args(["--date", exp.RUN_DATE])
    assert args.date == exp.RUN_DATE
    assert args.role == "experiment"
    with pytest.raises(SystemExit):
        exp.parse_args(["--date", "20260920"])


def test_req_7406_gate_row_and_summary_keep_missing_observations() -> None:
    """REQ-ARC-WMTE-7406: gate summaries preserve None and membership checks."""

    member = exp.gate_row(
        "verdict",
        "precondition",
        ["positive", "circular_positive", "null"],
        "null",
        upstream="upstream.json",
        artifact_field="verdict_class",
        operator="in",
    )
    missing = exp.gate_row(
        "field",
        "precondition",
        1,
        None,
        upstream="upstream.json",
        artifact_field="score",
    )
    summary = exp.gate_summary([member, missing])
    assert member["passed"] is True
    assert summary["failed_count"] == 1
    assert summary["first_failure"]["observed"] is None
    identity = exp.gate_row("identity", "test", None, None, operator="is")
    assert identity["passed"] is True
    with pytest.raises(ValueError, match="unsupported gate operator"):
        exp.gate_row("bad", "test", 1, 1, operator=">")


def test_req_7406_checkpoint_reader_rejects_wrong_episode_identity(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: Exp7398 journal integrity remains active."""

    episode = _schedule()[0]
    exp.append_episode_event(tmp_path, episode, "policy_entered", {"elapsed_s": 0.1})
    changed = {**episode, "seed": episode["seed"] + 1}
    with pytest.raises(exp.CheckpointIntegrityError, match="state hash mismatch"):
        exp.read_episode_events(tmp_path, changed)


def test_req_7406_historical_reference_loader_projects_no_solution_data(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: comparators expose counts, never solution actions."""

    paths = {}
    for game in exp.TARGET_GAMES:
        path = tmp_path / f"{game}.json"
        path.write_text(
            json.dumps(
                {
                    "game": game,
                    "reached_level": 2,
                    "solution": [{"action": 1}, {"action": 2}],
                    "solve_provenance": "development_proxy",
                    "mode": "offline",
                }
            ),
            encoding="utf-8",
        )
        paths[game] = path
    references = exp.load_historical_references(paths)
    assert references["bp35"]["actions"] == 2
    assert "solution" not in references["bp35"]
    assert references["bp35"]["causal_use"] is False


def test_req_7406_precondition_collection_authenticates_real_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7406: the real immutable inputs pass before model work."""

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    checks, hashes, registry, references, historical = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert exp.EXP7398_PATH.as_posix() in hashes
    assert {row["game"] for row in registry["games"]} >= set(exp.TARGET_GAMES)
    assert set(references) == set(exp.TARGET_GAMES)
    assert historical["historical_ledger_authenticated"] is True


def test_req_7406_precondition_collection_closes_malformed_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-7406: malformed registry and exclusion YAML never open a gate."""

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

    def malformed(_text: str) -> object:
        raise exp.yaml.YAMLError("malformed")

    monkeypatch.setattr(exp.yaml, "safe_load", malformed)
    checks, _hashes, registry, _references, _historical = exp.collect_preconditions(exp.REPO_ROOT)
    assert registry == {}
    assert (
        next(row for row in checks if row["check"] == "exclusion_manifest_parse")["passed"] is False
    )
    assert next(row for row in checks if row["check"] == "solve_registry_parse")["passed"] is False


def test_req_7406_small_io_and_phase_helpers_are_truthful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7406: helper receipts retain real files and elapsed boundaries."""

    assert exp.load_object(tmp_path / "missing.json") == {}
    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    assert exp.load_object(list_path) == {}
    assert exp.utc_now().endswith("+00:00")
    exp.progress(time.monotonic(), "test", "boundary", unit=1)
    assert "phase=test event=boundary" in capsys.readouterr().out

    session = {"model_loaded": True, "timed_out": False}
    rows = [
        {
            "generation_calls_attempted": 1,
            "generation_calls_completed": 1,
            "raw_generation_receipts": [
                {
                    "transport_completed": True,
                    "usable_answer": True,
                    "error": None,
                }
            ],
        }
    ]
    counts = exp._invocation_counts(session, rows)
    assert counts["model_loads_completed"] == 1
    assert counts["generation_calls_attempted"] == 1
    phase = exp._phase("test", time.monotonic(), time.monotonic() - 1.0, 1)
    assert phase["completed_units"] == 1

    monkeypatch.setattr(exp, "RAW_DIR", Path("raw"))
    monkeypatch.setattr(exp, "TERMINAL_CANDIDATE_PATH", Path("raw/candidate.json"))
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "evidence.json").write_text("{}", encoding="utf-8")
    (raw / "candidate.json").write_text("{}", encoding="utf-8")
    hashes: dict = {}
    exp._hash_raw_evidence(tmp_path, hashes)
    assert "raw/evidence.json" in hashes
    assert "raw/candidate.json" not in hashes

    panel_path = tmp_path / "panel.json"
    exp._write_raw_panel(panel_path, _schedule(), {"per_game_results": [], "planned_units": 6})
    assert json.loads(panel_path.read_text(encoding="utf-8"))["accounting"]["planned_units"] == 6


def test_req_7406_manifest_recursion_and_source_records(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7406: exclusion and source readers keep their exact scope."""

    assert exp._manifest_rejects({"nested": [{"experiment_id": "7406"}]}, exp.EXPERIMENT_ID)
    assert not exp._manifest_rejects([{"experiment_id": "other"}], exp.EXPERIMENT_ID)
    text = tmp_path / "source.txt"
    text.write_text("source", encoding="utf-8")
    text_record = exp._source_record(text, role="test")
    assert text_record["role"] == "test"
    data = tmp_path / "source.json"
    data.write_text(json.dumps({"experiment_id": "x", "verdict_class": "null"}), encoding="utf-8")
    data_record = exp._source_record(data, role="test")
    assert data_record["producer_experiment_id"] == "x"
