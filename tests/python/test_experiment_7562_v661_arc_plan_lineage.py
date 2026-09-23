"""Tests for REQ-ARC-WMTE-7562 live ARC plan-lineage observation."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import pytest

from carnot import experiment_7562_v661_arc_plan_lineage as exp
from carnot.agentic import arc_decision_telemetry as telemetry
from carnot.agentic.arc_competition_agent import E3AgentPolicy


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _policy(recorder: telemetry.DecisionTelemetryRecorder) -> E3AgentPolicy:
    """Build only the state read by the real planner and plan-consumption seams."""

    policy = E3AgentPolicy.__new__(E3AgentPolicy)
    policy._decision_telemetry = recorder
    policy.two_sided_goal_contract = None
    policy.structured_evidence_memory = None
    policy.phase = "execute"
    policy.plan = []
    policy.pi = 0
    policy.transitions = []
    policy.induction_attempts = []
    policy.proposer = SimpleNamespace(last_generated_tokens=7, last_prompt_tokens=11)
    return policy


class _Verdict:
    accuracy = 1.0
    cell_recall = 1.0
    change_accuracy = 1.0
    change_fidelity = 1.0
    correct_changed_cells = 1
    spurious_changed_cells = 0
    noop_hallucination_rate = 0.0


class _Verifier:
    def score(self, _engine: Any) -> _Verdict:
        return _Verdict()


def _arm(recorder: telemetry.DecisionTelemetryRecorder, policy: E3AgentPolicy) -> str:
    recorder.begin_step(level_before=0, phase="induce")
    recorder.record_induction_decision(
        policy,
        stalled=True,
        won=False,
        decision=(True, "stall"),
        wall_time_s=0.001,
    )
    return recorder._armed_induction_attempt_ids[-1]


def _accepted_plan(
    recorder: telemetry.DecisionTelemetryRecorder,
    policy: E3AgentPolicy,
    *,
    engine_name: str,
    execute: bool = True,
) -> tuple[str, tuple[int, Any]]:
    attempt_id = _arm(recorder, policy)
    engine = SimpleNamespace(name=engine_name)
    recorder.time_world_model_verification(_Verifier(), engine, candidate_source=engine_name)

    def planner(_engine: Any, _done: Any, _start: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [{"action": 1, "data": None}]

    plan = E3AgentPolicy._call_plan_in_model(
        policy,
        planner,
        engine,
        lambda _grid: False,
        [[0]],
        goal_energy_override=lambda _grid: 0.0,
    )
    policy.plan = list(plan)
    policy.pi = 0
    recorder.complete_induction(
        policy,
        {"reason": "stall", "planned": True, "plan_length": 1},
        0.01,
    )
    move = (1, None)
    if execute:
        move = E3AgentPolicy._next_plan_move(policy)
        recorder.record_policy_action(
            policy,
            proposed_move=move,
            selected_move=move,
            level_before=0,
            provenance="execute.plan_step",
        )
    return attempt_id, move


def test_positive_control_joins_real_policy_plan_action_and_level(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7562-POSITIVE-JOIN joins all immutable IDs."""

    path = tmp_path / "positive.jsonl"
    recorder = telemetry.DecisionTelemetryRecorder("sb26", path=path, episode_id="positive")
    policy = _policy(recorder)
    attempt_id, _move = _accepted_plan(recorder, policy, engine_name="accepted-a")
    recorder.begin_policy_step(policy, SimpleNamespace(levels_completed=1))
    recorder.observe_induction_progress(policy, SimpleNamespace(levels_completed=1))
    recorder.finish_episode(level_end=1, actions_used=1)

    reduction = exp.reduce_lineage_rows(_rows(path))
    terminal = reduction["terminal_rows"][0]
    assert reduction["errors"] == []
    assert reduction["positive_join_count"] == 1
    assert terminal["terminal_stage"] == "executed_with_level_progress"
    assert terminal["induction_attempt_id"] == attempt_id
    assert terminal["episode_id"] == "positive"
    assert terminal["model_version"].startswith("positive:model:")
    assert terminal["plan_id"].startswith("positive:plan:")
    assert terminal["executed_action_ids"] == ["positive:action:1"]


def test_rejected_replaced_and_censored_terminal_stages(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7562-REPLACEMENT-AND-CENSORING is exclusive."""

    rejected_path = tmp_path / "rejected.jsonl"
    rejected = telemetry.DecisionTelemetryRecorder("vc33", path=rejected_path)
    rejected_policy = _policy(rejected)
    _arm(rejected, rejected_policy)
    rejected.time_world_model_verification(
        _Verifier(), SimpleNamespace(name="bad"), candidate_source="bad"
    )
    rejected.complete_induction(
        rejected_policy,
        {"planned": False, "skipped": "world_model_accuracy_below_threshold"},
        0.01,
    )
    rejected.finish_episode(level_end=0)
    assert exp.reduce_lineage_rows(_rows(rejected_path))["terminal_stage_counts"] == {
        "verifier_rejected": 1
    }

    replaced_path = tmp_path / "replaced.jsonl"
    replaced = telemetry.DecisionTelemetryRecorder("su15", path=replaced_path)
    replaced_policy = _policy(replaced)
    _accepted_plan(replaced, replaced_policy, engine_name="first", execute=False)
    replaced_policy.pi = 0
    _accepted_plan(replaced, replaced_policy, engine_name="second", execute=False)
    replaced.finish_episode(level_end=0)
    replaced_rows = exp.reduce_lineage_rows(_rows(replaced_path))["terminal_rows"]
    assert replaced_rows[0]["terminal_stage"] == "planned_not_executed"
    assert replaced_rows[0]["closure_reason"] == "model_replaced"
    assert replaced_rows[0]["replaced_by_model_version"] == replaced_rows[1]["model_version"]
    assert replaced_rows[1]["terminal_stage"] == "censored"

    censored_path = tmp_path / "censored.jsonl"
    censored = telemetry.DecisionTelemetryRecorder("g50t", path=censored_path)
    censored_policy = _policy(censored)
    _accepted_plan(censored, censored_policy, engine_name="only")
    censored.finish_episode(level_end=0, actions_used=1)
    terminal = exp.reduce_lineage_rows(_rows(censored_path))["terminal_rows"][0]
    assert terminal["terminal_stage"] == "censored"
    assert terminal["closure_reason"] == "episode_end"


def test_transport_parse_no_plan_and_horizon_stages_are_distinct(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7562 keeps every non-execution terminal stage distinct."""

    cases = (
        ("transport", "proposer_failed", "connection timeout", "transport_failed"),
        ("parse", "proposer_failed", "invalid JSON tool call", "parse_rejected"),
        ("no-plan", "", "", "accepted_no_plan"),
    )
    for label, skipped, note, expected in cases:
        path = tmp_path / f"{label}.jsonl"
        recorder = telemetry.DecisionTelemetryRecorder(label, path=path)
        policy = _policy(recorder)
        _arm(recorder, policy)
        attempt = {"planned": False, "skipped": skipped, "proposer_note": note}
        if not skipped:
            recorder.time_world_model_verification(
                _Verifier(), SimpleNamespace(name=label), candidate_source=label
            )
        recorder.complete_induction(policy, attempt, 0.01)
        recorder.finish_episode(level_end=0)
        terminal = exp.reduce_lineage_rows(_rows(path))["terminal_rows"][0]
        assert terminal["terminal_stage"] == expected

    horizon_path = tmp_path / "horizon.jsonl"
    horizon = telemetry.DecisionTelemetryRecorder("m0r0", path=horizon_path)
    horizon_policy = _policy(horizon)
    _accepted_plan(horizon, horizon_policy, engine_name="horizon")
    for index in range(telemetry.INDUCTION_PROGRESS_WINDOW_ACTIONS - 1):
        horizon.record_policy_action(
            horizon_policy,
            proposed_move=(2, None),
            selected_move=(2, None),
            level_before=0,
            provenance=f"foreign:{index}",
        )
    horizon.finish_episode(level_end=0)
    terminal = exp.reduce_lineage_rows(_rows(horizon_path))["terminal_rows"][0]
    assert terminal["terminal_stage"] == "executed_no_level_progress"
    assert terminal["observation_actions"] == 32
    assert terminal["foreign_action_interleaving"] is True


def test_corrupted_attempt_model_and_plan_ids_fail_independent_join(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7562-NEGATIVE-JOINS rejects three ID mutations."""

    path = tmp_path / "valid.jsonl"
    recorder = telemetry.DecisionTelemetryRecorder("dc22", path=path, episode_id="corrupt")
    policy = _policy(recorder)
    _accepted_plan(recorder, policy, engine_name="accepted")
    recorder.begin_policy_step(policy, SimpleNamespace(levels_completed=1))
    recorder.observe_induction_progress(policy, SimpleNamespace(levels_completed=1))
    recorder.finish_episode(level_end=1)
    rows = _rows(path)
    assert exp.reduce_lineage_rows(rows)["errors"] == []

    expected_errors = {
        "induction_attempt_id": "terminal_attempt_join_missing",
        "model_version": "terminal_model_join_missing",
        "plan_id": "terminal_plan_join_missing",
    }
    for field, expected in expected_errors.items():
        corrupt = deepcopy(rows)
        terminal = next(row for row in corrupt if row.get("record_type") == "plan_lineage_terminal")
        terminal[field] = f"corrupt:{field}"
        assert expected in exp.reduce_lineage_rows(corrupt)["errors"]


def test_frozen_roster_is_twelve_unstarted_registered_rows() -> None:
    """SCENARIO-ARC-WMTE-7562-FROZEN-ROSTER seals but does not execute Exp7570."""

    roster = exp.freeze_arc_roster()
    assert len(roster) == 12
    assert {row["game"] for row in roster} == set(exp.FROZEN_GAMES)
    assert {row["seed"] for row in roster} == set(exp.FROZEN_SEEDS)
    assert all(row["policy_action_limit"] == 600 for row in roster)
    assert all(row["induction_limit"] == 2 for row in roster)
    assert all(row["request_token_ceiling"] == 4096 for row in roster)
    assert all(row["disposition"] == "unstarted_exp7570_only" for row in roster)


def test_preconditions_name_missing_upstream_path_field_and_observation(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7562 blocks without fabricating a missing Exp7557 input."""

    checks = exp.collect_preconditions(tmp_path)
    failed = exp.first_failed_precondition(checks)
    assert failed == {
        "check": "exp7557_upstream_available",
        "upstream": exp.UPSTREAM_PATH.as_posix(),
        "path": exp.UPSTREAM_PATH.as_posix(),
        "artifact_field": "path",
        "expected": "readable_file",
        "observed": None,
        "op": "==",
        "passed": False,
        "principle": "Dependent evidence must exist before measurement.",
    }
    blocked = exp.build_blocked_artifact("20260923", checks, duration_s=0.01)
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"] == {
        **failed,
        "category": "validity",
    }
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["planned_inference_substrate_class"] == "no_model_load"
    assert exp.validate_artifact(blocked, require_terminal=False) == []

    malformed = deepcopy(blocked)
    malformed["plan_lineage_ready_score"] = 1
    malformed["gate_check_summary"]["first_failure"] = None
    errors = exp.validate_artifact(malformed, require_terminal=False)
    assert "blocked_classification_mismatch" in errors
    assert "blocked_gate_summary_missing" in errors


def test_artifact_schema_reduction_and_identifier_validation(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7562 emits complete no-load custody with bare readiness."""

    controls = exp.run_scripted_controls(tmp_path / "controls")
    reduction = exp.reduce_lineage_rows(controls["rows"])
    checks = [
        exp.precondition_row(
            "exp7557_upstream_available",
            exp.UPSTREAM_PATH.as_posix(),
            "honest_verdict",
            "complete_null_feasibility_only_causal_endpoint_unavailable",
            "complete_null_feasibility_only_causal_endpoint_unavailable",
        )
    ]
    artifact = exp.build_artifact(
        "20260923",
        checks=checks,
        controls=controls,
        reduction=reduction,
        validation_receipts=[],
        duration_s=0.1,
        phase_spans=[],
        source_hashes={exp.UPSTREAM_PATH.as_posix(): "sha256:test"},
        terminal=False,
    )
    assert exp.validate_artifact(artifact, require_terminal=False) == []
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["plan_lineage_ready_score"] == 0
    assert artifact["sample_size_budget"] == {
        "planned": 6,
        "attempted": 6,
        "completed": 6,
        "excluded": 0,
        "failed": 0,
        "censored": 2,
        "unstarted": 0,
    }
    assert len(artifact["frozen_arc_roster"]) == 12
    assert artifact["per_game_results"] == []
    assert artifact["solve_provenance"]["new_solve_claimed"] is False

    broken = deepcopy(artifact)
    broken["MODEL_SPECS"] = [{"name": "fabricated"}]
    assert "current_model_specs_not_empty" in exp.validate_artifact(broken, require_terminal=False)

    mutations = (
        ("experiment_id", "wrong", "experiment_id_mismatch"),
        ("milestone", "wrong", "experiment_identity_mismatch"),
        (
            "honest_verdict",
            "complete_partial_terminal_validation_pending",
            "partial_success_prefix_contradiction",
        ),
        ("verdict_class", "wrong", "verdict_class_invalid"),
        ("plan_lineage_ready_score", 2, "plan_lineage_ready_score_invalid"),
        ("model_invoked", True, "current_model_invoked"),
        ("invocation_counts", {}, "current_invocation_counts_nonzero"),
        ("planned_inference_substrate_class", "wrong", "planned_substrate_class_mismatch"),
        ("execution_venue", "host_cpu", "execution_venue_invalid"),
        ("inference_substrate_class", "wrong", "inference_substrate_class_mismatch"),
        ("inference_substrate", "wrong", "inference_substrate_mismatch"),
        ("verifier_is_oracle", False, "fixture_oracle_declaration_missing"),
        ("positive_claim", True, "causal_benefit_overclaim"),
        ("raw_control_row_hash", "wrong", "raw_control_hash_mismatch"),
        ("per_game_results", [{"game": "fake"}], "fixture_promoted_to_game_outcome"),
        (
            "frozen_arc_roster",
            artifact["frozen_arc_roster"][:-1],
            "frozen_roster_mismatch",
        ),
        ("verdict_class", "null", "candidate_classification_mismatch"),
    )
    for field, value, expected in mutations:
        mutated = deepcopy(artifact)
        mutated[field] = value
        assert expected in exp.validate_artifact(mutated, require_terminal=False)

    missing_principle = deepcopy(artifact)
    missing_principle["field_principles"].pop("experiment_id")
    assert "field_principles_incomplete" in exp.validate_artifact(
        missing_principle, require_terminal=False
    )
    missing_gate_principle = deepcopy(artifact)
    missing_gate_principle["acceptance_gate_results"][0]["principle"] = ""
    assert "gate_principles_incomplete" in exp.validate_artifact(
        missing_gate_principle, require_terminal=False
    )
    bad_rows = deepcopy(artifact)
    terminal = next(
        row
        for row in bad_rows["raw_control_rows"]
        if row.get("record_type") == "plan_lineage_terminal"
    )
    terminal["terminal_stage"] = "bad"
    bad_rows["raw_control_row_hash"] = exp.canonical_hash(bad_rows["raw_control_rows"])
    assert "lineage_reduction_invalid" in exp.validate_artifact(bad_rows, require_terminal=False)

    missing_terminal = deepcopy(artifact)
    missing_terminal["raw_control_rows"] = [
        row
        for row in missing_terminal["raw_control_rows"]
        if row.get("record_type") != "plan_lineage_terminal"
        or row.get("terminal_stage") != "parse_rejected"
    ]
    missing_terminal["raw_control_row_hash"] = exp.canonical_hash(
        missing_terminal["raw_control_rows"]
    )
    assert "scripted_attempt_count_mismatch" in exp.validate_artifact(
        missing_terminal, require_terminal=False
    )

    solve_claim = deepcopy(artifact)
    solve_claim["solve_provenance"]["new_solve_claimed"] = True
    assert "fixture_promoted_to_solve" in exp.validate_artifact(solve_claim, require_terminal=False)

    receipts = [
        {"name": name, "passed": True, "exit_code": 0} for name in exp.REQUIRED_RECEIPT_NAMES
    ]
    final = exp.build_artifact(
        "20260923",
        checks=checks,
        controls=controls,
        reduction=reduction,
        validation_receipts=receipts,
        duration_s=0.2,
        phase_spans=[],
        source_hashes={exp.UPSTREAM_PATH.as_posix(): "sha256:test"},
        terminal=True,
    )
    assert exp.validate_artifact(final, require_terminal=True) == []
    final_path = tmp_path / "final.json"
    exp.atomic_json(final_path, final)
    assert exp.cold_replay(final_path)["plan_lineage_ready_score"] == 1

    wrong_final = deepcopy(final)
    wrong_final["verdict_class"] = "null"
    assert "terminal_verdict_class_mismatch" in exp.validate_artifact(
        wrong_final, require_terminal=True
    )

    missing_receipt = deepcopy(final)
    missing_receipt["validation_receipts"] = missing_receipt["validation_receipts"][:-1]
    errors = exp.validate_artifact(missing_receipt, require_terminal=True)
    assert "required_validation_failed" in errors
    assert "terminal_readiness_mismatch" in errors

    zero_score = deepcopy(final)
    zero_score["plan_lineage_ready_score"] = 0
    assert "terminal_readiness_mismatch" in exp.validate_artifact(zero_score, require_terminal=True)

    invalid_path = tmp_path / "invalid.json"
    exp.atomic_json(invalid_path, wrong_final)
    with pytest.raises(ValueError, match="cold_replay_invalid"):
        exp.cold_replay(invalid_path)


def test_thin_cli_parses_read_only_modes() -> None:
    """REQ-ARC-WMTE-7562 keeps the public entrypoint argument-only."""

    args = exp.parse_args(["--date", "20260923", "--validate", "/tmp/candidate.json"])
    assert args.date == "20260923"
    assert args.validate == Path("/tmp/candidate.json")


def test_io_progress_and_precondition_error_paths(tmp_path: Path, capsys) -> None:
    """REQ-ARC-WMTE-7562 keeps utility and malformed-input behavior explicit."""

    started = time.monotonic()
    exp.progress(started, "unit", "boundary", count=1)
    assert "phase=unit event=boundary" in capsys.readouterr().out

    path = tmp_path / "nested" / "value.json"
    exp.atomic_json(path, {"value": 1})
    assert exp.load_json(path) == {"value": 1}
    path.write_text("[]")
    with pytest.raises(ValueError, match="json_object_required"):
        exp.load_json(path)

    malformed = tmp_path / exp.UPSTREAM_PATH
    malformed.parent.mkdir(parents=True)
    malformed.write_text("not-json")
    checks = exp.collect_preconditions(tmp_path)
    assert checks[0]["observed"] == "malformed_json"
    assert exp.first_failed_precondition([{"passed": True}]) is None

    actual_checks = exp.collect_preconditions(exp.REPO_ROOT)
    assert all(row["passed"] is True for row in actual_checks)


def test_reducer_names_invalid_stage_action_level_and_exclusivity(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7562-NEGATIVE-JOINS covers each graph edge."""

    controls = exp.run_scripted_controls(tmp_path / "reduce-errors")
    rows = controls["rows"]
    terminal_index = next(
        index
        for index, row in enumerate(rows)
        if row.get("terminal_stage") == "executed_with_level_progress"
    )

    invalid_stage = deepcopy(rows)
    invalid_stage[terminal_index]["terminal_stage"] = "not-a-stage"
    assert "terminal_stage_invalid" in exp.reduce_lineage_rows(invalid_stage)["errors"]

    missing_action = deepcopy(rows)
    missing_action[terminal_index]["executed_action_ids"] = ["missing-action"]
    errors = exp.reduce_lineage_rows(missing_action)["errors"]
    assert "terminal_action_join_missing" in errors
    assert "terminal_level_transition_join_missing" in errors

    no_action = deepcopy(rows)
    no_action[terminal_index]["executed_action_ids"] = []
    assert "executed_stage_without_action" in exp.reduce_lineage_rows(no_action)["errors"]

    wrong_stage = deepcopy(rows)
    wrong_stage[terminal_index]["terminal_stage"] = "planned_not_executed"
    assert "not_executed_stage_with_action" in exp.reduce_lineage_rows(wrong_stage)["errors"]

    duplicate = deepcopy(rows)
    duplicate.append(deepcopy(duplicate[terminal_index]))
    assert "terminal_stage_not_mutually_exclusive" in exp.reduce_lineage_rows(duplicate)["errors"]


def test_validation_command_plans_are_scoped_and_private(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7562 freezes applicable E2E and terminal readers."""

    e2e = exp.e2e_commands(exp.REPO_ROOT, tmp_path / "e2e")
    assert [row.name for row in e2e] == list(exp.E2E_COMMAND_NAMES)
    assert all("tests/python" in " ".join(row.argv) or row.name.endswith("smoke") for row in e2e)
    assert "CARNOT_ARC_DISABLE_INDUCTION=1" in e2e[-1].argv

    candidate = tmp_path / "candidate.json"
    terminal = exp.terminal_commands(exp.REPO_ROOT, candidate)
    assert [row.name for row in terminal] == list(exp.TERMINAL_COMMAND_NAMES)
    assert all(str(candidate) in row.argv for row in terminal)

    started = time.monotonic()
    span = exp._span("unit", started, started)
    assert span["phase"] == "unit"
    assert span["duration_s"] >= 0.0


def test_lineage_alternate_joins_and_defensive_hooks(tmp_path: Path, monkeypatch) -> None:
    """REQ-ARC-WMTE-7562 keeps alternate joins safe and disabled hooks inert."""

    noop = telemetry.NOOP_RECORDER
    assert noop.record_planner_invocation(None, None, None, 0.0) is None
    assert noop.record_plan_consumption(None, None, 0, None) is None
    assert (
        noop.record_policy_action(
            None,
            proposed_move=None,
            selected_move=None,
            level_before=None,
            provenance=None,
        )
        is None
    )

    recorder = telemetry.DecisionTelemetryRecorder("branches", path=tmp_path / "branches.jsonl")
    selection = SimpleNamespace(
        rows=(),
        selected=SimpleNamespace(engine=SimpleNamespace(name="selected")),
        selected_score=SimpleNamespace(heldout_accuracy=1.0),
    )
    assert (
        recorder.time_world_model_selection(lambda *_args, **_kwargs: selection, [], [])
        is selection
    )

    policy = _policy(recorder)
    policy.plan = [{"action": 1, "data": None}]
    signature = recorder._plan_signature(policy.plan)
    recorder._planner_rows["planner-fallback"] = [
        {
            "plan_signature": signature,
            "plan_id": "branches:plan:manual",
            "model_version": "branches:model:manual",
        }
    ]
    recorder._complete_plan_lineage(policy, "planner-fallback", {"planned": True}, [], [])
    assert recorder._lineages["planner-fallback"]["model_version"] == "branches:model:manual"

    recorder._complete_plan_lineage(policy, "registered", {"planned": True}, [], [])
    assert recorder._lineages["registered"]["plan_id"].startswith("branches:plan:")
    recorder._complete_plan_lineage(policy, "rejected", {"planned": False}, [], ["reject"])
    assert recorder._lineages["rejected"]["terminal_stage"] == "verifier_rejected"
    assert (
        recorder._failure_terminal_stage(
            {"skipped": "exception", "exception": "HTTP connection failed"}
        )
        == "transport_failed"
    )
    assert (
        recorder._failure_terminal_stage({"skipped": "exception", "exception": "invalid response"})
        == "parse_rejected"
    )

    lineage = {"induction_attempt_id": "invalid", "terminal_stage": None}
    errors_before = recorder.error_count
    recorder._emit_lineage_terminal(lineage, "not-a-stage", "test")
    assert recorder.error_count == errors_before + 1
    lineage["terminal_stage"] = "censored"
    recorder._emit_lineage_terminal(lineage, "censored", "duplicate")

    class _BadInt:
        def __int__(self) -> int:
            raise ValueError("bad index")

    recorder._active_lineage_id = "registered"
    recorder.record_plan_consumption(policy, policy.plan, _BadInt(), policy.plan[0])
    assert recorder._pending_plan_action is None

    def fail_sanitize(_value: Any) -> Any:
        raise ValueError("bad value")

    monkeypatch.setattr(telemetry, "_sanitize", fail_sanitize)
    assert telemetry.DecisionTelemetryRecorder._plan_signature(policy.plan).startswith("[")
    assert telemetry.DecisionTelemetryRecorder._same_move((1, None), (1, None)) is False
    errors_before = recorder.error_count
    recorder.record_policy_action(
        policy,
        proposed_move=(1, None),
        selected_move=(1, None),
        level_before=0,
        provenance="defensive",
    )
    assert recorder.error_count == errors_before + 1

    monkeypatch.setattr(
        recorder, "_current_attempt_id", lambda: (_ for _ in ()).throw(ValueError())
    )
    recorder.record_planner_invocation(policy, object(), [], 0.0)
    assert recorder.error_count == errors_before + 2
