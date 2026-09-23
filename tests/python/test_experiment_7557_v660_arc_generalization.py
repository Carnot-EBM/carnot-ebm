"""Tests for REQ-ARC-WMTE-7557 held-out-game ARC analysis."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7557_v660_arc_generalization as analysis


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str) -> dict[str, object]:
    return {
        "name": name,
        "command_argv": [name],
        "scope": "fixture",
        "exit_code": 0,
        "passed": True,
        "timed_out": False,
        "duration_s": 0.01,
        "log_sha256": "sha256:" + "1" * 64,
    }


def _receipts() -> list[dict[str, object]]:
    return [_receipt(name) for name in analysis.REQUIRED_RECEIPT_NAMES]


@pytest.fixture(scope="module")
def upstream() -> dict[str, object]:
    return analysis.load_json(ROOT / analysis.UPSTREAM_PATH)


@pytest.fixture(scope="module")
def reduced(upstream: dict[str, object]) -> dict[str, object]:
    checks = analysis.collect_preconditions(ROOT)
    assert all(row["passed"] for row in checks)
    return analysis.reduce_evidence(ROOT, upstream)


def test_exact_upstream_gate_and_missing_path(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-UPSTREAM-GATE."""

    checks = analysis.collect_preconditions(ROOT)
    assert analysis.select_failed_precondition(checks) is None
    missing = analysis.collect_preconditions(tmp_path)
    failure = analysis.select_failed_precondition(missing)
    assert failure["path"] == analysis.UPSTREAM_PATH.as_posix()
    assert failure["observed"] is None


def test_mutated_gate_and_raw_hash_fail_exactly(
    tmp_path: Path, upstream: dict[str, object]
) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-UPSTREAM-GATE."""

    target = tmp_path / analysis.UPSTREAM_PATH
    target.parent.mkdir(parents=True)
    changed = deepcopy(upstream)
    changed["corrected_arc_ready_score"] = 0
    target.write_text(json.dumps(changed), encoding="utf-8")
    failure = analysis.select_failed_precondition(
        analysis.collect_preconditions(tmp_path, require_worktree_files=False)
    )
    assert failure["artifact_field"] == "corrected_arc_ready_score"
    raw = tmp_path / analysis.TELEMETRY_PATH
    raw.parent.mkdir(parents=True)
    raw.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_hash_mismatch"):
        analysis.reduce_evidence(tmp_path, upstream)


def test_causal_endpoint_is_unavailable_not_false(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-ENDPOINT-IDENTIFIABILITY."""

    endpoints = reduced["endpoint_identifiability"]
    assert endpoints["bounded_frame_change"]["positive_count"] == 35
    assert endpoints["bounded_frame_change"]["causal_efficacy"] is False
    assert endpoints["plan_linked_execution"]["available"] is False
    assert endpoints["plan_linked_execution"]["observed"] is None
    assert endpoints["plan_linked_execution"]["missing_joins"] == [
        "accepted_world_model",
        "executed_plan",
        "plan_to_action",
        "attributable_progress",
    ]
    assert endpoints["actual_level_progress"]["positive_count"] == 0
    assert reduced["outcome_counts"]["unidentifiable"] == 35


def test_support_floors_suppress_fit_and_oracle(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-SUPPORT-FLOORS."""

    assert reduced["support_counts"] == {
        "opportunities": 11813,
        "attempts": 35,
        "useful_attempts": 0,
        "useless_attempts": 0,
        "useful_games": 0,
        "useless_games": 0,
    }
    gates = {row["check"]: row for row in reduced["support_gate_results"]}
    assert gates["opportunity_floor"]["passed"] is True
    assert all(not gates[name]["passed"] for name in gates if name != "opportunity_floor")
    assert reduced["numeric_gate_quality_claim"] is False
    assert reduced["gate_fit"] is None
    assert reduced["gate_ready_to_ship"] is False
    oracle = reduced["analysis_only_oracle"]
    assert oracle["endpoint_identifiable"] is False
    assert oracle["optimistic_tokens_avoidable"] is None


def test_supervisor_rows_and_per_game_costs(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-ORACLE-AND-SUPERVISOR."""

    arms = {row["arm"]: row for row in reduced["supervisor_arm_rows"]}
    assert set(arms) == set(analysis.SUPERVISOR_ARMS)
    assert arms["force_exploration_diversity"]["fired"] == 33
    assert arms["force_exploration_diversity"]["helped"] == 0
    assert arms["force_exploration_diversity"]["games_fired"] == 12
    assert all(row["helped"] == 0 for row in arms.values())
    assert reduced["stagnations_unredirected"] == 0
    assert reduced["successor_hypothesis"] is None
    games = reduced["per_game_results"]
    assert len(games) == 12
    assert sum(row["attempts"] for row in games) == 35
    assert sum(row["completion_tokens"] for row in games) == 33 * 4096
    assert sum(row["generation_wall_time_s"] for row in games) > 0
    assert all(row["plan_linked_useful_attempts"] is None for row in games)
    assert all(row["solve_provenance"] == "live_agent_self_discovery" for row in games)


def test_terminal_artifact_is_ready_complete_null(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-TERMINAL-FEASIBILITY."""

    artifact = analysis.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=analysis.collect_preconditions(ROOT),
        validation_receipts=_receipts(),
        duration_s=1.25,
        phase_spans=[{"phase": "fixture", "start_s": 0.0, "end_s": 1.25}],
    )
    assert artifact["arc_analysis_complete_score"] == 1
    assert type(artifact["arc_analysis_complete_score"]) is int
    assert (
        artifact["honest_verdict"] == "complete_null_feasibility_only_causal_endpoint_unavailable"
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["positive_claim"] is False
    assert artifact["no_headroom"] is False
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == analysis.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["production_defaults_changed"] is False
    assert analysis.validate_artifact(artifact, require_terminal=True) == []


def test_blocked_artifact_names_failed_field() -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-UPSTREAM-GATE."""

    checks = [
        analysis.precondition_row(
            "corrected_arc_ready",
            analysis.UPSTREAM_PATH.as_posix(),
            "corrected_arc_ready_score",
            1,
            None,
        )
    ]
    artifact = analysis.build_blocked_artifact("20260923", checks, duration_s=0.1)
    assert artifact["honest_verdict"] == "complete_blocked_corrected_arc_not_ready"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["arc_analysis_complete_score"] == 0
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["gate_check_summary"]["first_failure"]["artifact_field"] == (
        "corrected_arc_ready_score"
    )
    assert analysis.validate_artifact(artifact) == []


def test_mutations_fail_and_independent_reduction_agrees(
    reduced: dict[str, object],
) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-TERMINAL-FEASIBILITY."""

    artifact = analysis.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=[],
        validation_receipts=_receipts(),
        duration_s=1.0,
        phase_spans=[],
    )
    replay = analysis.independent_reduce(artifact)
    assert replay["arc_analysis_complete_score"] == 1
    assert replay["verdict_class"] == "null"
    assert replay["opportunities"] == 11813
    assert replay["attempts"] == 35
    changed = deepcopy(artifact)
    changed["arc_analysis_complete_score"] = True
    changed["positive_claim"] = True
    changed["gate_ready_to_ship"] = True
    changed["rows"] = []
    errors = analysis.validate_artifact(changed, require_terminal=True)
    assert "analysis_complete_score_invalid" in errors
    assert "positive_claim_invalid" in errors
    assert "retrospective_ship_claim" in errors
    assert "row_reduction_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_io_progress_parsing_and_cold_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], reduced: dict[str, object]
) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-TERMINAL-FEASIBILITY."""

    analysis.progress(0.0, "fixture", "boundary", unit=1)
    assert "phase=fixture" in capsys.readouterr().out
    path = tmp_path / "nested/value.json"
    analysis.atomic_json(path, {"ok": True})
    assert analysis.load_json(path) == {"ok": True}
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        analysis.load_json(path)
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\n{"ok": true}\n', encoding="utf-8")
    assert analysis.load_jsonl(jsonl) == [{"ok": True}]
    jsonl.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        analysis.load_jsonl(jsonl)
    malformed = tmp_path / analysis.UPSTREAM_PATH
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{", encoding="utf-8")
    checks = analysis.collect_preconditions(tmp_path, require_worktree_files=False)
    assert checks[0]["observed"] == "malformed_json"
    with pytest.raises(ValueError, match="raw_path_missing"):
        analysis.reduce_evidence(tmp_path, {})

    artifact = analysis.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=[],
        validation_receipts=_receipts(),
        duration_s=1.0,
        phase_spans=[],
    )
    replay_path = tmp_path / "replay.json"
    analysis.atomic_json(replay_path, artifact)
    assert analysis.cold_replay(replay_path)["verdict_class"] == "null"
    artifact["positive_claim"] = True
    analysis.atomic_json(replay_path, artifact)
    with pytest.raises(ValueError, match="cold_replay_invalid"):
        analysis.cold_replay(replay_path)
    commands = analysis._terminal_commands(ROOT, replay_path)
    assert [row.name for row in commands] == list(analysis.TERMINAL_CHECK_NAMES)
    span = analysis._phase_span("fixture", 2.0, 1.0)
    assert span["phase"] == "fixture"
    args = analysis.parse_args(["--date", "20260923", "--output", str(path)])
    assert args.output == path


def test_identifiable_helper_and_all_reduction_guards(
    reduced: dict[str, object],
) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-ENDPOINT-IDENTIFIABILITY."""

    endpoint = analysis._endpoint_summary(
        [{"content_bytes": 4, "planned": True, "level_up_progress": True}]
    )
    assert endpoint["plan_linked_execution"]["available"] is True
    assert endpoint["plan_linked_execution"]["observed"] == 0
    arms, empty = analysis._supervisor_rows(
        [
            {
                "seam": "supervisor_arm_selection",
                "chosen_arm": "allow_reinduction",
                "episode_id": "e1",
                "game_id": "g1",
                "level_before": 0,
            },
            {
                "record_type": "episode_end",
                "episode_id": "e1",
                "level_end": 1,
            },
            {"seam": "supervisor_arm_selection", "chosen_arm": "no_redirect"},
        ]
    )
    assert next(row for row in arms if row["arm"] == "allow_reinduction")["helped"] == 1
    assert empty == 1
    changed = deepcopy(reduced)
    changed["support_counts"]["opportunities"] = 0
    changed["support_counts"]["attempts"] = 0
    changed["attempt_rows"] = []
    changed["endpoint_identifiability"]["plan_linked_execution"] = {
        "available": True,
        "observed": 0,
    }
    changed["numeric_gate_quality_claim"] = True
    changed["gate_fit"] = {}
    changed["supervisor_arm_rows"] = [{"arm": "bad", "fired": 0, "helped": 1}]
    changed["gate_ready_to_ship"] = True
    changed["solve_provenance"]["new_level_solve_claimed"] = True
    errors = analysis.validate_reduction(changed)
    assert len(errors) == 8


def test_all_artifact_schema_guards(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7557; SCENARIO-ARC-WMTE-7557-TERMINAL-FEASIBILITY."""

    base = analysis.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=[],
        validation_receipts=_receipts(),
        duration_s=1.0,
        phase_spans=[],
    )
    changed = deepcopy(base)
    changed.update(
        experiment_id="wrong",
        milestone="wrong",
        honest_verdict="unfinished",
        verdict_class="unknown",
        arc_analysis_complete_score=True,
        MODEL_SPECS=[{}],
        model_invoked=True,
        invocation_counts={},
        planned_inference_substrate_class="wrong",
        execution_venue="host_cpu",
        positive_claim=True,
        gate_ready_to_ship=True,
        inference_substrate="wrong",
        inference_substrate_class="wrong",
        numeric_gate_quality_claim=True,
    )
    changed["field_principles"] = {}
    changed["acceptance_gate_results"][0]["principle"] = ""
    changed["support_counts"] = {"opportunities": 0, "attempts": 0}
    changed["endpoint_identifiability"]["plan_linked_execution"]["available"] = True
    errors = analysis.validate_artifact(changed, require_terminal=True)
    expected = {
        "experiment_id_mismatch",
        "experiment_identity_mismatch",
        "terminal_prefix_missing",
        "verdict_class_invalid",
        "analysis_complete_score_invalid",
        "current_model_specs_not_empty",
        "current_model_invoked",
        "current_invocation_counts_nonzero",
        "planned_substrate_class_mismatch",
        "execution_venue_invalid",
        "field_principles_incomplete",
        "gate_principles_incomplete",
        "scientific_verdict_mismatch",
        "inference_substrate_mismatch",
        "inference_substrate_class_mismatch",
        "support_reduction_mismatch",
        "unsupported_numeric_claim",
        "causal_endpoint_misclassified",
    }
    assert expected <= set(errors)

    blocked = analysis.build_blocked_artifact(
        "20260923",
        [analysis.precondition_row("x", "x", "x", 1, 0)],
        duration_s=0.1,
    )
    blocked.update(
        arc_analysis_complete_score=1,
        honest_verdict="complete_wrong",
        inference_substrate_class="wrong",
    )
    blocked["gate_check_summary"]["first_failure"] = None
    blocked_errors = analysis.validate_artifact(blocked)
    assert "blocked_classification_mismatch" in blocked_errors
    assert "blocked_substrate_mismatch" in blocked_errors
    assert "blocked_gate_summary_missing" in blocked_errors

    incomplete = analysis.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=[],
        validation_receipts=[],
        duration_s=1.0,
        phase_spans=[],
    )
    zero_errors = analysis.validate_artifact(incomplete, require_terminal=True)
    assert "required_validation_failed" in zero_errors
    assert "terminal_readiness_mismatch" in zero_errors
    incomplete["arc_analysis_complete_score"] = 1
    incomplete["reproducibility_checksum"] = analysis.reproducibility_checksum(incomplete)
    incomplete_errors = analysis.validate_artifact(incomplete, require_terminal=True)
    assert "analysis_complete_score_mismatch" in incomplete_errors
    assert "required_validation_failed" in incomplete_errors
