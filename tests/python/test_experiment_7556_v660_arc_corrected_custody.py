"""Tests for REQ-ARC-WMTE-7556 corrected B2 evidence custody."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time

import pytest

from carnot import experiment_7556_v660_arc_corrected_custody as custody


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reduced() -> dict[str, object]:
    """Reduce the immutable historical bytes once for the focused suite."""

    source, checks = custody.collect_preconditions(ROOT, pid_probe=lambda _pid: False)
    assert source == ROOT / custody.LOCAL_SOURCE_PATH
    assert all(row["passed"] for row in checks)
    return custody.reduce_corrected_evidence(ROOT, source, pid_probe=lambda _pid: False)


def _receipt(name: str) -> dict[str, object]:
    return {
        "name": name,
        "command_argv": [name],
        "scope": "focused_fixture",
        "exit_code": 0,
        "passed": True,
        "timed_out": False,
        "duration_s": 0.01,
        "log_sha256": "sha256:" + "1" * 64,
    }


def _all_receipts() -> list[dict[str, object]]:
    return [_receipt(name) for name in custody.REQUIRED_RECEIPT_NAMES]


def test_local_source_precedes_documented_external_handoff(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-STABLE-HANDOFF."""

    local = tmp_path / custody.LOCAL_SOURCE_PATH
    external = tmp_path / "external" / custody.LOCAL_SOURCE_PATH
    local.parent.mkdir(parents=True)
    external.parent.mkdir(parents=True)
    local.write_text("{}", encoding="utf-8")
    external.write_text("{}", encoding="utf-8")

    selected = custody.locate_corrected_source(tmp_path, external.parent.parent)

    assert selected == local
    local.unlink()
    assert custody.locate_corrected_source(tmp_path, external.parent.parent) == external
    external.unlink()
    assert custody.locate_corrected_source(tmp_path, external.parent.parent) is None


def test_real_handoff_is_terminal_stable_and_hash_bound(
    reduced: dict[str, object],
) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-STABLE-HANDOFF."""

    handoff = reduced["handoff_receipt"]
    assert isinstance(handoff, dict)
    assert handoff["producer_pid"] == 1_008_021
    assert handoff["producer_exited"] is True
    assert handoff["producer_returncode"] == 0
    assert handoff["session_run_id"] == "induction_gate_telemetry:1008021"
    assert handoff["raw_hashes_match"] is True
    assert handoff["source_stable"] is True
    assert reduced["custody_qualified"] is True
    assert custody.validate_reduction(reduced) == []


def test_active_producer_mutation_fails_closed() -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-STABLE-HANDOFF."""

    source, _checks = custody.collect_preconditions(ROOT, pid_probe=lambda _pid: True)

    assert source == ROOT / custody.LOCAL_SOURCE_PATH
    reduced = custody.reduce_corrected_evidence(ROOT, source, pid_probe=lambda _pid: True)
    assert reduced["custody_qualified"] is False
    assert "producer_still_active" in custody.validate_reduction(reduced)


def test_swapped_session_mutation_fails_closed(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-STABLE-HANDOFF."""

    changed = deepcopy(reduced)
    changed["handoff_receipt"]["session_run_id"] = "induction_gate_telemetry:swapped"

    assert "session_identity_mismatch" in custody.validate_reduction(changed)


def test_wrong_cap_mutation_fails_closed(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-CAP-AND-PROVENANCE."""

    changed = deepcopy(reduced)
    changed["response_custody"]["requested_token_caps"] = [2048]

    assert "corrected_cap_mismatch" in custody.validate_reduction(changed)


def test_response_bytes_reproduce_distribution_and_second_saturation(
    reduced: dict[str, object],
) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-CAP-AND-PROVENANCE."""

    response = reduced["response_custody"]
    assert response["completed_response_count"] == 33
    assert response["completion_token_total"] == 33 * 4096
    assert response["completion_token_histogram"] == {"4096": 33}
    assert response["finish_reason_histogram"] == {"length": 33}
    assert response["content_byte_total"] == 0
    assert response["reasoning_byte_total"] > 0
    assert response["second_saturation_observed"] is True
    assert response["source_distribution_matches"] is True


def test_every_schedule_unit_and_attempt_keeps_its_disposition(
    reduced: dict[str, object],
) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-PER-UNIT-RECONSTRUCTION."""

    rows = reduced["rows"]
    attempts = reduced["attempt_rows"]
    budget = reduced["sample_size_budget"]

    assert len(rows) == 144
    assert len({(row["game"], row["seed"]) for row in rows}) == 144
    assert len(attempts) == 35
    assert budget == {
        "planned": 144,
        "attempted": 33,
        "completed": 33,
        "excluded": 0,
        "failed": 0,
        "censored": 0,
        "unstarted": 111,
    }
    assert reduced["induction_attempt_budget"] == {
        "attempted": 35,
        "completed_responses": 33,
        "failed_responses": 0,
        "censored_responses": 2,
    }
    assert sum(row["response_disposition"] == "completed" for row in attempts) == 33
    assert sum(row["response_disposition"] == "censored_no_response" for row in attempts) == 2
    assert all(row["level_up_progress"] is False for row in attempts)
    assert all(row["frame_change_progress"] is True for row in attempts)
    assert all(row["progress_attribution"] == "incidental_frame_change_only" for row in attempts)


def test_live_policy_controls_and_joins_are_preserved(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-CAP-AND-PROVENANCE."""

    path = reduced["live_policy_path"]
    joins = reduced["join_summary"]

    assert path == {
        "factory": "make_carnot_agent",
        "policy_class": "E3AgentPolicy",
        "adapter_disabled": True,
        "banked_trajectories_disabled": True,
        "stored_engines_disabled": True,
        "game_source_read": False,
    }
    assert joins["attempt_count"] == 35
    assert joins["planned_count"] == 0
    assert joins["verifier_observed_count"] == 0
    assert joins["credited_level_count"] == 0
    assert joins["action_provenance_complete"] is True


def test_missing_action_provenance_mutation_blocks_credit(
    reduced: dict[str, object],
) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-CAP-AND-PROVENANCE."""

    changed = deepcopy(reduced)
    changed["join_summary"]["credited_level_count"] = 1
    changed["join_summary"]["credited_levels_with_action_provenance"] = 0

    assert "credited_level_missing_action_provenance" in custody.validate_reduction(changed)


def test_terminal_artifact_is_ready_but_scientifically_null(
    reduced: dict[str, object],
) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-TERMINAL-NULL."""

    artifact = custody.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=[],
        validation_receipts=_all_receipts(),
        duration_s=1.25,
        phase_spans=[{"phase": "fixture", "start_s": 0.0, "end_s": 1.25}],
    )

    assert artifact["experiment_id"] == "exp7556-arc-corrected-custody"
    assert artifact["milestone"] == "2026.09.660"
    assert artifact["corrected_arc_ready_score"] == 1
    assert type(artifact["corrected_arc_ready_score"]) is int
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_corrected_b2_authenticated_no_efficacy"
    assert artifact["positive_claim"] is False
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert not any(artifact["invocation_counts"].values())
    assert artifact["historical_model_calls"]["generation_calls_attempted"] == 66
    assert artifact["solve_provenance"]["credited_level_count"] == 0
    assert artifact["verifier_is_oracle"] is False
    assert artifact["flagged_adversarial"] is False
    assert custody.validate_artifact(artifact, require_terminal=True) == []


def test_checksum_and_terminal_receipt_mutations_are_rejected(
    reduced: dict[str, object],
) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-TERMINAL-NULL."""

    artifact = custody.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=[],
        validation_receipts=_all_receipts(),
        duration_s=0.5,
        phase_spans=[],
    )
    changed = deepcopy(artifact)
    changed["rows"][0]["disposition"] = "invented"
    assert "reproducibility_checksum_mismatch" in custody.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["validation_receipts"][-1]["exit_code"] = 4
    errors = custody.validate_artifact(changed, require_terminal=True)
    assert "required_validation_failed" in errors


def test_blocked_artifact_names_exact_missing_upstream(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-STABLE-HANDOFF."""

    source, checks = custody.collect_preconditions(
        tmp_path,
        external_root=tmp_path / "missing-external",
        pid_probe=lambda _pid: False,
    )
    artifact = custody.build_blocked_artifact(
        "20260923", checks, duration_s=0.01, source_path=source
    )

    assert source is None
    assert artifact["honest_verdict"] == "complete_blocked_corrected_b2_not_available"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["corrected_arc_ready_score"] == 0
    failure = artifact["gate_check_summary"]["first_failure"]
    assert failure["upstream"] == "experiment_10008_b2_induction_gate_measurement_v2"
    assert failure["artifact_field"] == "path"
    assert failure["expected"] == "readable_terminal_artifact"
    assert failure["observed"] is None
    assert custody.validate_artifact(artifact, require_terminal=False) == []


def test_json_helpers_and_cli_contract(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-TERMINAL-NULL."""

    path = tmp_path / "nested" / "value.json"
    custody.atomic_json(path, {"b": 2, "a": 1})

    assert custody.load_json(path) == {"a": 1, "b": 2}
    assert custody.sha256_file(path).startswith("sha256:")
    assert custody.canonical_hash({"a": 1}).startswith("sha256:")
    args = custody.parse_args(["--date", "20260923", "--output", str(path)])
    assert args.date == "20260923"
    assert args.output == path
    with pytest.raises(ValueError, match="run_date_mismatch"):
        custody.run_experiment(tmp_path, "20260922", output_path=path)


def test_field_principles_cover_every_terminal_field(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-TERMINAL-NULL."""

    artifact = custody.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=[],
        validation_receipts=_all_receipts(),
        duration_s=0.1,
        phase_spans=[],
    )

    assert set(artifact) - {"field_principles"} <= set(artifact["field_principles"])
    assert all(artifact["field_principles"].values())


def test_defensive_json_process_and_progress_helpers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-STABLE-HANDOFF."""

    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json_object_required"):
        custody.load_json(array_path)

    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text('\n{"ok": true}\n', encoding="utf-8")
    assert custody.load_jsonl(jsonl_path) == [{"ok": True}]
    jsonl_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_object_required"):
        custody.load_jsonl(jsonl_path)

    assert custody._pid_alive(os.getpid()) is True
    assert custody._pid_alive(999_999_999) is False
    assert custody.utc_now().endswith("Z")
    custody.progress(time.monotonic(), "fixture", "boundary", unit=1)
    assert "phase=fixture event=boundary" in capsys.readouterr().out


def test_malformed_present_source_fails_preconditions(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-STABLE-HANDOFF."""

    source = tmp_path / custody.LOCAL_SOURCE_PATH
    source.parent.mkdir(parents=True)
    source.write_text("not-json", encoding="utf-8")

    selected, checks = custody.collect_preconditions(
        tmp_path, tmp_path / "external", pid_probe=lambda _pid: False
    )

    assert selected == source
    status = next(row for row in checks if row["check"] == "source_status_terminal")
    assert status["observed"] is None
    assert status["passed"] is False


def test_low_level_custody_reducers_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-STABLE-HANDOFF."""

    rows, passed = custody._raw_citation_checks({"cited_artifacts": ["bad-row"]}, tmp_path)
    assert rows == []
    assert passed is False

    invocation = custody._invocation_summary(
        [
            {"call_id": "open", "operation": "generation", "state": "attempted"},
            {"call_id": "wrong", "operation": "unknown", "state": "attempted"},
        ]
    )
    assert invocation["generation_calls_in_flight"] == 1
    assert invocation["all_calls_terminal"] is False
    assert set(invocation["incomplete_call_ids"]) == {"open", "wrong"}
    assert custody._event_file("unrelated/path.json", tmp_path) == Path(
        "/__invalid_exp7556_event_path__"
    )


def test_attempt_join_distinguishes_level_up_and_no_progress() -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-PER-UNIT-RECONSTRUCTION."""

    telemetry = [
        {
            "record_type": "induction_attempt",
            "attempt_id": "g:seed-1:induction:1",
            "episode_id": "g:seed-1",
            "game_id": "g",
            "monotonic_timestamp_s": 1.0,
            "step_index": 1,
            "level_up_progress": True,
            "frame_change_progress": True,
            "verifier_result": "accept",
            "planned": True,
        },
        {
            "record_type": "induction_attempt",
            "attempt_id": "g:seed-1:induction:2",
            "episode_id": "g:seed-1",
            "game_id": "g",
            "monotonic_timestamp_s": 2.0,
            "step_index": 3,
            "level_up_progress": False,
            "frame_change_progress": False,
        },
    ]
    actions = [
        {"episode_id": "g:seed-1", "action_index": 2},
        {"episode_id": "g:seed-1", "action_index": 4},
    ]
    response = {
        "episode_id": "g:seed-1",
        "call_index": 0,
        "sha256": "sha256:x",
        "finish_reason": "stop",
        "completion_tokens": 1,
        "prompt_tokens": 2,
        "content_bytes": 3,
        "reasoning_bytes": 0,
    }

    rows = custody._attempt_rows(telemetry, actions, {"g:seed-1": [response, response]})

    assert rows[0]["progress_attribution"] == "attributable_level_up"
    assert rows[1]["progress_attribution"] == "no_progress_observed"
    assert rows[0]["action_provenance"] == "observed_live_actions"


def test_reduction_validator_names_each_boundary(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-CAP-AND-PROVENANCE."""

    mutations = [
        (
            "producer_not_terminal",
            lambda row: row["handoff_receipt"].__setitem__("producer_returncode", 9),
        ),
        (
            "source_or_raw_hash_mismatch",
            lambda row: row["handoff_receipt"].__setitem__("source_stable", False),
        ),
        (
            "historical_call_incomplete",
            lambda row: row["handoff_receipt"].__setitem__("historical_calls_terminal", False),
        ),
        (
            "transport_hash_mismatch",
            lambda row: row["response_custody"].__setitem__("transport_hashes_match", False),
        ),
        (
            "source_distribution_mismatch",
            lambda row: row["response_custody"].__setitem__("source_distribution_matches", False),
        ),
        (
            "second_saturation_missing",
            lambda row: row["response_custody"].__setitem__("second_saturation_observed", False),
        ),
        ("model_identity_mismatch", lambda row: row.__setitem__("model_identity_qualified", False)),
        (
            "live_policy_path_mismatch",
            lambda row: row["live_policy_path"].__setitem__("factory", "other"),
        ),
        ("schedule_reconstruction_mismatch", lambda row: row["rows"].pop()),
        ("attempt_reconstruction_mismatch", lambda row: row["attempt_rows"].pop()),
        (
            "action_provenance_incomplete",
            lambda row: row["join_summary"].__setitem__("action_provenance_complete", False),
        ),
    ]
    for expected, mutate in mutations:
        changed = deepcopy(reduced)
        mutate(changed)
        assert expected in custody.validate_reduction(changed)
        assert "custody_qualified_mismatch" in custody.validate_reduction(changed)


def test_artifact_validator_names_each_boundary(reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-TERMINAL-NULL."""

    valid = custody.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=[],
        validation_receipts=_all_receipts(),
        duration_s=0.2,
        phase_spans=[],
    )
    mutations = [
        ("experiment_id_mismatch", "experiment_id", "wrong"),
        ("experiment_identity_mismatch", "milestone", "wrong"),
        ("terminal_prefix_missing", "honest_verdict", "null"),
        ("verdict_class_invalid", "verdict_class", "unknown"),
        ("corrected_arc_ready_score_invalid", "corrected_arc_ready_score", True),
        ("current_model_specs_not_empty", "MODEL_SPECS", [{"name": "wrong"}]),
        ("current_model_invoked", "model_invoked", True),
        ("current_invocation_counts_nonzero", "invocation_counts", {"loads": 1}),
        ("execution_venue_invalid", "execution_venue", "host_cpu"),
        ("positive_claim_invalid", "positive_claim", True),
        ("inference_substrate_mismatch", "inference_substrate", "wrong"),
        ("inference_substrate_class_mismatch", "inference_substrate_class", "wrong"),
        ("row_count_mismatch", "rows", []),
        ("attempt_count_mismatch", "induction_attempt_rows", []),
        ("scientific_verdict_mismatch", "verdict_class", "positive"),
        ("ready_score_mismatch", "corrected_arc_ready_score", 0),
    ]
    for expected, key, value in mutations:
        changed = deepcopy(valid)
        changed[key] = value
        assert expected in custody.validate_artifact(changed, require_terminal=True)

    changed = deepcopy(valid)
    changed["field_principles"] = {}
    assert "field_principles_incomplete" in custody.validate_artifact(changed)
    changed = deepcopy(valid)
    changed["solve_provenance"]["credited_level_count"] = 1
    assert "solve_provenance_mismatch" in custody.validate_artifact(changed)


def test_blocked_validator_rejects_bad_classification_and_summary() -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-STABLE-HANDOFF."""

    check = custody.precondition_row("missing", "source", "path", "present", None)
    blocked = custody.build_blocked_artifact("20260923", [check], duration_s=0.01, source_path=None)
    changed = deepcopy(blocked)
    changed["corrected_arc_ready_score"] = 1
    assert "blocked_classification_mismatch" in custody.validate_artifact(changed)
    changed = deepcopy(blocked)
    changed["gate_check_summary"] = {}
    assert "blocked_gate_summary_missing" in custody.validate_artifact(changed)


def test_independent_and_cold_replay_paths(tmp_path: Path, reduced: dict[str, object]) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-TERMINAL-NULL."""

    artifact = custody.build_artifact(
        ROOT,
        "20260923",
        reduced,
        preconditions_checked=[],
        validation_receipts=_all_receipts(),
        duration_s=0.2,
        phase_spans=[],
    )
    path = tmp_path / "candidate.json"
    custody.atomic_json(path, artifact)

    replay = custody.cold_replay(path)
    assert replay["corrected_arc_ready_score"] == 1
    assert replay["verdict_class"] == "null"
    beneficial = deepcopy(artifact)
    beneficial["induction_attempt_rows"][0].update(
        {"planned": True, "verifier_result": "accept", "level_up_progress": True}
    )
    assert custody.independent_reduce(beneficial)["benefit_observed"] is True

    artifact["experiment_id"] = "wrong"
    custody.atomic_json(path, artifact)
    with pytest.raises(ValueError, match="cold_replay_invalid"):
        custody.cold_replay(path)


def test_terminal_command_and_phase_span_contract(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7556; SCENARIO-ARC-WMTE-7556-TERMINAL-NULL."""

    commands = custody._terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.name for row in commands] == list(custody.TERMINAL_CHECK_NAMES)
    started = time.monotonic()
    span = custody._phase_span("fixture", started, started)
    assert span["phase"] == "fixture"
    assert span["duration_s"] >= 0.0
    assert custody._gate_summary([{"passed": True}])["all_passed"] is True
