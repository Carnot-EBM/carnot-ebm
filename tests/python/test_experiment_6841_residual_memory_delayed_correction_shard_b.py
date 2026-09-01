"""Tests for residual-memory delayed-correction shard B.

Spec refs: REQ-CL-6841, SCENARIO-CL-6841-PRECONDITIONS,
SCENARIO-CL-6841-SHARD-DISJOINTNESS, SCENARIO-CL-6841-REVEAL-TIMING,
SCENARIO-CL-6841-DELAYED-CREDIT, SCENARIO-CL-6841-STALE-REPLACEMENT,
SCENARIO-CL-6841-JOINT-ERROR-CORRECTION, SCENARIO-CL-6841-ROLLBACK,
SCENARIO-CL-6841-CHECKPOINT-RESTART, and SCENARIO-CL-6841-METRICS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from carnot import experiment_6841_residual_memory_delayed_correction_shard_b as exp
import scripts.adversarial_verify as adversarial_verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = exp.source_paths_for_root(REPO_ROOT)


@pytest.fixture(scope="module")
def sources() -> dict[str, dict]:
    """SCENARIO-CL-6841-PRECONDITIONS reuses frozen upstream artifacts."""

    return exp.load_sources(SOURCE_PATHS)


@pytest.fixture(scope="module")
def artifact(sources: dict[str, dict], tmp_path_factory: pytest.TempPathFactory) -> dict:
    """REQ-CL-6841 builds one deterministic shard B artifact."""

    return exp.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        state_root=tmp_path_factory.mktemp("exp6841-state"),
        run_date="20260901",
        duration_s=0.25,
    )


def test_req_cl_6841_spec_precedes_implementation() -> None:
    """REQ-CL-6841 declares the paths, fields, and scenario anchors."""

    spec = (REPO_ROOT / exp.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = spec.split("## REQ-CL-6841", 1)[1]
    for requirement_id in exp.OPEN_SPEC_IDS[1:]:
        assert requirement_id in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for path in (exp.MODULE_RELATIVE_PATH, exp.SCRIPT_RELATIVE_PATH, exp.RESULT_RELATIVE_PATH):
        assert path.as_posix() in section


def test_scenario_cl_6841_preconditions_accept_sources(sources: dict[str, dict]) -> None:
    """SCENARIO-CL-6841-PRECONDITIONS accepts complete frozen inputs."""

    summary = exp.check_preconditions(sources, SOURCE_PATHS)
    assert summary["passed"] is True
    assert summary["failed_checks"] == []
    assert all(row["passed"] for row in summary["checks"])


@pytest.mark.parametrize(
    ("fault", "failed_check"),
    [
        ("kernel", "residual_memory_kernel_ready_score"),
        ("assigned_order", "complete_assigned_orders"),
        ("source_hash", "source_artifact_hashes"),
        ("outcome", "exact_later_outcomes"),
        ("delayed_headroom", "delayed_correction_headroom"),
        ("overlap", "no_exp6840_identity_overlap"),
    ],
)
def test_scenario_cl_6841_preconditions_fail_closed(
    sources: dict[str, dict],
    tmp_path: Path,
    fault: str,
    failed_check: str,
) -> None:
    """SCENARIO-CL-6841-PRECONDITIONS records blocked gate details."""

    changed = deepcopy(sources)
    paths = dict(SOURCE_PATHS)
    if fault == "kernel":
        changed["exp6839"]["residual_memory_kernel_ready_score"] = 0.0
    elif fault == "assigned_order":
        changed["exp6827"]["rows"] = [
            row for row in changed["exp6827"]["rows"] if row.get("order_id") != "order_5"
        ]
    elif fault == "source_hash":
        paths["exp6839"] = tmp_path / "missing.json"
    elif fault == "outcome":
        event_id = exp.select_assigned_events(changed["exp6827"])[0]["row_id"]
        for row in changed["exp6827"]["rows"]:
            if row["row_id"] == event_id:
                row["outcome_identity"] = None
                break
    elif fault == "delayed_headroom":
        changed["exp6827"]["headroom_metrics"]["stale_pressure_recovery_count"] = 0
        changed["exp6827"]["headroom_metrics"]["conflict_event_count"] = 0
    else:
        overlap_id = next(
            row["row_id"]
            for row in changed["exp6827"]["rows"]
            if row.get("order_id") == "order_1" and row.get("counterfactual_applicable") is True
        )
        for row in changed["exp6827"]["rows"]:
            if row.get("order_id") == "order_4" and row.get("counterfactual_applicable") is True:
                row["row_id"] = overlap_id
                break

    summary = exp.check_preconditions(changed, paths)
    failed = next(row for row in summary["checks"] if row["check"] == failed_check)
    assert summary["passed"] is False
    assert failed["passed"] is False
    assert "observed" in failed

    blocked = exp.build_artifact(
        changed,
        source_paths=paths,
        state_root=tmp_path,
        run_date="20260901",
        duration_s=0.25,
    )
    assert blocked["status"] == exp.BLOCKED_STATUS
    assert blocked["rows"] == []
    assert blocked["csl_shard_b_complete_score"] == 0.0
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith(exp.BLOCKED_STATUS)
    assert failed_check in blocked["gate_check_summary"]["failed_checks"]


def test_scenario_cl_6841_shard_disjointness_and_cell_labels(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6841-SHARD-DISJOINTNESS selects only shard B row IDs."""

    events = exp.select_assigned_events(sources["exp6827"])
    prior_ids = exp.select_exp6840_reference_event_ids(sources["exp6827"])
    cell_counts = exp.delayed_correction_cell_counts(events)

    assert len(events) == exp.EXPECTED_ASSIGNED_EVENT_COUNT
    assert {exp.order_index(event["order_id"]) for event in events} == {3, 4}
    assert set(event["row_id"] for event in events).isdisjoint(prior_ids)
    assert cell_counts["stale_evidence"] > 0
    assert cell_counts["joint_error_evidence"] > 0
    assert cell_counts["delayed_evidence"] > 0
    assert exp.selected_event_digest(events) == exp.EXPECTED_EVENT_ID_DIGEST


def test_scenario_cl_6841_reveal_timing_and_delayed_credit(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6841-REVEAL-TIMING and DELAYED-CREDIT freeze decisions."""

    event = next(
        row
        for row in exp.select_assigned_events(sources["exp6827"])
        if exp.correction_family(row) == "delayed_evidence"
    )
    state = exp.CorrectionState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    proposal = exp.freeze_decision(event, state, seed=exp.RANDOM_SEEDS[0], event_sequence_index=7)
    repeated = exp.freeze_decision(event, state, seed=exp.RANDOM_SEEDS[0], event_sequence_index=7)
    changed_seed = exp.freeze_decision(
        event,
        state,
        seed=exp.RANDOM_SEEDS[1],
        event_sequence_index=7,
    )

    assert proposal == repeated
    assert proposal["decision_hash"] != changed_seed["decision_hash"]
    assert proposal["decision_frozen_before_outcome_reveal"] is True
    assert set(proposal["pre_reveal_input_keys"]).isdisjoint(exp.OUTCOME_FIELD_DENYLIST)
    assert "outcome_identity" not in proposal["decision_material"]

    outcome = exp.reveal_exact_outcome(event, event_sequence_index=7)
    credit = exp.credit_for_decision(event, exp.VERIFIED_RESIDUAL_ARM, proposal, outcome)
    assert outcome["outcome_revealed_after_decision"] is True
    assert credit["signed_credit"] == pytest.approx(
        outcome["signed_direction"] * proposal["memory_dose"]
    )
    assert credit["reveal_delay_events"] >= exp.MIN_REVEAL_DELAY_EVENTS
    assert credit["correction_latency_events"] == credit["reveal_delay_events"]
    assert exp.bounded_dose(10.0) == 1.0
    assert exp.bounded_dose(-10.0) == -1.0


def test_scenario_cl_6841_stale_replacement_uses_exact_receipt(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6841-STALE-REPLACEMENT expires stale memory first."""

    event = next(
        row
        for row in exp.select_assigned_events(sources["exp6827"])
        if exp.correction_family(row) == "stale_evidence"
    )
    state = exp.CorrectionState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    state.inject_stale_record(event, event_sequence_index=1)
    proposal = exp.freeze_decision(event, state, seed=exp.RANDOM_SEEDS[0], event_sequence_index=20)
    outcome = exp.reveal_exact_outcome(event, event_sequence_index=20)
    credit = exp.credit_for_decision(event, exp.VERIFIED_RESIDUAL_ARM, proposal, outcome)
    transition = state.apply_update(
        event,
        proposal,
        outcome,
        credit,
        event_sequence_index=20,
        seed=exp.RANDOM_SEEDS[0],
    )

    assert transition["reason"] in {"stale_replaced", "admitted"}
    assert transition["expired_entries"]
    assert transition["committed_entries"]
    assert "expired" in transition["transition_statuses"]
    assert "committed" in transition["transition_statuses"]
    assert all(
        record["key"] not in state.active_record_keys() for record in transition["expired_entries"]
    )


def test_scenario_cl_6841_joint_error_correction_revises_two_components(
    sources: dict[str, dict],
) -> None:
    """SCENARIO-CL-6841-JOINT-ERROR-CORRECTION revises both wrong parts."""

    event = next(
        row
        for row in exp.select_assigned_events(sources["exp6827"])
        if exp.correction_family(row) == "joint_error_evidence"
    )
    state = exp.CorrectionState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    state.inject_joint_error_records(event, event_sequence_index=3)
    proposal = exp.freeze_decision(event, state, seed=exp.RANDOM_SEEDS[0], event_sequence_index=12)
    outcome = exp.reveal_exact_outcome(event, event_sequence_index=12)
    credit = exp.credit_for_decision(event, exp.VERIFIED_RESIDUAL_ARM, proposal, outcome)
    transition = state.apply_update(
        event,
        proposal,
        outcome,
        credit,
        event_sequence_index=12,
        seed=exp.RANDOM_SEEDS[0],
    )

    assert transition["reason"] == "joint_error_revised"
    assert len(transition["revised_entries"]) == 2
    assert {row["component"] for row in transition["revised_entries"]} == {
        "authority_a",
        "authority_b",
    }
    assert "revised" in transition["transition_statuses"]
    assert transition["cross_arm_state_read"] is False
    assert all(
        row["new_signed_direction"] == outcome["signed_direction"]
        for row in transition["revised_entries"]
    )


def test_scenario_cl_6841_defensive_paths_fail_closed(
    sources: dict[str, dict],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6841-DELAYED-CREDIT rejects bad authority and bad dose."""

    events = exp.select_assigned_events(sources["exp6827"])
    event = next(row for row in events if exp.correction_family(row) == "delayed_evidence")
    joint_event = next(
        row for row in events if exp.correction_family(row) == "joint_error_evidence"
    )
    state = exp.CorrectionState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    proposal = exp.freeze_decision(event, state, seed=exp.RANDOM_SEEDS[0], event_sequence_index=9)
    outcome = exp.reveal_exact_outcome(event, event_sequence_index=9)
    credit = exp.credit_for_decision(event, exp.VERIFIED_RESIDUAL_ARM, proposal, outcome)

    missing = state.apply_update(
        {**event, "action_identity": None},
        proposal,
        {**outcome, "outcome_identity": None},
        credit,
        event_sequence_index=9,
        seed=exp.RANDOM_SEEDS[0],
    )
    assert missing["reason"] == "missing_update_authority"

    dose_state = exp.CorrectionState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    bad_dose = dose_state.apply_update(
        event,
        {**proposal, "memory_dose": 2.0},
        outcome,
        credit,
        event_sequence_index=9,
        seed=exp.RANDOM_SEEDS[0],
    )
    assert bad_dose["reason"] == "dose_out_of_bounds"

    duplicate_state = exp.CorrectionState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    first = duplicate_state.apply_update(
        event,
        proposal,
        outcome,
        credit,
        event_sequence_index=9,
        seed=exp.RANDOM_SEEDS[0],
    )
    duplicate = duplicate_state.apply_update(
        event,
        proposal,
        outcome,
        credit,
        event_sequence_index=9,
        seed=exp.RANDOM_SEEDS[0],
    )
    assert first["reason"] != "duplicate_update"
    assert duplicate["reason"] == "duplicate_update"

    joint_state = exp.CorrectionState.for_arm(exp.VERIFIED_RESIDUAL_ARM)
    joint_proposal = exp.freeze_decision(
        joint_event,
        joint_state,
        seed=exp.RANDOM_SEEDS[0],
        event_sequence_index=11,
    )
    joint_outcome = exp.reveal_exact_outcome(joint_event, event_sequence_index=11)
    joint_credit = exp.credit_for_decision(
        joint_event,
        exp.VERIFIED_RESIDUAL_ARM,
        joint_proposal,
        joint_outcome,
    )
    replacement = joint_state.apply_update(
        joint_event,
        joint_proposal,
        joint_outcome,
        joint_credit,
        event_sequence_index=11,
        seed=exp.RANDOM_SEEDS[0],
    )
    assert replacement["reason"] == "joint_error_revised"
    assert replacement["committed_entries"]

    assert exp.CorrectionState.for_arm(exp.NO_MEMORY_ARM).inject_stale_record(
        event,
        event_sequence_index=1,
    ) == {"loaded": False, "reason": "non_mutating_arm"}
    assert exp.CorrectionState.for_arm(exp.READ_ONLY_ARM).inject_joint_error_records(
        event,
        event_sequence_index=1,
    ) == {"loaded": False, "reason": "non_mutating_arm"}

    immutable_state = exp.CorrectionState.for_arm(exp.READ_ONLY_ARM)
    immutable_state.capacity_budget = 1
    immutable_state.records.append(
        {
            "key": "bootstrap:second",
            "source_event_row_id": "bootstrap",
            "source_family": "control",
            "case_type": "safe_control",
            "signed_direction": 1,
            "stored_dose": exp.MIN_STORED_DOSE,
            "event_sequence_index": -2,
            "immutable": True,
            "component": "bootstrap",
            "exact_outcome_hash": "bootstrap",
        }
    )
    assert immutable_state._evict_if_needed() == []

    list_path = tmp_path / "list.json"
    bad_path = tmp_path / "bad.json"
    missing_path = tmp_path / "missing.json"
    list_path.write_text("[]", encoding="utf-8")
    bad_path.write_text("{", encoding="utf-8")
    loaded = exp.load_sources({"list": list_path, "bad": bad_path, "missing": missing_path})
    assert loaded["list"] == {"_load_error": "not_object"}
    assert loaded["bad"] == {"_load_error": "JSONDecodeError"}
    assert loaded["missing"] == {"_load_error": "FileNotFoundError"}
    assert exp.select_assigned_events({"rows": "not-a-list"}) == []
    assert exp.select_exp6840_reference_event_ids({"rows": "not-a-list"}) == set()

    malformed_split = deepcopy(sources)
    malformed_split["exp6827"]["split_manifest"]["by_family"]["bad"] = []
    assert exp.check_preconditions(malformed_split, SOURCE_PATHS)["passed"] is True
    assert exp._transition_statuses([], [], [], [], rolled_back=True) == ["rolled_back"]


def test_scenario_cl_6841_rollback_and_checkpoint_restart_are_exact(
    sources: dict[str, dict],
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6841-ROLLBACK and CHECKPOINT-RESTART preserve bytes."""

    events = exp.select_assigned_events(sources["exp6827"])[:60]
    run = exp.run_comparison(
        events,
        state_root=tmp_path / "live",
        exercise_restart=True,
        exercise_rollback=True,
    )
    clean = exp.run_comparison(
        events,
        state_root=tmp_path / "clean",
        exercise_restart=False,
        exercise_rollback=False,
    )
    manifest = exp.checkpoint_manifest_for_run(run, clean.final_state_hash)

    assert run.checkpoint_receipt["bytes_identity"] is True
    assert run.rollback_receipt["restored_parent_bytes"] is True
    assert run.rollback_receipt["rolled_back"] is True
    assert manifest["matches_clean_replay"] is True
    assert manifest["rollback_restored_parent_bytes"] is True

    persisted = tmp_path / "live" / "checkpoint.json"
    recovered = exp.load_persisted_states(persisted)
    assert exp.combined_state_hash(recovered) == run.final_state_hash


def test_scenario_cl_6841_metrics_and_writer_are_row_complete(
    artifact: dict,
    sources: dict[str, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CL-6841-METRICS validates complete row-derived fields."""

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["csl_shard_b_complete_score"] == 1.0
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] in {"null", "positive", "partial"}
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    floor = adversarial_verify.duration_floor_for_artifact(artifact)
    assert floor is not None
    assert floor["reason"] == "deterministic_verifier"
    assert exp.validate_artifact(artifact) == []

    expected_rows = exp.EXPECTED_ASSIGNED_EVENT_COUNT * len(exp.ARM_NAMES) * len(exp.RANDOM_SEEDS)
    assert len(artifact["rows"]) == expected_rows
    assert len(artifact["memory_state_transitions"]) == expected_rows
    assert artifact["split_manifest"]["assigned_order_indices"] == [3, 4]
    assert artifact["split_manifest"]["no_pooling_with_exp6840"] is True
    assert not artifact["source_artifact_hashes"].get("exp6840")

    for row in artifact["rows"]:
        assert exp.REQUIRED_ROW_FIELDS <= set(row)
        assert row["decision_frozen_before_outcome_reveal"] is True
        assert row["outcome_revealed_after_decision"] is True
        assert row["capacity_budget"] == exp.CAPACITY_BUDGET
        assert row["held_future_metric"]["is_held_future"] == (row["split"] == "held_future")
        assert row["correction_latency_events"] >= 0

    assert any(
        "rejected" in row["transition_statuses"] for row in artifact["memory_state_transitions"]
    )
    assert any(
        "revised" in row["transition_statuses"] for row in artifact["memory_state_transitions"]
    )
    assert any(
        "expired" in row["transition_statuses"] for row in artifact["memory_state_transitions"]
    )
    assert any(
        "committed" in row["transition_statuses"] for row in artifact["memory_state_transitions"]
    )
    assert any(
        "rolled_back" in row["transition_statuses"] for row in artifact["memory_state_transitions"]
    )

    for summary_name in (
        "held_future_results",
        "delayed_correction_results",
        "correction_latency_results",
        "residual_error_results",
        "negative_transfer_results",
    ):
        for arm_summary in artifact[summary_name].values():
            assert arm_summary["by_family_and_order"]
            assert all("order_" in key for key in arm_summary["by_family_and_order"])

    for arm, result in artifact["negative_transfer_results"].items():
        assert result["wins"] + result["ties"] + result["losses"] == result["held_future_rows"]
        if arm == exp.NO_MEMORY_ARM:
            assert result["negative_transfer_count"] == 0

    changed = deepcopy(artifact)
    changed.pop("rows")
    assert "required field set mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["field_principles"].pop("rows")
    assert "field_principles coverage mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["inference_substrate"] = "LLM"
    assert "inference_substrate mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["continuous_self_learning_task"] = False
    assert "continuous_self_learning_task must be true" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["verifier_is_oracle"] = True
    assert "verifier_is_oracle must be false" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["verdict_class"] = "invented"
    assert "verdict_class outside closed enum" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["honest_verdict"] = "blocked: no prefix"
    assert "honest_verdict lacks complete_ prefix" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"].pop()
    assert "complete row count mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["rows"][0].pop("row_id")
    assert "row field coverage mismatch" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["csl_shard_b_complete_score"] = 0.0
    assert "complete artifact missing shard score" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["status"] = exp.BLOCKED_STATUS
    assert "blocked artifact must not expose rows" in exp.validate_artifact(changed)

    result_path = tmp_path / "artifact.json"
    exp.write_artifact(result_path, artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert exp.main(["--validate", "--result-path", str(result_path)]) == 0

    with pytest.raises(ValueError, match="required field set mismatch"):
        exp.write_artifact(tmp_path / "bad.json", {"field_principles": {}})

    bad_validate_path = tmp_path / "bad-validate.json"
    bad_validate_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="blocked artifact must not expose rows"):
        exp.main(["--validate", "--result-path", str(bad_validate_path)])

    original_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced build error"])
    with pytest.raises(ValueError, match="forced build error"):
        exp.build_artifact(
            sources,
            source_paths=SOURCE_PATHS,
            state_root=tmp_path / "forced-build",
            run_date="20260901",
            duration_s=0.25,
        )
    monkeypatch.setattr(exp, "validate_artifact", original_validate)

    temp_artifact = exp.build_artifact(
        sources,
        source_paths=SOURCE_PATHS,
        state_root=None,
        run_date="20260901",
        duration_s=0.25,
    )
    assert temp_artifact["checkpoint_manifest"]["matches_clean_replay"] is True

    generated_path = tmp_path / "generated.json"
    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: artifact)
    assert exp.main(["--date", "20260901", "--result-path", str(generated_path)]) == 0
    assert json.loads(generated_path.read_text(encoding="utf-8")) == artifact

    monkeypatch.setattr(exp, "build_artifact", lambda *args, **kwargs: changed)
    with pytest.raises(ValueError, match="blocked artifact must not expose rows"):
        exp.main(["--date", "20260901", "--result-path", str(tmp_path / "bad-main.json")])

    script = REPO_ROOT / exp.SCRIPT_RELATIVE_PATH
    monkeypatch.setattr(
        "sys.argv",
        [script.name, "--validate", "--result-path", str(result_path)],
    )
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
