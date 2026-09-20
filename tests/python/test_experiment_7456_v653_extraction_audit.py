"""Tests for REQ-REPORT-7456 and SCENARIO-REPORT-7456-*.

The tests exercise only the new V653 reducer. Shared parsing and validation
helpers keep their existing test coverage in their owning modules.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7456_v653_extraction_audit as audit


def _producer() -> dict:
    return json.loads((audit.REPO_ROOT / audit.EXP7451_PATH).read_text(encoding="utf-8"))


def test_real_sidecars_reproduce_all_planned_cells() -> None:
    """SCENARIO-REPORT-7456-RAW reduces raw sidecars, not producer scores."""

    checks, sources, producer = audit.collect_preconditions(audit.REPO_ROOT)
    assert all(row["passed"] for row in checks)
    evidence = audit.audit_producer(audit.REPO_ROOT, producer)
    assert evidence["errors"] == []
    assert evidence["disposition_counts"] == {
        "complete": 2,
        "empty": 6,
        "unstarted": 96,
    }
    assert len(evidence["rows"]) == 104
    assert evidence["sample_size_budget"] == {
        "planned": 104,
        "attempted": 8,
        "completed": 8,
        "failed": 0,
        "censored": 0,
        "unstarted": 96,
        "development_planned": 8,
        "evaluation_planned": 96,
        "independent_paired_evaluation_units": 48,
        "stop_rule": "audit all eight development calls and all 96 fixed evaluation cells without retry",
    }
    assert evidence["paired_completion_effect"]["pairs"] == 0
    assert evidence["paired_completion_effect"]["estimate"] is None
    assert evidence["paired_token_cost_effect"]["estimate"] is None
    assert sources[audit.EXP7451_PATH.as_posix()]["original_verdict_class"] == "null"
    assert sources[audit.EXP7451_PATH.as_posix()]["original_flagged_adversarial"] is False


def test_empty_introductions_never_receive_factual_recall_credit() -> None:
    """SCENARIO-REPORT-7456-SEMANTICS keeps transport validity separate."""

    evidence = audit.audit_producer(audit.REPO_ROOT, _producer())
    empty_rows = [row for row in evidence["rows"] if row["disposition"] == "empty"]
    assert len(empty_rows) == 6
    assert all(row["factual_recall_credit"] is False for row in empty_rows)
    assert all(row["semantic_ground_truth"] == "unavailable" for row in empty_rows)
    assert {row["source_text_sha256"] for row in empty_rows} == set(
        audit.NONFACTUAL_INTRODUCTION_HASHES
    )
    coverage = {row["metric"]: row for row in evidence["coverage_rows"]}
    assert coverage["real_paragraph_factual_recall"]["denominator"] == 0
    assert coverage["constructed_qualifier_retention"]["denominator"] == 12
    assert coverage["constructed_qualifier_retention"]["authority"] == ("constructed_exact_string")
    assert evidence["synthetic_to_natural_extrapolation"] is False


def test_constructed_mutations_reject_modifier_offset_json_and_source_drift() -> None:
    """SCENARIO-REPORT-7456-MUTATIONS rejects extraction corruption."""

    rows = audit.run_mutation_controls()
    assert {row["attack"] for row in rows} == set(audit.REQUIRED_MUTATIONS)
    assert all(row["passed"] is True for row in rows)
    observed = {row["attack"]: row["observed_errors"] for row in rows}
    assert any("constructed_modifier_missing" in error for error in observed["deleted_modifier"])
    assert "span_bounds" in observed["off_by_one_offset"]
    assert "truncated_json_rejected" in observed["truncated_json"]
    assert "verbatim_not_literal" in observed["absent_source_claim"]


def test_runtime_replay_balances_identity_cleanup_and_lease() -> None:
    """SCENARIO-REPORT-7456-RUNTIME replays producer ownership events."""

    producer = _producer()
    runtime = audit.replay_runtime(producer)
    assert runtime["errors"] == []
    assert runtime["historical_producer_invocation_counts"] == {
        "scope": "historical",
        "counts": producer["invocation_counts"],
    }
    assert runtime["loads_balanced"] is True
    assert runtime["generations_balanced"] is True
    assert runtime["cleanup_verified"] is True
    assert runtime["lease_release_verified"] is True

    missing_terminal = deepcopy(producer)
    missing_terminal["current_invocation_events"].pop()
    assert "unfinished_call:generation-7" in audit.replay_runtime(missing_terminal)["errors"]
    wrong_lease = deepcopy(producer)
    wrong_lease["runner_receipt"]["lease_release"]["lease_id"] = "lease:changed"
    assert "lease_identity_mismatch" in audit.replay_runtime(wrong_lease)["errors"]


def test_fixture_artifact_preserves_null_and_zero_current_calls() -> None:
    """REQ-REPORT-7456 separates audit completion from scientific value."""

    artifact = audit.build_artifact_for_test()
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == audit.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["extraction_audit_complete_score"] == 1
    assert artifact["audited_scientific_disposition"]["verdict_class"] == "null"
    assert artifact["verdict_class"] == "null"
    assert artifact["promotion_score"] == 0
    assert audit.validate_artifact(artifact, verify_source_bytes=False) == []

    changed = deepcopy(artifact)
    changed["rows"][0]["factual_recall_credit"] = True
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    assert "row_reduction_mismatch" in audit.validate_artifact(changed, verify_source_bytes=False)


def test_blocked_artifact_names_absent_current_producer(tmp_path: Path) -> None:
    """REQ-REPORT-7456 reports missing current inference as a blocked branch."""

    checks, sources, producer = audit.collect_preconditions(tmp_path)
    assert producer == {}
    artifact = audit.build_blocked_artifact(checks, sources)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_missing_exp7451_current_inference"
    assert artifact["gate_check_summary"]["path"] == audit.EXP7451_PATH.as_posix()
    assert artifact["gate_check_summary"]["observed"] is None


def test_validation_plan_is_scoped_and_private(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7456-ARTIFACT freezes only affected paths."""

    commands = audit.build_validation_commands(audit.REPO_ROOT, tmp_path)
    assert audit.validate_command_plan(audit.REPO_ROOT, audit.V653_MANIFEST, commands) == []
    assert {command.name for command in commands} == set(
        audit.validation_scope.REQUIRED_CHECK_NAMES
    )
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert "-n" in focused.argv and "--no-cov" in focused.argv
    assert "tests/python" not in focused.argv


def test_cold_modes_and_date_guard(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-REPORT-7456-ARTIFACT exposes fresh bounded readers."""

    artifact = audit.build_artifact_for_test()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.main(["--cold-replay", str(path), "--skip-source-bytes"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    assert audit.main(["--independent-reduce", str(path), "--skip-source-bytes"]) == 0
    assert json.loads(capsys.readouterr().out)["errors"] == []
    with pytest.raises(SystemExit, match="--date must be"):
        audit.run_experiment(audit.REPO_ROOT, "20260919", output_path=tmp_path / "x.json")


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("MODEL_SPECS", ["model"], "current_model_declaration_invalid"),
        ("model_invoked", True, "current_model_declaration_invalid"),
        ("invocation_counts", {}, "current_invocation_counts_invalid"),
        ("inference_substrate_class", "model", "inference_substrate_class_invalid"),
        ("execution_venue", "external", "execution_venue_invalid"),
        ("promotion_score", 1, "promotion_score_invalid"),
        ("extraction_audit_complete_score", 0, "audit_complete_score_mismatch"),
    ],
)
def test_artifact_contract_rejects_declaration_drift(field: str, value: object, error: str) -> None:
    """SCENARIO-REPORT-7456-ARTIFACT fails ordinary field drift closed."""

    artifact = audit.build_artifact_for_test()
    artifact[field] = value
    artifact["reproducibility_checksum"] = audit.reproducibility_checksum(artifact)
    assert error in audit.validate_artifact(artifact, verify_source_bytes=False)


def test_defensive_json_sidecar_and_constructed_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7456-MUTATIONS names malformed evidence shapes."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert audit._load_object(missing) == {}
    assert audit._load_object(malformed) == {}
    assert audit._load_object(sequence) == {}
    assert audit._load_text(missing) == ""

    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    bad = sidecars / f"sha256-{'0' * 64}.json"
    bad.write_text("{", encoding="utf-8")
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    values, references, errors = audit._byte_named_objects(sidecars)
    assert values == [] and references == []
    assert any(error.startswith("sidecar_filename_hash_mismatch") for error in errors)
    assert any(error.startswith("sidecar_json_invalid") for error in errors)

    controls = audit.reduce_constructed_pairs(audit.constructed_qualifier_pairs())
    controls.pop()
    controls[0]["literal_span_reconstruction"] = False
    _rows, errors = audit._constructed_audit(controls)
    assert "constructed_literal_failure:qualifier-01-negation:span" in errors
    assert "constructed_arm_count:qualifier-12-attribution" in errors
    assert "constructed_pair_count_mismatch" in errors


def test_runtime_defensive_failures_are_explicit() -> None:
    """SCENARIO-REPORT-7456-RUNTIME rejects malformed ownership histories."""

    value = audit._runtime_fixture()
    value["current_invocation_events"][0].update(
        {"monotonic_ns": True, "scope": "historical", "operation": "unknown"}
    )
    value["current_invocation_events"].append(
        {
            **value["current_invocation_events"][1],
            "state": "completed",
            "monotonic_ns": 3,
        }
    )
    value["runner_receipt"].update(
        {
            "pid": None,
            "lease_released": False,
            "cleanup": {
                "action": "foreign",
                "bounded": False,
                "leak_free": False,
                "unrelated_process_kill_count_delta": 1,
            },
        }
    )
    errors = audit.replay_runtime(value)["errors"]
    assert "event_time_invalid:generation-0" in errors
    assert "event_identity_mismatch:generation-0" in errors
    assert "operation_invalid:generation-0" in errors
    assert "producer_invocation_counts_mismatch" in errors
    assert "pid_start_tick_identity_invalid" in errors
    assert "lease_release_invalid" in errors
    assert "cleanup_invalid" in errors

    duplicate = audit._runtime_fixture()
    duplicate["current_invocation_events"].extend(deepcopy(duplicate["current_invocation_events"]))
    duplicate["current_invocation_events"][2]["monotonic_ns"] = 3
    duplicate["current_invocation_events"][3]["monotonic_ns"] = 4
    errors = audit.replay_runtime(duplicate)["errors"]
    assert "attempt_count_invalid:generation-0" in errors
    assert "terminal_count_invalid:generation-0" in errors


def test_source_audit_detects_artifact_manifest_and_semantic_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7456-RAW rejects producer and sidecar projection drift."""

    producer = _producer()
    changed = deepcopy(producer)
    changed["development_rows"][0]["raw_reply"] = "changed"
    changed["raw_capture_manifest"].pop()
    changed["raw_capture_manifest"][0]["raw_reply_sha256"] = "sha256:changed"
    changed["semantic_pair_rows"] = []
    evidence = audit.audit_producer(audit.REPO_ROOT, changed)
    assert "development_sidecars_artifact_mismatch" in evidence["errors"]
    assert "raw_capture_manifest_count_mismatch" in evidence["errors"]
    assert any(error.startswith("raw_capture_manifest_missing:") for error in evidence["errors"])
    assert any(
        error.startswith("raw_capture_manifest_hash_mismatch:") for error in evidence["errors"]
    )
    assert "constructed_pair_summary_mismatch" in evidence["errors"]

    original = audit._byte_named_objects

    def changed_sidecars(directory: Path) -> tuple[list[dict], list[dict], list[str]]:
        values, references, errors = original(directory)
        if directory.name == "evaluation":
            return values[:-1], references[:-1], errors
        if directory.name == "events":
            return [], references, errors
        if directory.name == "responses":
            return [], references, errors
        return values, references, errors

    monkeypatch.setattr(audit, "_byte_named_objects", changed_sidecars)
    evidence = audit.audit_producer(audit.REPO_ROOT, producer)
    assert "evaluation_sidecars_artifact_mismatch" in evidence["errors"]
    assert "event_sidecars_artifact_mismatch" in evidence["errors"]
    assert "response_sidecars_artifact_mismatch" in evidence["errors"]


def test_validation_helpers_and_cold_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7456-ARTIFACT checks both terminal and failure paths."""

    assert audit._validation_passed(audit._passing_receipts(candidate=False), candidate=False)
    assert audit._gate_summary([])["all_passed"] is True
    assert audit.cold_replay(tmp_path / "missing.json") == ["candidate_artifact_unreadable"]
    assert {row.spec.name for row in audit._terminal_commands(tmp_path / "x.json")} == set(
        audit.TERMINAL_CHECKS
    )
    monkeypatch.setattr(audit, "validate_command_plan", lambda *_args: ["changed"])
    with pytest.raises(ValueError, match="validation_plan_invalid"):
        audit.build_validation_commands(audit.REPO_ROOT, tmp_path)
    with pytest.raises(SystemExit, match="--date is required"):
        audit.main([])


def test_artifact_validator_names_all_evidence_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7456-ARTIFACT rejects source and summary mutations."""

    base = audit.build_artifact_for_test()
    nonfixture = deepcopy(base)
    nonfixture["fixture_artifact"] = False
    nonfixture["reproducibility_checksum"] = audit.reproducibility_checksum(nonfixture)
    assert audit.validate_artifact(nonfixture, verify_source_bytes=False) == []
    mutations = {
        "required_field_missing:schema": lambda value: value.pop("schema"),
        "artifact_identity_invalid": lambda value: value.update(schema="wrong"),
        "artifact_schedule_invalid": lambda value: value.update(run_date="20260919"),
        "disposition_counts_mismatch": lambda value: value.update(raw_disposition_counts={}),
        "sample_size_budget_mismatch": lambda value: value["sample_size_budget"].update(planned=0),
        "semantic_scope_invalid": lambda value: value.update(
            synthetic_to_natural_extrapolation=True
        ),
        "runtime_audit_invalid": lambda value: value["runtime_audit"].update(
            cleanup_verified=False
        ),
        "source_disposition_laundered": lambda value: value[
            "audited_scientific_disposition"
        ].update(verdict_class="positive"),
        "terminal_disposition_invalid": lambda value: value.update(verdict_class="positive"),
        "mutation_controls_invalid": lambda value: value.update(audit_mutation_rows=[]),
        "reproducibility_checksum_mismatch": lambda value: value.update(
            reproducibility_checksum="sha256:changed"
        ),
    }
    for expected, mutate in mutations.items():
        value = deepcopy(base)
        mutate(value)
        if expected != "reproducibility_checksum_mismatch":
            value["reproducibility_checksum"] = audit.reproducibility_checksum(value)
        assert expected in audit.validate_artifact(value, verify_source_bytes=False)

    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    value = deepcopy(base)
    value["source_artifact_hashes"] = {
        "bad-shape": "not-a-reference",
        "changed": {"path": "source.json", "sha256": "sha256:changed"},
    }
    value["historical_evidence_sidecars"] = [
        "not-a-reference",
        {"path": "source.json", "sha256": "sha256:changed"},
    ]
    value["reproducibility_checksum"] = audit.reproducibility_checksum(value)
    errors = audit.validate_artifact(value, root=tmp_path)
    assert "source_reference_invalid:bad-shape" in errors
    assert "source_hash_mismatch:changed" in errors
    assert "historical_sidecar_reference_invalid" in errors
    assert "historical_sidecar_hash_mismatch:source.json" in errors


def test_independent_replay_detects_projection_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7456-ARTIFACT compares fresh raw reduction."""

    artifact = audit.build_artifact_for_test()
    artifact["raw_disposition_counts"] = {}
    artifact["reproducibility_checksum"] = audit.reproducibility_checksum(artifact)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert "independent_reduction_mismatch:raw_disposition_counts" in audit.independent_replay(path)
