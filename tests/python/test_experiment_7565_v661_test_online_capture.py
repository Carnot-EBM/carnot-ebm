"""Tests for the V661 test and online capture.

Spec refs: REQ-VERIFY-7565 and SCENARIO-VERIFY-7565-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7548_v660_capture_runner as capture
from carnot import experiment_7565_v661_test_online_capture as exp


def _pilot() -> dict[str, object]:
    return {
        "experiment_id": "exp7563-v661-native-pilot",
        "native_tool_ready_score": 1,
        "eval_capture_feasible_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "MODEL_SPECS": [exp.MODEL_ID],
        "source_artifact_hashes": {
            exp.PROTOCOL_PATH.as_posix(): "sha256:protocol",
            exp.RUNNER_PATH.as_posix(): "sha256:runner",
            exp.PILOT_MODULE_PATH.as_posix(): "sha256:pilot-module",
            exp.PILOT_WRAPPER_PATH.as_posix(): "sha256:pilot-wrapper",
        },
        "model_specs": [{"model_sha256": "sha256:model"}],
        "role_manifest": {
            "sealed_capture": {
                "evaluation": {
                    "roles": ["test", "online"],
                    "group_count": 240,
                    "forward_count": 1440,
                    "role_schedule_hashes": {
                        "test": "sha256:test",
                        "online": "sha256:online",
                    },
                }
            }
        },
    }


def _bound_hashes() -> dict[str, str]:
    return {
        exp.PROTOCOL_PATH.as_posix(): "sha256:protocol",
        exp.RUNNER_PATH.as_posix(): "sha256:runner",
        exp.PILOT_MODULE_PATH.as_posix(): "sha256:pilot-module",
        exp.PILOT_WRAPPER_PATH.as_posix(): "sha256:pilot-wrapper",
    }


def _fixture_groups() -> list[dict[str, object]]:
    groups = [capture._fixture_group(role, 0) for role in exp.ROLE_COUNTS]
    for group in groups:
        group["donor_role"] = group["role"]
    return groups


def _fixture_schedule() -> list[dict[str, object]]:
    counts = {role: 1 for role in exp.ROLE_COUNTS}
    return exp.compile_evaluation_schedule(_fixture_groups(), expected_role_counts=counts)


def _fixture_rows(schedule: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = [capture._scripted_native_row(row, index) for index, row in enumerate(schedule)]
    for row in rows:
        row.update(
            {
                "reply": "",
                "reply_utf8_bytes": 0,
                "prompt_encoding": "utf-8",
                "reply_encoding": "utf-8",
                "complete_source_window": row["prompt"],
                "complete_response_window": "",
                "model_identity_sha256": "sha256:model-identity",
            }
        )
    return rows


def _fixture_labels() -> list[dict[str, object]]:
    return [
        {
            "component_hash": group["component_hash"],
            "role": group["role"],
            "label": index,
            "arrival_rank": index,
            "label_provenance": "injected_tool_errors",
        }
        for index, group in enumerate(_fixture_groups())
    ]


def test_pilot_and_bound_sources_authenticate_before_measurement() -> None:
    """REQ-VERIFY-7565 / SCENARIO-VERIFY-7565-PRECONDITIONS."""

    checks, context = exp.authenticate_pilot(_pilot(), _bound_hashes())

    assert all(row["passed"] is True for row in checks)
    assert context["model_sha256"] == "sha256:model"
    assert context["role_counts"] == exp.ROLE_COUNTS
    assert context["source_label_access"] == []


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("native_tool_ready_score", 0, "pilot_native_tool_ready"),
        ("eval_capture_feasible_score", 0, "pilot_eval_capture_feasible"),
        ("verdict_class", "blocked", "pilot_verdict_class"),
        ("flagged_adversarial", True, "pilot_adversarial_state"),
    ],
)
def test_pilot_gate_drift_fails_named_check(field: str, value: object, failed_check: str) -> None:
    """REQ-VERIFY-7565 blocks changed pilot authority without fallback data."""

    pilot = _pilot()
    pilot[field] = value
    checks, _ = exp.authenticate_pilot(pilot, _bound_hashes())
    assert failed_check in {row["check"] for row in checks if not row["passed"]}


def test_evaluation_schedule_freezes_two_roles_and_six_cells() -> None:
    """REQ-VERIFY-7565 / SCENARIO-VERIFY-7565-CAPTURE."""

    schedule = _fixture_schedule()
    counts = {role: 1 for role in exp.ROLE_COUNTS}
    reduction = exp.reduce_evaluation_rows(_fixture_rows(schedule), schedule, counts)

    assert len(schedule) == 12
    assert reduction["passed"] is True
    assert reduction["complete_group_count"] == 2
    assert reduction["complete_forward_count"] == 12
    assert reduction["role_forward_counts"] == {"online": 6, "test": 6}
    assert reduction["generated_token_count"] == 0
    assert all(row["donor_role"] == row["role"] for row in schedule)


def test_schedule_rejects_role_shrinking_cross_role_donor_and_overlap() -> None:
    """REQ-VERIFY-7565 keeps both roles disjoint from fitting and each other."""

    counts = {role: 1 for role in exp.ROLE_COUNTS}
    with pytest.raises(exp.CaptureError, match="evaluation_role_counts"):
        exp.compile_evaluation_schedule(_fixture_groups()[:-1], expected_role_counts=counts)

    groups = _fixture_groups()
    groups[0]["donor_role"] = "online"
    with pytest.raises(exp.CaptureError, match="cross_role_donor"):
        exp.compile_evaluation_schedule(groups, expected_role_counts=counts)

    groups = _fixture_groups()
    groups[1]["component_hash"] = groups[0]["component_hash"]
    with pytest.raises(exp.CaptureError, match="component_overlap"):
        exp.compile_evaluation_schedule(groups, expected_role_counts=counts)


def test_prediction_and_evaluator_sidecars_are_separate(tmp_path: Path) -> None:
    """REQ-VERIFY-7565 / SCENARIO-VERIFY-7565-CUSTODY."""

    schedule = _fixture_schedule()
    rows = _fixture_rows(schedule)
    manifest = exp.write_role_sidecars(tmp_path, rows, _fixture_labels())
    predictions = capture._read_jsonl_receipt(manifest["predictions"], exp.REPO_ROOT)
    test_labels = capture._read_jsonl_receipt(manifest["test_labels"], exp.REPO_ROOT)
    online_labels = capture._read_jsonl_receipt(manifest["online_labels"], exp.REPO_ROOT)

    assert len(predictions) == 2
    assert all("label" not in row for row in predictions)
    assert all("display_logits" not in row for row in test_labels + online_labels)
    assert {row["role"] for row in test_labels} == {"test"}
    assert {row["role"] for row in online_labels} == {"online"}
    assert manifest["prediction_freeze_sha256"] == exp.canonical_hash(predictions)
    assert manifest["label_release_after_prediction_freeze"] is True


def test_online_baseline_uses_only_mapped_original_probabilities() -> None:
    """REQ-VERIFY-7565 freezes the online baseline without intervention labels."""

    schedule = _fixture_schedule()
    rows = _fixture_rows(schedule)
    online = [row for row in rows if row["role"] == "online"]
    original = [row for row in online if row["condition"] == "original"]
    for index, row in enumerate(original):
        row["display_logits"] = {" A": 0.0, " B": 0.0 if index == 0 else 2.0}

    baseline = exp.online_baseline(online)

    expected = sum(exp.protocol.semantic_unsupported_probability(row) for row in original) / 2
    assert baseline["method"] == "mean_mapped_original_source_probability"
    assert baseline["unsupported_label"] == 1
    assert baseline["intervention_absence_used_as_label"] is False
    assert baseline["original_forward_count"] == 2
    assert baseline["mean_probability"] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("role", "request_identity"),
        ("option_order", "request_identity"),
        ("donor_role", "cross_role_donor"),
        ("checkpoint_identity", "checkpoint_identity"),
    ],
)
def test_required_private_mutations_fail(
    tmp_path: Path, mutation: str, expected_failure: str
) -> None:
    """REQ-VERIFY-7565 / SCENARIO-VERIFY-7565-MUTATIONS."""

    controls = exp.run_qualification_controls(tmp_path)
    assert controls["passed"] is True
    assert controls["mutation_failures"][mutation]["passed"] is False
    assert expected_failure in controls["mutation_failures"][mutation]["failed_checks"]


def test_checkpoint_binds_complete_groups_and_model(tmp_path: Path) -> None:
    """REQ-VERIFY-7565 accepts only complete, identical resume state."""

    schedule = _fixture_schedule()
    rows = _fixture_rows(schedule[:6])
    path = tmp_path / "checkpoint.json"
    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")

    assert exp.load_checkpoint(path, schedule, "sha256:model-identity") == rows
    with pytest.raises(exp.CaptureError, match="checkpoint_partial_or_changed"):
        exp.write_checkpoint(path, schedule, rows[:5], "sha256:model-identity")

    document = json.loads(path.read_text(encoding="utf-8"))
    document["model_identity_sha256"] = "sha256:changed"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(exp.CaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")


def test_exposure_audit_preserves_prior_inspection() -> None:
    """REQ-VERIFY-7565 / SCENARIO-VERIFY-7565-EXPOSURE."""

    audit = exp.build_exposure_audit(
        {
            "candidate_official_training_groups": 2487,
            "exposed_candidate_groups": 2487,
            "fresh_eligible_groups": 0,
            "inventory_sha256": "sha256:inventory",
        },
        [{"event": "protocol_builder_evaluator_seal", "labels_consumed": 480}],
    )

    assert audit["prior_outcome_inspection"] is True
    assert audit["fresh_evidence_claim"] is False
    assert audit["confirmatory_claim_allowed"] is False
    assert audit["preserved_inventory_sha256"] == "sha256:inventory"


def test_blocked_artifact_has_zero_calls_and_exact_route() -> None:
    """REQ-VERIFY-7565 / SCENARIO-VERIFY-7565-BLOCKED."""

    failed = exp.gate(
        "pilot_missing",
        "external_precondition",
        "readable_nonempty_bytes",
        None,
        "==",
        False,
        "Missing pilot evidence cannot authorize collection.",
        upstream="exp7563-v661-native-pilot",
        field="path.bytes",
        path=exp.PILOT_PATH.as_posix(),
    )
    artifact = exp.build_blocked_artifact(
        failed_gate=failed,
        preconditions=[failed],
        source_hashes={},
        duration_s=2.0,
        phase_spans=[],
    )

    assert exp.validate_artifact(artifact, require_validation=False) == []
    assert artifact["honest_verdict"] == "complete_blocked_pilot_missing"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["planned_inference_substrate_class"] == "model_load_no_generation"
    assert artifact["sample_size_budget"]["unstarted"] == 1440
    assert artifact["test_capture_ready_score"] == 0
    assert artifact["online_capture_ready_score"] == 0
    assert artifact["capture_bundle_complete_score"] == 0
    first = artifact["gate_check_summary"]["first_failure"]
    assert first["upstream"] == "exp7563-v661-native-pilot"
    assert first["path"] == exp.PILOT_PATH.as_posix()


def test_complete_fixture_is_null_ready_and_independently_reducible(tmp_path: Path) -> None:
    """REQ-VERIFY-7565 / SCENARIO-VERIFY-7565-E2E."""

    artifact = exp.build_artifact_for_test(tmp_path)

    assert exp.validate_artifact(artifact, require_validation=False) == []
    replay = exp.independent_reduce(artifact, root=exp.REPO_ROOT)
    assert replay["passed"] is True
    assert artifact["test_capture_ready_score"] == 1
    assert artifact["online_capture_ready_score"] == 1
    assert artifact["capture_bundle_complete_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["predictive_benefit_measured"] is False
    assert artifact["source_label_access"] == []
    assert artifact["evaluator_label_access"]["after_prediction_freeze"] is True
    assert artifact["sample_size_budget"]["completed"] == 1440
    assert artifact["invocation_counts"]["generations"]["attempted"] == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "changed", "schema_mismatch"),
        ("experiment_id", "changed", "identity_mismatch"),
        ("honest_verdict", "not-terminal", "honest_verdict_not_terminal"),
        ("test_capture_ready_score", True, "test_capture_ready_score_not_bare_binary"),
        ("online_capture_ready_score", 2, "online_capture_ready_score_not_bare_binary"),
        ("capture_bundle_complete_score", None, "capture_bundle_complete_score_not_bare_binary"),
        ("MODEL_SPECS", [], "planned_model_specs_invalid"),
        ("source_label_access", ["test"], "capture_label_access_invalid"),
        ("predictive_benefit_measured", True, "predictive_benefit_claimed"),
        ("role_counts", {}, "role_counts_invalid"),
        ("field_principles", {}, "field_principles_missing"),
        ("current_invocation_events", None, "invocation_evidence_invalid"),
        ("sample_size_budget", {}, "sample_size_budget_invalid"),
        ("gate_check_summary", {}, "gate_check_summary_mismatch"),
        ("inference_substrate_class", "wrong", "inference_substrate_class_invalid"),
        ("capture_reduction", None, "capture_reduction_missing"),
        ("verdict_class", "positive-ish", "verdict_class_invalid"),
    ],
)
def test_terminal_validator_rejects_claim_and_schema_drift(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    """REQ-VERIFY-7565 keeps readiness separate from efficacy."""

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact[field] = value
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert error in exp.validate_artifact(artifact, require_validation=False)


def test_independent_reduction_detects_changed_prediction_bytes(tmp_path: Path) -> None:
    """REQ-VERIFY-7565 reloads raw custody without model calls."""

    artifact = exp.build_artifact_for_test(tmp_path)
    receipt = artifact["raw_manifest"]["role_sidecars"]["predictions"]
    raw_path = exp.REPO_ROOT / receipt["path"]
    raw_path.write_text("{}\n", encoding="utf-8")

    assert exp.independent_reduce(artifact, root=exp.REPO_ROOT) == {
        "passed": False,
        "error": "sidecar_receipt_invalid",
    }


def test_validator_rejects_ready_count_checksum_and_generation_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-7565 validates exact role budgets and zero generation."""

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["capture_reduction"]["role_forward_counts"]["test"] = 479
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert "ready_capture_counts_invalid" in exp.validate_artifact(
        artifact, require_validation=False
    )

    artifact = exp.build_artifact_for_test(tmp_path / "checksum")
    artifact["rows"][0]["disposition"] = "changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        artifact, require_validation=False
    )

    artifact = exp.build_artifact_for_test(tmp_path / "generation")
    artifact["current_invocation_events"].append(exp._event("generation", "attempted", "forbidden"))
    artifact["invocation_counts"] = exp.reduce_invocation_events(
        artifact["current_invocation_events"]
    )
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert "generation_call_detected" in exp.validate_artifact(artifact, require_validation=False)


def test_checkpoint_reader_rejects_each_changed_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7565 resume accepts only exact complete state."""

    schedule = _fixture_schedule()
    rows = _fixture_rows(schedule[:6])
    path = tmp_path / "checkpoint.json"
    assert exp.load_checkpoint(path, schedule, "sha256:model-identity") == []

    path.write_text("{", encoding="utf-8")
    with pytest.raises(exp.CaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(exp.CaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")

    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    first = exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    assert first["completed_group_hashes"]
    for field, value in (
        ("group_receipts", {}),
        ("group_receipts", [None]),
        ("extra", True),
    ):
        exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
        document = json.loads(path.read_text(encoding="utf-8"))
        document[field] = value
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(exp.CaptureError, match="checkpoint_identity"):
            exp.load_checkpoint(path, schedule, "sha256:model-identity")

    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["group_receipts"][0]["sha256"] = "sha256:changed"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(exp.CaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")

    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    monkeypatch.setattr(exp, "reduce_evaluation_rows", lambda *_a, **_k: {"passed": False})
    with pytest.raises(exp.CaptureError, match="checkpoint_partial_or_changed"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")


def test_reducer_and_feature_builders_reject_corrupt_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-7565 rejects missing bytes, identity, cells, and labels."""

    schedule = _fixture_schedule()
    rows = _fixture_rows(schedule)
    counts = {role: 1 for role in exp.ROLE_COUNTS}

    unknown = deepcopy(rows)
    unknown.append({"request_id": "unknown", "role": "test"})
    assert exp.reduce_evaluation_rows(unknown, schedule, counts)["passed"] is False
    changed = deepcopy(rows)
    changed[0]["reply_utf8_bytes"] = 1
    assert (
        "transport_bytes" in exp.reduce_evaluation_rows(changed, schedule, counts)["failed_checks"]
    )
    changed = deepcopy(rows)
    changed[0]["model_identity_sha256"] = None
    assert (
        "model_identity" in exp.reduce_evaluation_rows(changed, schedule, counts)["failed_checks"]
    )

    with pytest.raises(exp.CaptureError, match="readout_mapping_invalid"):
        exp._protocol_row({})
    with pytest.raises(exp.CaptureError, match="prediction_group_incomplete"):
        exp._prediction_rows(rows[:5])
    no_original = deepcopy(rows[:6])
    for row in no_original:
        row["condition"] = "absent"
    with pytest.raises(exp.CaptureError, match="prediction_original_cells_invalid"):
        exp._prediction_rows(no_original)
    with pytest.raises(exp.CaptureError, match="online_original_predictions_missing"):
        exp.online_baseline([row for row in rows if row["role"] == "test"])

    predictions, receipt = exp.freeze_prediction_sidecar(tmp_path, rows)
    bad_labels = _fixture_labels()
    bad_labels[0]["label"] = 3
    with pytest.raises(exp.CaptureError, match="evaluator_identity_or_label_invalid"):
        exp.write_evaluator_sidecars(tmp_path, predictions, receipt, bad_labels)
    with pytest.raises(exp.CaptureError, match="evaluator_identity_or_label_invalid"):
        exp.write_evaluator_sidecars(tmp_path, predictions, receipt, _fixture_labels()[:-1])


def test_raw_manifest_and_independent_reducer_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7565 retains bounded hash-bound raw custody."""

    schedule = _fixture_schedule()[:6]
    rows = _fixture_rows(schedule)
    empty = exp.write_raw_manifest(tmp_path / "empty", schedule, [])
    assert empty["native_rows"]["rows"] == 0
    small = exp.write_raw_manifest(tmp_path / "small", schedule, rows)
    assert small["forward_schedule"]["rows"] == 6

    original = exp.capture.write_jsonl_sidecar

    def oversized(*args: object, **kwargs: object) -> dict[str, object]:
        receipt = original(*args, **kwargs)
        receipt["bytes"] = 20 * 1024 * 1024
        return receipt

    monkeypatch.setattr(exp.capture, "write_jsonl_sidecar", oversized)
    with pytest.raises(exp.CaptureError, match="raw_sidecar_size_limit"):
        exp.write_raw_manifest(tmp_path / "large", schedule, rows)
    monkeypatch.setattr(exp.capture, "write_jsonl_sidecar", original)

    with pytest.raises(exp.CaptureError, match="sidecar_receipt_invalid"):
        exp._reload_shards(None, exp.REPO_ROOT)
    with pytest.raises(exp.CaptureError, match="sidecar_receipt_invalid"):
        exp._reload_shards([None], exp.REPO_ROOT)

    blocked = exp.build_blocked_artifact(
        failed_gate=exp.gate(
            "gpu",
            "external_precondition",
            True,
            False,
            "==",
            False,
            "principle",
            upstream="gpu",
            field="available",
        ),
        preconditions=[],
        source_hashes={},
        duration_s=2.0,
        phase_spans=[],
    )
    assert exp.independent_reduce(blocked)["mode"] == "blocked_no_measurement"

    artifact = exp.build_artifact_for_test(tmp_path / "fixture")
    artifact["role_counts"] = None
    assert exp.independent_reduce(artifact) == {
        "passed": False,
        "error": "role_counts_invalid",
    }
    artifact = exp.build_artifact_for_test(tmp_path / "missing-sidecars")
    artifact["raw_manifest"].pop("role_sidecars")
    assert exp.independent_reduce(artifact) == {
        "passed": False,
        "error": "sidecar_receipt_invalid",
    }


def test_complete_builder_classifies_partial_and_disqualified(tmp_path: Path) -> None:
    """REQ-VERIFY-7565 separates owned incompletion from invalid evidence."""

    baseline = exp.build_artifact_for_test(tmp_path)
    reduction = deepcopy(baseline["capture_reduction"])
    reduction.update(
        {
            "passed": False,
            "complete_group_count": 1,
            "complete_forward_count": 6,
            "role_group_counts": {"test": 1},
            "role_forward_counts": {"test": 6},
            "failed_checks": ["partial_group"],
        }
    )
    arguments = {
        "preconditions": [],
        "source_hashes": {},
        "model_specs": baseline["model_specs"],
        "events": baseline["current_invocation_events"][:14],
        "historical_invocation_counts": {},
        "raw_manifest": baseline["raw_manifest"],
        "reduction": reduction,
        "controls": baseline["qualification_controls"],
        "ownership": baseline["ownership_receipt"],
        "validation_receipts": [],
        "duration_s": 3.0,
        "phase_spans": [],
        "require_validation": False,
    }
    partial = exp.build_complete_artifact(**arguments, live_error="RuntimeError:stopped")
    assert partial["verdict_class"] == "partial"
    assert partial["live_error"] == "RuntimeError:stopped"
    disqualified = exp.build_complete_artifact(**arguments)
    assert disqualified["verdict_class"] == "disqualified"
    assert exp._validation_passed(None) is False


def test_validator_rejects_blocked_and_ready_custody_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-7565 validates blocked no-run and ready sidecar claims."""

    assert exp.validate_artifact(None, require_validation=False) == ["artifact_not_object"]
    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["readout_kind"] = "generation"
    artifact["invocation_counts"]["forward_calls_completed"] = 0
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    errors = exp.validate_artifact(artifact, require_validation=False)
    assert "readout_kind_invalid" in errors
    assert "invocation_counts_mismatch" in errors
    artifact = exp.build_artifact_for_test(tmp_path / "counter-shape")
    artifact["invocation_counts"] = None
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert "invocation_counts_mismatch" in exp.validate_artifact(artifact, require_validation=False)
    assert "required_validation_failed" in exp.validate_artifact(
        exp.build_artifact_for_test(tmp_path / "required")
    )

    failed = exp.gate(
        "gpu",
        "external_precondition",
        True,
        False,
        "==",
        False,
        "principle",
        upstream="gpu",
        field="available",
    )
    blocked = exp.build_blocked_artifact(
        failed_gate=failed,
        preconditions=[failed],
        source_hashes={},
        duration_s=2.0,
        phase_spans=[],
    )
    blocked.update(
        {
            "inference_substrate_class": "wrong",
            "planned_inference_substrate_class": "wrong",
            "rows": [{"fabricated": True}],
            "test_capture_ready_score": 1,
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    errors = set(exp.validate_artifact(blocked, require_validation=False))
    assert {
        "blocked_substrate_invalid",
        "blocked_planned_substrate_invalid",
        "blocked_artifact_fabricated_work",
        "blocked_artifact_ready",
    } <= errors

    for field, error in (
        ("raw_manifest", "role_sidecar_custody_invalid"),
        ("evaluator_label_access", "evaluator_release_order_invalid"),
        ("exposure_audit", "exposure_audit_invalid"),
        ("online_baseline", "online_baseline_invalid"),
    ):
        changed = exp.build_artifact_for_test(tmp_path / field)
        changed[field] = None
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert error in exp.validate_artifact(changed, require_validation=False)
