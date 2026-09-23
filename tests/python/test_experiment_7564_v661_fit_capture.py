"""Tests for the V661 fit capture.

Spec refs: REQ-VERIFY-7564 and SCENARIO-VERIFY-7564-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7548_v660_capture_runner as capture
from carnot import experiment_7564_v661_fit_capture as exp


def _pilot() -> dict[str, object]:
    return {
        "experiment_id": "exp7563-v661-native-pilot",
        "native_tool_ready_score": 1,
        "fit_capture_feasible_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "MODEL_SPECS": [exp.MODEL_ID],
        "source_artifact_hashes": {
            exp.PROTOCOL_PATH.as_posix(): "sha256:protocol",
            exp.RUNNER_PATH.as_posix(): "sha256:runner",
            exp.PILOT_MODULE_PATH.as_posix(): "sha256:pilot-module",
            exp.PILOT_WRAPPER_PATH.as_posix(): "sha256:pilot-wrapper",
            "/cache/Qwen3.8-27B-Q4_K_M.gguf": "sha256:model",
        },
        "model_specs": [{"model_sha256": "sha256:model"}],
        "role_manifest": {
            "sealed_capture": {
                "fit": {
                    "roles": ["fit", "tune", "policy"],
                    "group_count": 240,
                    "forward_count": 1440,
                    "role_schedule_hashes": {
                        "fit": "sha256:fit",
                        "tune": "sha256:tune",
                        "policy": "sha256:policy",
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
    groups = [capture._fixture_group(role, 0) for role in exp.FIT_ROLE_COUNTS]
    for group in groups:
        group["donor_role"] = group["role"]
    return groups


def _fixture_schedule() -> list[dict[str, object]]:
    counts = {role: 1 for role in exp.FIT_ROLE_COUNTS}
    return exp.compile_fit_schedule(_fixture_groups(), expected_role_counts=counts)


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


def test_pilot_and_bound_sources_authenticate_before_measurement() -> None:
    """REQ-VERIFY-7564 / SCENARIO-VERIFY-7564-PRECONDITIONS."""

    checks, context = exp.authenticate_pilot(_pilot(), _bound_hashes())

    assert all(row["passed"] is True for row in checks)
    assert context["model_sha256"] == "sha256:model"
    assert context["role_counts"] == exp.FIT_ROLE_COUNTS
    assert context["source_label_access"] == []


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("native_tool_ready_score", 0, "pilot_native_tool_ready"),
        ("fit_capture_feasible_score", 0, "pilot_fit_capture_feasible"),
        ("verdict_class", "blocked", "pilot_verdict_class"),
        ("flagged_adversarial", True, "pilot_adversarial_state"),
    ],
)
def test_pilot_gate_drift_fails_named_check(field: str, value: object, failed_check: str) -> None:
    """REQ-VERIFY-7564 blocks changed pilot authority without fallback data."""

    pilot = _pilot()
    pilot[field] = value
    checks, _ = exp.authenticate_pilot(pilot, _bound_hashes())
    assert failed_check in {row["check"] for row in checks if not row["passed"]}


def test_pilot_rejects_changed_hash_bound_wrapper() -> None:
    """REQ-VERIFY-7564 keeps the pilot's exact wrapper and protocol bytes."""

    hashes = _bound_hashes()
    hashes[exp.PILOT_WRAPPER_PATH.as_posix()] = "sha256:changed"
    checks, _ = exp.authenticate_pilot(_pilot(), hashes)
    failed = next(row for row in checks if row["check"] == "pilot_bound_sources")
    assert failed["passed"] is False
    assert failed["path"] == exp.PILOT_PATH.as_posix()


def test_fit_schedule_freezes_three_roles_and_six_cells() -> None:
    """REQ-VERIFY-7564 / SCENARIO-VERIFY-7564-CAPTURE."""

    schedule = _fixture_schedule()
    reduction = exp.reduce_fit_rows(
        _fixture_rows(schedule), schedule, {role: 1 for role in exp.FIT_ROLE_COUNTS}
    )

    assert len(schedule) == 18
    assert reduction["passed"] is True
    assert reduction["complete_group_count"] == 3
    assert reduction["complete_forward_count"] == 18
    assert reduction["generated_token_count"] == 0
    assert {row["condition"] for row in schedule} == set(exp.CONDITIONS)
    assert all(row["donor_role"] == row["role"] for row in schedule)


def test_fit_schedule_rejects_role_shrinking_and_cross_role_donor() -> None:
    """REQ-VERIFY-7564 does not shrink roles or accept cross-role donors."""

    counts = {role: 1 for role in exp.FIT_ROLE_COUNTS}
    with pytest.raises(exp.FitCaptureError, match="fit_role_counts"):
        exp.compile_fit_schedule(_fixture_groups()[:-1], expected_role_counts=counts)

    groups = _fixture_groups()
    groups[0]["donor_role"] = "policy"
    with pytest.raises(exp.FitCaptureError, match="cross_role_donor"):
        exp.compile_fit_schedule(groups, expected_role_counts=counts)


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("option_order", "request_identity"),
        ("donor_role", "cross_role_donor"),
        ("checkpoint_identity", "checkpoint_identity"),
    ],
)
def test_required_private_mutations_fail(
    tmp_path: Path, mutation: str, expected_failure: str
) -> None:
    """REQ-VERIFY-7564 / SCENARIO-VERIFY-7564-MUTATIONS."""

    controls = exp.run_qualification_controls(tmp_path)
    assert controls["passed"] is True
    assert controls["mutation_failures"][mutation]["passed"] is False
    assert expected_failure in controls["mutation_failures"][mutation]["failed_checks"]


def test_checkpoint_binds_schedule_rows_resets_and_model(tmp_path: Path) -> None:
    """REQ-VERIFY-7564 / SCENARIO-VERIFY-7564-RESUME."""

    schedule = _fixture_schedule()
    rows = _fixture_rows(schedule[:6])
    path = tmp_path / "checkpoint.json"
    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")

    loaded = exp.load_checkpoint(path, schedule, "sha256:model-identity")
    assert loaded == rows

    document = json.loads(path.read_text(encoding="utf-8"))
    document["model_identity_sha256"] = "sha256:changed"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(exp.FitCaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")


def test_checkpoint_rejects_partial_rows_and_reuses_identical_group_bytes(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7564 checkpoints only complete groups."""

    schedule = _fixture_schedule()
    rows = _fixture_rows(schedule[:6])
    path = tmp_path / "checkpoint.json"
    with pytest.raises(exp.FitCaptureError, match="checkpoint_partial_or_changed"):
        exp.write_checkpoint(path, schedule, rows[:5], "sha256:model-identity")

    first = exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    second = exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    assert second == first


def test_checkpoint_reader_fails_closed_for_each_malformed_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7564 resume accepts only identical content-hashed state."""

    schedule = _fixture_schedule()
    rows = _fixture_rows(schedule[:6])
    path = tmp_path / "checkpoint.json"
    assert exp.load_checkpoint(path, schedule, "sha256:model-identity") == []

    path.write_text("{", encoding="utf-8")
    with pytest.raises(exp.FitCaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(exp.FitCaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")

    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["group_receipts"] = {}
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(exp.FitCaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")

    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["group_receipts"] = [None]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(exp.FitCaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")

    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["group_receipts"][0]["sha256"] = "sha256:changed"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(exp.FitCaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")

    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["extra"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(exp.FitCaptureError, match="checkpoint_identity"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")

    exp.write_checkpoint(path, schedule, rows, "sha256:model-identity")
    monkeypatch.setattr(exp, "reduce_fit_rows", lambda *_args, **_kwargs: {"passed": False})
    with pytest.raises(exp.FitCaptureError, match="checkpoint_partial_or_changed"):
        exp.load_checkpoint(path, schedule, "sha256:model-identity")


def test_reducer_rejects_unknown_rows_bytes_and_model_identity() -> None:
    """REQ-VERIFY-7564 retains exact transport bytes and model identity."""

    schedule = _fixture_schedule()
    role_counts = {role: 1 for role in exp.FIT_ROLE_COUNTS}
    baseline = _fixture_rows(schedule)

    unknown = deepcopy(baseline)
    unknown.append({"request_id": "unknown", "role": "fit"})
    assert exp.reduce_fit_rows(unknown, schedule, role_counts)["passed"] is False

    changed_bytes = deepcopy(baseline)
    changed_bytes[0]["reply_utf8_bytes"] = 1
    assert (
        "transport_bytes"
        in exp.reduce_fit_rows(changed_bytes, schedule, role_counts)["failed_checks"]
    )

    changed_model = deepcopy(baseline)
    changed_model[0]["model_identity_sha256"] = None
    assert (
        "model_identity"
        in exp.reduce_fit_rows(changed_model, schedule, role_counts)["failed_checks"]
    )


def test_blocked_artifact_has_zero_calls_and_exact_route() -> None:
    """REQ-VERIFY-7564 / SCENARIO-VERIFY-7564-BLOCKED."""

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
    assert artifact["invocation_counts"] == exp.reduce_invocation_events([])
    assert artifact["gate_check_summary"]["first_failure"]["path"] == exp.PILOT_PATH.as_posix()


def test_complete_fixture_is_null_ready_and_label_blind(tmp_path: Path) -> None:
    """REQ-VERIFY-7564 / SCENARIO-VERIFY-7564-E2E."""

    artifact = exp.build_artifact_for_test(tmp_path)

    assert exp.validate_artifact(artifact, require_validation=False) == []
    assert exp.independent_reduce(artifact, root=exp.REPO_ROOT)["passed"] is True
    assert artifact["experiment_id"] == "exp7564-v661-fit-capture"
    assert artifact["fit_capture_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_fit_capture_ready_benefit_unmeasured"
    assert artifact["role_counts"] == exp.FIT_ROLE_COUNTS
    assert artifact["source_label_access"] == []
    assert artifact["predictive_benefit_measured"] is False
    assert artifact["readout_kind"] == "option_logits"
    assert artifact["sample_size_budget"]["completed"] == 1440
    assert artifact["invocation_counts"]["generations"]["attempted"] == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("schema", "changed", "schema_mismatch"),
        ("experiment_id", "changed", "identity_mismatch"),
        ("honest_verdict", "not-terminal", "honest_verdict_not_terminal"),
        ("fit_capture_ready_score", True, "fit_capture_ready_score_not_bare_binary"),
        ("MODEL_SPECS", [], "planned_model_specs_invalid"),
        ("source_label_access", ["fit"], "capture_label_access_invalid"),
        ("predictive_benefit_measured", True, "predictive_benefit_claimed"),
        ("readout_kind", "generation", "readout_kind_invalid"),
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
    """REQ-VERIFY-7564 keeps readiness separate from efficacy."""

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact[field] = value
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert error in exp.validate_artifact(artifact, require_validation=False)


def test_independent_reduction_detects_changed_raw_bytes(tmp_path: Path) -> None:
    """REQ-VERIFY-7564 reloads raw custody without new model calls."""

    artifact = exp.build_artifact_for_test(tmp_path)
    receipt = artifact["raw_manifest"]["native_row_shards"][0]
    raw_path = exp.REPO_ROOT / receipt["path"]
    raw_path.write_text("{}\n", encoding="utf-8")

    assert exp.independent_reduce(artifact, root=exp.REPO_ROOT) == {
        "passed": False,
        "error": "sidecar_receipt_invalid",
    }


def test_raw_manifest_handles_empty_small_and_oversized_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-7564 keeps each raw sidecar bounded and hash-bound."""

    schedule = _fixture_schedule()[:6]
    rows = _fixture_rows(schedule)
    empty = exp.write_raw_manifest(tmp_path / "empty", schedule, [])
    assert empty["native_rows"]["rows"] == 0

    small = exp.write_raw_manifest(tmp_path / "small", schedule, rows)
    assert small["forward_schedule"]["rows"] == 6
    assert small["native_rows"]["rows"] == 6

    original = exp.capture.write_jsonl_sidecar

    def oversized(*args: object, **kwargs: object) -> dict[str, object]:
        receipt = original(*args, **kwargs)
        receipt["bytes"] = 20 * 1024 * 1024
        return receipt

    monkeypatch.setattr(exp.capture, "write_jsonl_sidecar", oversized)
    with pytest.raises(exp.FitCaptureError, match="raw_sidecar_size_limit"):
        exp.write_raw_manifest(tmp_path / "large", schedule, rows)


def test_raw_reload_and_independent_reduction_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-7564 cold reduction rejects malformed receipts and role counts."""

    with pytest.raises(exp.FitCaptureError, match="sidecar_receipt_invalid"):
        exp._reload_shards(None, exp.REPO_ROOT)
    with pytest.raises(exp.FitCaptureError, match="sidecar_receipt_invalid"):
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


def test_complete_builder_classifies_partial_and_disqualified(tmp_path: Path) -> None:
    """REQ-VERIFY-7564 keeps owned incompletion distinct from invalid evidence."""

    baseline = exp.build_artifact_for_test(tmp_path)
    reduction = deepcopy(baseline["capture_reduction"])
    reduction.update(
        {
            "passed": False,
            "complete_group_count": 1,
            "complete_forward_count": 6,
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


def test_validation_helper_rejects_non_list() -> None:
    """REQ-VERIFY-7564 requires explicit command receipts."""

    assert exp._validation_passed(None) is False


def test_validator_requires_publication_receipts_only_when_requested(tmp_path: Path) -> None:
    """REQ-VERIFY-7564 separates cold shape checks from publication checks."""

    artifact = exp.build_artifact_for_test(tmp_path)
    assert "required_validation_failed" in exp.validate_artifact(artifact)
    assert exp.validate_artifact(None, require_validation=False) == ["artifact_not_object"]

    changed = deepcopy(artifact)
    changed["current_invocation_events"].append(
        {"operation": "generation", "state": "attempted", "call_id": "forbidden"}
    )
    changed["invocation_counts"] = exp.reduce_invocation_events(
        changed["current_invocation_events"]
    )
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "generation_call_detected" in exp.validate_artifact(changed, require_validation=False)

    mismatched = exp.build_artifact_for_test(tmp_path / "mismatched")
    mismatched["invocation_counts"]["forward_calls_completed"] = 0
    mismatched["reproducibility_checksum"] = exp.artifact_checksum(mismatched)
    assert "invocation_counts_mismatch" in exp.validate_artifact(
        mismatched, require_validation=False
    )


def test_validator_rejects_count_checksum_and_blocked_drift(tmp_path: Path) -> None:
    """REQ-VERIFY-7564 validates ready counts and blocked no-run claims."""

    artifact = exp.build_artifact_for_test(tmp_path)
    artifact["capture_reduction"]["complete_forward_count"] = 1439
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert "ready_capture_counts_invalid" in exp.validate_artifact(
        artifact, require_validation=False
    )

    artifact = exp.build_artifact_for_test(tmp_path / "checksum")
    artifact["rows"][0]["disposition"] = "changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        artifact, require_validation=False
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
            "fit_capture_ready_score": 1,
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
