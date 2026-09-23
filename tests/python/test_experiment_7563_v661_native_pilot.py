"""Tests for the V661 native option-forward pilot.

Spec refs: REQ-VERIFY-7563 and SCENARIO-VERIFY-7563-*.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from carnot import experiment_7563_v661_native_pilot as exp


def _protocol_artifact() -> dict[str, object]:
    groups = []
    donors = []
    for role, count in exp.ROLE_COUNTS.items():
        for index in range(count):
            component = exp.sha256_text(f"{role}-component-{index}")
            donor = exp.sha256_text(f"{role}-donor-{index}")
            groups.append(
                {
                    "role": role,
                    "component_hash": component,
                    "official_split": "test" if role == "test" else "train",
                    "tool_type": "grep",
                }
            )
            donors.append(
                {
                    "role": role,
                    "component_hash": component,
                    "donor_component_hash": donor,
                    "tool_type": "grep",
                }
            )
    return {
        "experiment_id": "exp7533-v659-tool-protocol",
        "tool_protocol_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
        "license_attribution": {
            "dataset": exp.DATASET_ID,
            "revision": exp.DATASET_REVISION,
        },
        "role_manifest": {"groups": groups},
        "donor_manifest": {
            "groups": donors,
            "altered_sources_have_labels": False,
        },
        "reader_access_contract": {
            "capture": "predictors_only_no_labels",
            "actual_capture_label_access": [],
        },
        "exposure_inventory": {
            "events": [
                {"event": "capture_reader", "labels_consumed": 0},
                {"event": "role_selection", "labels_consumed": 0},
            ]
        },
    }


def _runner_artifact() -> dict[str, object]:
    return {
        "experiment_id": "exp7548-capture-runner",
        "capture_runner_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_gpu_capacity_unavailable",
        "role_counts": {"fit": 160, "tune": 40, "policy": 40, "test": 80, "development": 12},
        "qualification_controls": {
            "passed": True,
            "resume_parity": True,
            "mutation_failures": {
                "donor_component_hash": True,
                "group_hash": True,
                "option_order": True,
                "prompt_token_count": True,
                "request_id": True,
            },
            "reduction": {"passed": True},
        },
    }


def _request(condition: str, order: tuple[str, str], prefix: str) -> dict[str, object]:
    prompt = f"{prefix} {condition} {'/'.join(order)}"
    return {
        "condition": condition,
        "option_order": list(order),
        "prompt": prompt,
        "prompt_sha256": exp.sha256_text(prompt),
        "prompt_token_count": len(prompt.split()) + 1,
        "readout_kind": "option_logits",
        "generated_token_budget": 0,
    }


def _group(role: str, index: int) -> dict[str, object]:
    requests = [
        _request(condition, order, f"{role}-{index}")
        for condition in exp.CONDITIONS
        for order in exp.OPTION_ORDERS
    ]
    return {
        "component_hash": exp.sha256_text(f"component-{role}-{index}"),
        "donor_component_hash": exp.sha256_text(f"donor-{role}-{index}"),
        "role": role,
        "tool_type": "grep",
        "label_scope": "original_source_only",
        "requests": requests,
    }


def test_upstreams_authenticate_separately_and_keep_historic_block() -> None:
    """REQ-VERIFY-7563 / SCENARIO-VERIFY-7563-PRECONDITIONS."""

    checks, context = exp.authenticate_upstreams(_protocol_artifact(), _runner_artifact())

    assert all(row["passed"] is True for row in checks)
    assert context["role_counts"] == exp.ROLE_COUNTS
    assert context["development_excluded_count"] == 12
    assert context["historical_runner_verdict"] == "complete_blocked_gpu_capacity_unavailable"
    assert context["runner_transport_ready"] is True
    assert context["source_label_access"] == []


@pytest.mark.parametrize(
    ("target", "field", "value", "check"),
    [
        ("protocol", "revision", "wrong", "dataset_revision"),
        ("protocol", "actual_capture_label_access", ["test"], "capture_label_access"),
        ("runner", "capture_runner_ready_score", 0, "capture_runner_ready"),
        ("runner", "flagged_adversarial", True, "capture_runner_adversarial"),
        ("runner", "controls", False, "transport_controls"),
    ],
)
def test_upstream_authentication_fails_named_operands(
    target: str, field: str, value: object, check: str
) -> None:
    """REQ-VERIFY-7563 blocks changed sealed evidence without fallback data."""

    protocol = _protocol_artifact()
    runner = _runner_artifact()
    if target == "protocol" and field == "revision":
        protocol["license_attribution"]["revision"] = value
    elif target == "protocol":
        protocol["reader_access_contract"][field] = value
    elif field == "controls":
        runner["qualification_controls"]["passed"] = value
    else:
        runner[field] = value
    checks, _ = exp.authenticate_upstreams(protocol, runner)
    failed = {row["check"] for row in checks if row["passed"] is not True}
    assert check in failed


def test_online_wrapper_builds_two_independent_1440_forward_captures() -> None:
    """REQ-VERIFY-7563 / SCENARIO-VERIFY-7563-FORECAST."""

    counts = {role: 1 for role in exp.ROLE_COUNTS}
    groups = [_group(role, 0) for role in counts]
    captures = exp.compile_capture_roles(groups, expected_role_counts=counts)

    assert captures["fit"]["roles"] == ["fit", "tune", "policy"]
    assert captures["evaluation"]["roles"] == ["test", "online"]
    assert captures["fit"]["group_count"] == 3
    assert captures["evaluation"]["group_count"] == 2
    assert captures["fit"]["forward_count"] == 18
    assert captures["evaluation"]["forward_count"] == 12
    assert captures["source_label_access"] == []
    assert all(row["generated_token_budget"] == 0 for row in captures["fit"]["schedule"])


def test_capture_wrapper_rejects_missing_online_role() -> None:
    """REQ-VERIFY-7563 does not shrink roles to open feasibility."""

    counts = {role: 1 for role in exp.ROLE_COUNTS}
    groups = [_group(role, 0) for role in counts if role != "online"]
    with pytest.raises(exp.NativePilotError, match="capture_role_counts"):
        exp.compile_capture_roles(groups, expected_role_counts=counts)


def test_forecasts_use_lengths_and_independent_checkpoint_costs() -> None:
    """REQ-VERIFY-7563 / SCENARIO-VERIFY-7563-FORECAST."""

    pilot_rows = [
        {"forward_seconds": 1.0, "prompt_token_count": 100, "disposition": "complete"},
        {"forward_seconds": 2.0, "prompt_token_count": 200, "disposition": "complete"},
    ]
    fit_lengths = [100] * exp.CAPTURE_FORWARDS
    evaluation_lengths = [201] * exp.CAPTURE_FORWARDS
    forecast = exp.forecast_captures(
        load_seconds=100.0,
        pilot_rows=pilot_rows,
        fit_token_lengths=fit_lengths,
        evaluation_token_lengths=evaluation_lengths,
        fit_checkpoint_seconds=10.0,
        evaluation_checkpoint_seconds=20.0,
    )

    assert forecast["formula"] == "load + sum(p95_forward * tokens / p95_pilot_tokens) + checkpoint"
    assert forecast["fit"]["planned_groups"] == 240
    assert forecast["fit"]["planned_forwards"] == 1440
    assert forecast["evaluation"]["planned_groups"] == 240
    assert forecast["evaluation"]["acquisition_seconds"] > forecast["fit"]["acquisition_seconds"]
    assert forecast["fit_capture_feasible_score"] == 1
    assert forecast["eval_capture_feasible_score"] == 0
    assert forecast["validation_reserve_seconds"] == 900.0


@pytest.mark.parametrize(
    ("pilot_rows", "fit_lengths", "evaluation_lengths", "message"),
    [
        ([], [1] * 1440, [1] * 1440, "pilot_durations_missing"),
        (
            [{"forward_seconds": 1.0, "prompt_token_count": 1}],
            [1],
            [1] * 1440,
            "capture_length_count",
        ),
        (
            [{"forward_seconds": 1.0, "prompt_token_count": 1}],
            [4097] * 1440,
            [1] * 1440,
            "capture_prompt_overlength",
        ),
    ],
)
def test_forecasts_fail_closed_on_missing_or_changed_samples(
    pilot_rows: list[dict[str, object]],
    fit_lengths: list[int],
    evaluation_lengths: list[int],
    message: str,
) -> None:
    """REQ-VERIFY-7563 preserves the full fixed capture budget."""

    with pytest.raises(exp.NativePilotError, match=message):
        exp.forecast_captures(
            load_seconds=1.0,
            pilot_rows=pilot_rows,
            fit_token_lengths=fit_lengths,
            evaluation_token_lengths=evaluation_lengths,
            fit_checkpoint_seconds=1.0,
            evaluation_checkpoint_seconds=1.0,
        )


def test_blocked_artifact_records_planned_class_and_exact_gate() -> None:
    """REQ-VERIFY-7563 / SCENARIO-VERIFY-7563-BLOCKED."""

    failed = exp.gate(
        "owned_gpu_available",
        "external_precondition",
        True,
        False,
        "==",
        False,
        "Fresh ownership prevents use of stale capacity evidence.",
        upstream="nvidia-smi",
        field="admissible_gpu",
        path="nvidia-smi_compute_process_inventory",
    )
    artifact = exp.build_blocked_artifact(
        failed_gate=failed,
        preconditions=[failed],
        source_hashes={},
        duration_s=2.1,
        phase_spans=[],
    )

    assert exp.validate_artifact(artifact, require_validation=False) == []
    assert artifact["honest_verdict"] == "complete_blocked_owned_gpu_available"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["planned_inference_substrate_class"] == "model_load_no_generation"
    assert artifact["model_specs"] == []
    assert artifact["sample_size_budget"]["unstarted"] == 72
    assert (
        artifact["gate_check_summary"]["first_failure"]["path"]
        == "nvidia-smi_compute_process_inventory"
    )


def test_fixture_artifact_is_cold_valid_and_preserves_roles() -> None:
    """REQ-VERIFY-7563 / SCENARIO-VERIFY-7563-E2E."""

    artifact = exp.build_artifact_for_test()

    assert exp.validate_artifact(artifact, require_validation=False) == []
    assert artifact["experiment_id"] == "exp7563-v661-native-pilot"
    assert artifact["run_date"] == "20260923"
    assert artifact["MODEL_SPECS"] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert artifact["native_tool_ready_score"] == 1
    assert artifact["readout_kind"] == "option_logits"
    assert artifact["inference_substrate"] == "live_llm_embedding_extraction"
    assert artifact["role_manifest"]["role_counts"] == exp.ROLE_COUNTS
    assert artifact["role_manifest"]["development_excluded_count"] == 12
    assert artifact["source_label_access"] == []
    assert artifact["sample_size_budget"] == {
        "planned": 72,
        "attempted": 72,
        "completed": 72,
        "excluded": 0,
        "failed": 0,
        "censored": 0,
        "unstarted": 0,
    }
    counts = artifact["invocation_counts"]
    assert counts["model_loads_completed"] == 1
    assert counts["forward_calls_completed"] == 72
    assert counts["generations"] == {
        "attempted": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
    }


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("schema", "changed", "schema_mismatch"),
        ("verdict_class", "unknown", "verdict_class_invalid"),
        ("native_tool_ready_score", True, "native_tool_ready_score_not_bare_binary"),
        ("source_label_access", ["test"], "capture_label_access_invalid"),
        ("readout_kind", "text_generation", "readout_kind_invalid"),
        ("field_principles", {}, "field_principles_missing"),
    ],
)
def test_cold_validator_rejects_terminal_drift(field: str, value: object, expected: str) -> None:
    """REQ-VERIFY-7563 rejects changed terminal declarations."""

    artifact = exp.build_artifact_for_test()
    artifact[field] = value
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert expected in exp.validate_artifact(artifact, require_validation=False)


def test_cold_validator_rejects_checksum_and_invocation_drift() -> None:
    """REQ-VERIFY-7563 binds raw custody and typed call counts."""

    artifact = exp.build_artifact_for_test()
    artifact["rows"][0]["disposition"] = "changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        artifact, require_validation=False
    )

    artifact = exp.build_artifact_for_test()
    artifact["invocation_counts"]["forward_calls_completed"] = 71
    artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
    assert "invocation_counts_mismatch" in exp.validate_artifact(artifact, require_validation=False)


def test_gate_summary_keeps_all_failed_checks_and_first_route() -> None:
    """REQ-VERIFY-7563 keeps the exact blocked routing operands."""

    gates = [
        exp.gate(
            "first",
            "validity",
            1,
            0,
            "==",
            False,
            "First failure principle.",
            upstream="upstream-a",
            field="field-a",
            path="path-a",
        ),
        exp.gate(
            "second",
            "readiness",
            True,
            False,
            "is",
            False,
            "Second failure principle.",
            upstream="upstream-b",
            field="field-b",
        ),
    ]
    summary = exp.gate_check_summary(gates)

    assert summary["failed_checks"] == ["first", "second"]
    assert summary["first_failure"]["upstream"] == "upstream-a"
    assert summary["first_failure"]["path"] == "path-a"


def test_validation_requires_current_receipts_only_when_requested() -> None:
    """REQ-VERIFY-7563 separates cold shape checks from publication checks."""

    artifact = exp.build_artifact_for_test()
    assert "required_validation_failed" in exp.validate_artifact(artifact)
    assert exp.validate_artifact(None, require_validation=False) == ["artifact_not_object"]

    changed = deepcopy(artifact)
    changed["honest_verdict"] = "not_terminal"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "honest_verdict_not_terminal" in exp.validate_artifact(changed, require_validation=False)


def test_capture_wrapper_rejects_failed_schedule_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-7563 rejects a six-cell schedule that fails independent checks."""

    counts = {role: 1 for role in exp.ROLE_COUNTS}
    groups = [_group(role, 0) for role in counts]
    calls = 0

    def reduce_schedule(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"passed": calls == 1}

    monkeypatch.setattr(exp.capture, "validate_capture_schedule", reduce_schedule)
    with pytest.raises(exp.NativePilotError, match="capture_schedule_invalid"):
        exp.compile_capture_roles(groups, expected_role_counts=counts)


def test_validator_exercises_missing_and_blocked_invariants() -> None:
    """REQ-VERIFY-7563 checks each required terminal field independently."""

    baseline = exp.build_artifact_for_test()
    mutations = {
        "identity_mismatch": {"experiment_id": "changed"},
        "planned_model_specs_invalid": {"MODEL_SPECS": []},
        "invocation_evidence_invalid": {"current_invocation_events": None},
        "sample_size_budget_invalid": {"sample_size_budget": {}},
        "role_manifest_invalid": {"role_manifest": {}},
        "gate_check_summary_mismatch": {"gate_check_summary": {}},
        "inference_substrate_class_invalid": {"inference_substrate_class": "wrong"},
    }
    for expected, updates in mutations.items():
        artifact = deepcopy(baseline)
        artifact.update(updates)
        artifact["reproducibility_checksum"] = exp.artifact_checksum(artifact)
        assert expected in exp.validate_artifact(artifact, require_validation=False)

    generated = deepcopy(baseline)
    generated["current_invocation_events"].append(
        {"operation": "generation", "state": "attempted", "call_id": "forbidden"}
    )
    generated["invocation_counts"] = exp.reduce_invocation_events(
        generated["current_invocation_events"]
    )
    generated["reproducibility_checksum"] = exp.artifact_checksum(generated)
    assert "generation_call_detected" in exp.validate_artifact(generated, require_validation=False)

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
            "native_tool_ready_score": 1,
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    errors = exp.validate_artifact(blocked, require_validation=False)
    assert {
        "blocked_substrate_invalid",
        "blocked_planned_substrate_invalid",
        "blocked_artifact_fabricated_work",
        "blocked_artifact_ready",
    } <= set(errors)


def test_independent_reduction_reloads_exact_sidecars(tmp_path: Path) -> None:
    """REQ-VERIFY-7563 / SCENARIO-VERIFY-7563-CUSTODY."""

    panel = exp.native.select_development_panel(
        exp.native._fixture_candidates(),
        excluded_component_hashes=set(),
        token_count=exp.native._fixture_token_count,
    )
    schedule = exp.native.build_forward_schedule(
        panel["groups"], token_count=exp.native._fixture_token_count
    )
    rows = [exp.native._fixture_native_row(row, index) for index, row in enumerate(schedule)]
    schedule_receipt = exp.capture.write_jsonl_sidecar(
        tmp_path / "schedule.jsonl", schedule, root=exp.REPO_ROOT
    )
    rows_receipt = exp.capture.write_jsonl_sidecar(
        tmp_path / "rows.jsonl", rows, root=exp.REPO_ROOT
    )
    value = {
        "raw_native_rows": {
            "forward_schedule": schedule_receipt,
            "native_rows": rows_receipt,
        },
        "transport_reduction": exp.native.reduce_native_rows(rows, schedule),
    }

    assert exp.independent_reduce(value)["passed"] is True
    assert exp.independent_reduce({}) == {"passed": False, "error": "sidecar_receipt_missing"}
    (tmp_path / "rows.jsonl").write_text("{}\n", encoding="utf-8")
    assert exp.independent_reduce(value) == {"passed": False, "error": "sidecar_receipt_invalid"}

    blocked = exp.build_blocked_artifact(
        failed_gate=exp.gate(
            "gpu",
            "external",
            True,
            False,
            "==",
            False,
            "principle",
            upstream="gpu",
            field="gpu",
        ),
        preconditions=[],
        source_hashes={},
        duration_s=2.0,
        phase_spans=[],
    )
    assert exp.independent_reduce(blocked)["mode"] == "blocked_no_measurement"
    assert exp._validation_passed(None) is False
