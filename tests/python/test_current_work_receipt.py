"""Tests for producer-side current-work receipts.

Spec refs: REQ-REPORT-7395, SCENARIO-REPORT-7395-CURRENT,
SCENARIO-REPORT-7395-MUTATIONS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import current_work_receipt as receipt


def _sidecars(tmp_path: Path) -> list[dict[str, str]]:
    scripted = tmp_path / "simulated_transport_events.json"
    historical = tmp_path / "historical_model_receipts.json"
    receipt.atomic_json(scripted, {"events": [{"generation_calls_attempted": 2}]})
    receipt.atomic_json(historical, {"invocation_counts": {"model_loads_attempted": 1}})
    return [
        receipt.sidecar_reference(scripted, root=tmp_path, scope="simulated_transport"),
        receipt.sidecar_reference(historical, root=tmp_path, scope="historical"),
    ]


def _valid_receipt(tmp_path: Path) -> dict:
    return receipt.build_current_work_receipt(
        inference_substrate="host CPU aggregation and exact receipt reduction",
        inference_substrate_details={"device": "cpu", "software": "python"},
        inference_substrate_class="aggregation",
        execution_venue="host",
        duration_s=2.0,
        phase_spans=[{"phase": "evaluate", "start_s": 0.1, "end_s": 1.9}],
        owned_run_events=[],
        sidecar_references=_sidecars(tmp_path),
        small_ebm_training={"performed": False},
    )


def test_scenario_report_7395_current_uses_only_owned_events(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-CURRENT excludes scripted and historical counts."""

    value = _valid_receipt(tmp_path)
    assert value["inference_substrate"] == "host CPU aggregation and exact receipt reduction"
    assert value["inference_substrate_details"]["device"] == "cpu"
    assert value["model_invoked"] is False
    assert value["invocation_counts"] == receipt.ZERO_INVOCATION_COUNTS
    assert value["current_invocation_events"] == []
    assert {row["scope"] for row in value["receipt_sidecars"]} == {
        "historical",
        "simulated_transport",
    }
    assert receipt.validate_current_work_receipt(value, root=tmp_path) == []


def test_scenario_report_7395_current_counts_failed_attempts() -> None:
    """REQ-REPORT-7395 counts a failed current load as an invocation attempt."""

    events = [
        {"event_id": "load-1", "operation": "model_load", "state": "attempted"},
        {"event_id": "load-1", "operation": "model_load", "state": "failed"},
    ]
    value = receipt.build_current_work_receipt(
        inference_substrate="owned native model load attempt",
        inference_substrate_details={"device": "gpu"},
        inference_substrate_class="model_load_no_generation",
        execution_venue="host",
        duration_s=61.0,
        phase_spans=[{"phase": "load", "start_s": 0.0, "end_s": 60.5}],
        owned_run_events=events,
        sidecar_references=[],
        small_ebm_training={"performed": False},
    )
    assert value["model_invoked"] is True
    assert value["invocation_counts"]["model_loads_attempted"] == 1
    assert value["invocation_counts"]["model_loads_failed"] == 1
    assert value["invocation_counts"]["model_loads_in_flight"] == 0
    assert receipt.validate_current_work_receipt(value) == []


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("unreported_load", "model_invoked_mismatch"),
        ("falsified_duration", "phase_span_exceeds_duration:evaluate"),
        ("dropped_failed_call", "invocation_counts_mismatch"),
        ("missing_completion", "unfinished_current_invocation:gen-1"),
        ("invalid_venue", "execution_venue_invalid"),
        ("changed_source_hash", "sidecar_hash_mismatch:simulated_transport_events.json"),
    ],
)
def test_scenario_report_7395_mutations_fail_closed(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    """SCENARIO-REPORT-7395-MUTATIONS rejects each named receipt defect."""

    value = _valid_receipt(tmp_path)
    if mutation == "unreported_load":
        value["current_invocation_events"] = [
            {"event_id": "load-1", "operation": "model_load", "state": "attempted"},
            {"event_id": "load-1", "operation": "model_load", "state": "completed"},
        ]
    elif mutation == "falsified_duration":
        value["duration_s"] = 0.5
    elif mutation == "dropped_failed_call":
        value["invocation_counts"]["generation_calls_failed"] = 1
    elif mutation == "missing_completion":
        value["current_invocation_events"] = [
            {"event_id": "gen-1", "operation": "generation", "state": "attempted"}
        ]
        value["model_invoked"] = True
        value["invocation_counts"]["generation_calls_attempted"] = 1
        value["invocation_counts"]["generation_calls_in_flight"] = 1
    elif mutation == "invalid_venue":
        value["execution_venue"] = "host_cpu"
    else:
        path = tmp_path / "simulated_transport_events.json"
        path.write_text(json.dumps({"events": []}), encoding="utf-8")
    assert expected_error in receipt.validate_current_work_receipt(value, root=tmp_path)


def test_req_report_7395_rejects_malformed_events_and_sidecars(tmp_path: Path) -> None:
    """REQ-REPORT-7395 rejects bad event vocabulary and unscoped sidecars."""

    value = _valid_receipt(tmp_path)
    changed = deepcopy(value)
    changed["inference_substrate"] = {"device": "cpu"}
    changed["receipt_sidecars"][0]["scope"] = "current"
    changed["current_invocation_events"] = [
        {"event_id": "x", "operation": "training", "state": "started"}
    ]
    errors = receipt.validate_current_work_receipt(changed, root=tmp_path)
    assert "inference_substrate_not_string" in errors
    assert "sidecar_scope_invalid:simulated_transport_events.json" in errors
    assert "event_operation_invalid:x" in errors
    assert "event_state_invalid:x" in errors


def test_req_report_7395_covers_ledger_and_reference_failure_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-7395 names contradictory transitions and malformed references."""

    outside = tmp_path.parent / "outside-receipt.json"
    receipt.atomic_json(outside, {"value": 1})
    reference = receipt.sidecar_reference(outside, root=tmp_path, scope="historical")
    assert reference["path"] == str(outside.resolve())

    value = _valid_receipt(tmp_path)
    value["duration_s"] = False
    value["phase_spans"] = ["not-a-row"]
    value["receipt_sidecars"].append("not-a-reference")
    value["current_invocation_events"] = [
        {"event_id": "changed", "operation": "model_load", "state": "attempted"},
        {"event_id": "changed", "operation": "generation", "state": "completed"},
        {"event_id": "terminal-only", "operation": "generation", "state": "failed"},
        {"event_id": "double-terminal", "operation": "generation", "state": "attempted"},
        {"event_id": "double-terminal", "operation": "generation", "state": "failed"},
        {"event_id": "double-terminal", "operation": "generation", "state": "cancelled"},
    ]
    errors = receipt.validate_current_work_receipt(value, root=tmp_path)
    assert "event_operation_changed:changed" in errors
    assert "event_attempt_count_invalid:terminal-only" in errors
    assert "event_terminal_count_invalid:double-terminal" in errors
    assert "duration_invalid" in errors
    assert "sidecar_reference_invalid" in errors

    value["duration_s"] = 2.0
    errors = receipt.validate_current_work_receipt(value, root=tmp_path)
    assert "phase_span_invalid" in errors
