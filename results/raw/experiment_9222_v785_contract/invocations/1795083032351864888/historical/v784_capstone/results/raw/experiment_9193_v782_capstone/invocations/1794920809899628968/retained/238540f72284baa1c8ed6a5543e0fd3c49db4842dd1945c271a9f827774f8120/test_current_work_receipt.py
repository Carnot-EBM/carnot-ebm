"""Focused tests for producer-side current-work receipts.

Spec refs: REQ-REPORT-7395, SCENARIO-REPORT-7395-CURRENT,
SCENARIO-REPORT-7395-MUTATIONS.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from carnot.reporting import current_work_receipt as receipt


def _event(state: str, timestamp: int) -> dict[str, object]:
    """Return one event owned by the fixed test run."""

    return {
        "scope": "current",
        "transport": "owned_runtime",
        "run_id": "test-run",
        "owner_pid": 42,
        "call_id": "load-1",
        "operation": "model_load",
        "state": state,
        "monotonic_ns": timestamp,
    }


def _valid(tmp_path: Path) -> dict[str, object]:
    """Build one complete no-model receipt with both sidecar scopes."""

    historical = receipt.write_immutable_sidecar(
        tmp_path / "historical.json",
        scope="historical_model_receipts",
        payload={"model_invoked": True},
        root=tmp_path,
    )
    scripted = receipt.write_immutable_sidecar(
        tmp_path / "scripted.json",
        scope="simulated_transport_events",
        payload={"simulated_transport_events": []},
        root=tmp_path,
    )
    return receipt.build_current_work_receipt(
        run_id="test-run",
        owner_pid=42,
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"device": "cpu"},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=1_000_000_000,
        ended_monotonic_ns=3_000_000_000,
        phase_spans=[{"phase": "evaluate", "start_s": 0.0, "end_s": 1.0}],
        sidecar_references=[historical, scripted],
    )


def test_scenario_report_7395_current_keeps_sidecar_counts_external(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-CURRENT uses only the empty owned ledger."""

    value = _valid(tmp_path)
    assert value["model_invoked"] is False
    assert value["invocation_counts"] == receipt.ZERO_INVOCATION_COUNTS
    assert value["event_sha256"] == receipt.canonical_hash([])
    assert receipt.validate_current_work_receipt(value, root=tmp_path) == []


def test_scenario_report_7395_current_counts_a_failed_owned_load() -> None:
    """SCENARIO-REPORT-7395-CURRENT counts failed attempts as invocation."""

    events = [_event("attempted", 1), _event("failed", 2)]
    value = receipt.build_current_work_receipt(
        run_id="test-run",
        owner_pid=42,
        events=events,
        inference_substrate="owned native load",
        inference_substrate_details={"device": "gpu"},
        inference_substrate_class="model_load_no_generation",
        execution_venue="host",
        started_monotonic_ns=0,
        ended_monotonic_ns=61_000_000_000,
    )
    assert value["model_invoked"] is True
    assert value["invocation_counts"]["model_loads_failed"] == 1
    assert receipt.validate_current_work_receipt(value) == []


def test_req_report_7395_validator_reports_reference_and_span_shapes(tmp_path: Path) -> None:
    """REQ-REPORT-7395 reports malformed sidecars and phase rows directly."""

    value = _valid(tmp_path)
    changed = deepcopy(value)
    changed["phase_spans"] = ["bad"]
    changed["receipt_sidecars"] = ["bad"]
    errors = receipt.validate_current_work_receipt(changed, root=tmp_path)
    assert "phase_span_invalid" in errors
    assert "sidecar_reference_invalid:0" in errors

    changed = deepcopy(value)
    changed["phase_spans"] = [{"phase": "late", "end_s": 3.0}]
    assert "phase_span_exceeds_duration:late" in receipt.validate_current_work_receipt(
        changed, root=tmp_path
    )


def test_req_report_7395_helper_defensive_branches(tmp_path: Path) -> None:
    """REQ-REPORT-7395 names invalid identity, time, transitions, and references."""

    outside = tmp_path.parent / "outside-current-work-sidecar.json"
    receipt.atomic_json(outside, {"value": 1})
    reference = receipt.sidecar_reference(outside, root=tmp_path, scope="historical_model_receipts")
    assert reference["path"] == str(outside.resolve())
    with pytest.raises(ValueError, match="sidecar_scope_invalid"):
        receipt.sidecar_reference(outside, root=tmp_path, scope="current")
    with pytest.raises(FileNotFoundError):
        receipt.sidecar_reference(
            tmp_path / "missing.json", root=tmp_path, scope="historical_model_receipts"
        )

    malformed = [
        {**_event("attempted", 1), "monotonic_ns": False},
        {**_event("attempted", 2), "call_id": "changed"},
        {
            **_event("completed", 3),
            "call_id": "changed",
            "operation": "generation",
        },
        {**_event("attempted", 4), "call_id": "double"},
        {**_event("failed", 5), "call_id": "double"},
        {**_event("cancelled", 6), "call_id": "double"},
    ]
    value = _valid(tmp_path)
    value["current_invocation_events"] = malformed
    value["event_count"] = len(malformed)
    value["event_sha256"] = receipt.canonical_hash(malformed)
    errors = receipt.validate_current_work_receipt(value, malformed, root=tmp_path)
    assert "event_time_invalid:load-1" in errors
    assert "operation_changed:changed" in errors
    assert "duplicate_terminal:double" in errors

    with pytest.raises(ValueError, match="monotonic_boundary_order_invalid"):
        receipt.build_current_work_receipt(
            run_id="test-run",
            owner_pid=42,
            events=[],
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_details={},
            inference_substrate_class="aggregation",
            execution_venue="host",
            started_monotonic_ns=2,
            ended_monotonic_ns=1,
        )
    with pytest.raises(ValueError, match="invalid_current_event_ledger"):
        receipt.build_current_work_receipt(
            run_id="test-run",
            owner_pid=42,
            events=[_event("attempted", 1)],
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_details={},
            inference_substrate_class="aggregation",
            execution_venue="host",
            started_monotonic_ns=0,
            ended_monotonic_ns=1,
        )

    changed = _valid(tmp_path)
    changed.update(
        inference_substrate_details="bad",
        current_run_id=None,
        current_owner_pid=None,
        started_monotonic_ns=None,
        ended_monotonic_ns=None,
        duration_s=False,
    )
    errors = receipt.validate_current_work_receipt(changed, root=tmp_path)
    assert "inference_substrate_details_invalid" in errors
    assert "current_owner_identity_invalid" in errors
    assert "monotonic_boundaries_invalid" in errors

    changed = _valid(tmp_path)
    changed["ended_monotonic_ns"] = 0
    assert "monotonic_boundary_order_invalid" in receipt.validate_current_work_receipt(
        changed, root=tmp_path
    )
    changed = _valid(tmp_path)
    changed["duration_s"] = False
    assert "duration_invalid" in receipt.validate_current_work_receipt(changed, root=tmp_path)
