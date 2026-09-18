"""Tests for the V649 producer-side current-work receipt protocol.

Spec refs: REQ-REPORT-7395 and SCENARIO-REPORT-7395-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7383_v648_canary_reducer as reducer
from carnot import experiment_7395_v649_receipt_protocol as mod
from carnot.reporting import current_work_receipt as receipt
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from scripts.adversarial_verify import verify_artifact
from scripts.verdict_row_consistency_lint import check_artifact


ROOT = Path(__file__).resolve().parents[2]


def _event(
    call_id: str,
    operation: str,
    state: str,
    monotonic_ns: int,
    *,
    run_id: str = "run-7395",
    owner_pid: int = 7395,
) -> dict[str, Any]:
    """Build one owned event with the fields required by REQ-REPORT-7395."""

    row: dict[str, Any] = {
        "scope": "current",
        "transport": "owned_runtime",
        "run_id": run_id,
        "owner_pid": owner_pid,
        "call_id": call_id,
        "operation": operation,
        "state": state,
        "monotonic_ns": monotonic_ns,
    }
    if state == "failed":
        row["error"] = "owned failure"
    return row


def _build_receipt(
    events: list[dict[str, Any]],
    *,
    sidecars: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic helper receipt for mutation tests."""

    return receipt.build_current_work_receipt(
        run_id="run-7395",
        owner_pid=7395,
        events=events,
        inference_substrate="aggregation_from_upstream_artifacts -- host CPU receipt reduction",
        inference_substrate_details={"device": "test-cpu", "python": "test"},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=1_000_000_000,
        ended_monotonic_ns=2_000_000_000,
        sidecar_references=sidecars or [],
    )


def _command_receipt(name: str, *, passed: bool = True) -> dict[str, Any]:
    """Build the exact fields consumed by the Exp7395 cold reducer."""

    row: dict[str, Any] = {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "command_environment": {"COVERAGE_FILE": "/tmp/exp7395/.coverage"},
        "scope": "unit_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "output_tail": "ok" if passed else "failed",
    }
    if name == "worktree_imports":
        row["resolved_imports"] = {
            "carnot.experiment_7383_v648_canary_reducer": str(
                (ROOT / reducer.MODULE_PATH).resolve()
            ),
            "carnot.experiment_7395_v649_receipt_protocol": str((ROOT / mod.MODULE_PATH).resolve()),
            "carnot.reporting.current_work_receipt": str(
                (ROOT / mod.RECEIPT_HELPER_PATH).resolve()
            ),
        }
    return row


def _passing_receipts(*, terminal: bool = True) -> list[dict[str, Any]]:
    """Return one receipt for every frozen affected and terminal check."""

    names = list(mod.AFFECTED_CHECK_NAMES)
    if terminal:
        names.extend(mod.TERMINAL_CHECK_NAMES)
    return [_command_receipt(name) for name in names]


def test_scenario_report_7395_current_counts_only_owned_events(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-CURRENT derives counters from owned events only."""

    historical = receipt.write_immutable_sidecar(
        tmp_path / "historical.json",
        scope="historical_model_receipts",
        payload={
            "model_invoked": True,
            "invocation_counts": {"generation_calls_attempted": 99},
        },
        root=tmp_path,
    )
    scripted = receipt.write_immutable_sidecar(
        tmp_path / "scripted.json",
        scope="simulated_transport_events",
        payload={"simulated_transport_events": [{"call_id": "scripted-1"}]},
        root=tmp_path,
    )
    value = _build_receipt([], sidecars=[historical, scripted])
    assert value["model_invoked"] is False
    assert value["invocation_counts"] == receipt.ZERO_INVOCATION_COUNTS
    assert isinstance(value["inference_substrate"], str)
    assert value["inference_substrate_details"]["device"] == "test-cpu"
    assert value["execution_venue"] == "host"
    assert value["duration_s"] == 1.0
    assert all(set(row) == {"path", "sha256", "scope"} for row in value["receipt_sidecars"])
    assert receipt.validate_current_work_receipt(value, [], root=tmp_path) == []


def test_scenario_report_7395_current_attempts_include_failed_loads() -> None:
    """SCENARIO-REPORT-7395-CURRENT counts a failed real load as invoked."""

    events = [
        _event("load-1", "model_load", "attempted", 1_100_000_000),
        _event("load-1", "model_load", "failed", 1_200_000_000),
    ]
    value = _build_receipt(events)
    assert value["model_invoked"] is True
    assert value["invocation_counts"]["model_loads_attempted"] == 1
    assert value["invocation_counts"]["model_loads_failed"] == 1
    assert value["invocation_counts"]["model_loads_in_flight"] == 0
    assert receipt.validate_current_work_receipt(value, events, root=ROOT) == []


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("unreported_load", "model_invoked_mismatch"),
        ("falsified_duration", "duration_mismatch"),
        ("dropped_failed_call", "unfinished_call:load-1"),
        ("missing_completion", "unfinished_call:load-1"),
        ("invalid_venue", "execution_venue_invalid"),
        ("changed_source_hash", "sidecar_hash_mismatch"),
    ],
)
def test_scenario_report_7395_mutations_fail_at_producer_boundary(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    """SCENARIO-REPORT-7395-MUTATIONS rejects each named receipt defect."""

    sidecar = receipt.write_immutable_sidecar(
        tmp_path / "history.json",
        scope="historical_model_receipts",
        payload={"original_verdict_class": "disqualified"},
        root=tmp_path,
    )
    complete = [
        _event("load-1", "model_load", "attempted", 1_100_000_000),
        _event("load-1", "model_load", "failed", 1_200_000_000),
    ]
    current_events = complete
    value = _build_receipt(complete, sidecars=[sidecar])
    if mutation == "unreported_load":
        value["model_invoked"] = False
    elif mutation == "falsified_duration":
        value["duration_s"] = 0.01
    elif mutation == "dropped_failed_call":
        current_events = complete[:-1]
    elif mutation == "missing_completion":
        current_events = [_event("load-1", "model_load", "attempted", 1_100_000_000)]
        value = _build_receipt([], sidecars=[sidecar])
        value["event_sha256"] = receipt.canonical_hash(current_events)
        value["event_count"] = 1
    elif mutation == "invalid_venue":
        value["execution_venue"] = "host_cpu"
    elif mutation == "changed_source_hash":
        value["receipt_sidecars"][0]["sha256"] = "sha256:" + "0" * 64
    errors = receipt.validate_current_work_receipt(value, current_events, root=tmp_path)
    assert any(expected_error in error for error in errors)


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([_event("load", "model_load", "completed", 2)], "terminal_without_attempt"),
        (
            [
                _event("load", "model_load", "attempted", 2),
                _event("load", "model_load", "attempted", 3),
                _event("load", "model_load", "completed", 4),
            ],
            "duplicate_attempt",
        ),
        (
            [
                _event("load", "model_load", "attempted", 4),
                _event("load", "model_load", "completed", 3),
            ],
            "event_time_regression",
        ),
        ([_event("load", "unknown", "attempted", 2)], "operation_invalid"),
        ([_event("load", "model_load", "unknown", 2)], "state_invalid"),
        (
            [_event("load", "model_load", "attempted", 2, owner_pid=1)],
            "event_not_owned",
        ),
    ],
)
def test_req_report_7395_rejects_contradictory_current_ledgers(
    events: list[dict[str, Any]], expected: str
) -> None:
    """REQ-REPORT-7395 rejects malformed, foreign, and contradictory ledgers."""

    value = _build_receipt([])
    value["event_sha256"] = receipt.canonical_hash(events)
    value["event_count"] = len(events)
    errors = receipt.validate_current_work_receipt(value, events, root=ROOT)
    assert any(expected in error for error in errors)


def test_req_report_7395_sidecars_are_immutable_and_scoped(tmp_path: Path) -> None:
    """REQ-REPORT-7395 does not overwrite or mis-scope preserved evidence."""

    path = tmp_path / "sidecar.json"
    first = receipt.write_immutable_sidecar(
        path,
        scope="simulated_transport_events",
        payload={"simulated_transport_events": []},
        root=tmp_path,
    )
    assert (
        receipt.write_immutable_sidecar(
            path,
            scope="simulated_transport_events",
            payload={"simulated_transport_events": []},
            root=tmp_path,
        )
        == first
    )
    with pytest.raises(FileExistsError, match="immutable_sidecar_conflict"):
        receipt.write_immutable_sidecar(
            path,
            scope="simulated_transport_events",
            payload={"simulated_transport_events": [{"changed": True}]},
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="sidecar_scope_invalid"):
        receipt.write_immutable_sidecar(
            tmp_path / "bad.json", scope="current", payload={}, root=tmp_path
        )


def test_scenario_report_7395_reducer_preserves_numeric_and_sat_boundaries() -> None:
    """SCENARIO-REPORT-7395-REDUCER keeps parsing, fidelity, and SAT separate."""

    result = mod.reduce_assignment_boundary(ROOT)
    assignment = result["assignment_reduction"]
    proof = result["proof_boundary_replay"]
    assert assignment["usable_proposal_count"] == 4
    assert assignment["response_fidelity_count"] == 4
    assert assignment["sat_extendible_count"] == 0
    assert assignment["assignment_reducer_ready_score"] == 1
    assert proof["proof_boundary_replay_ready_score"] == 1
    assert proof["errors"] == []

    calls, formulas = reducer.load_assignment_evidence(ROOT)
    changed = deepcopy(calls)
    changed[0]["raw_reply"] = '{"assignments":[1.0,-2]}'
    changed[0]["raw_reply_sha256"] = reducer.sha256_text(changed[0]["raw_reply"])
    changed[0]["raw_response"]["choices"][0]["message"]["content"] = changed[0]["raw_reply"]
    changed[0]["decoded_assignments"] = None
    invalid = reducer.reduce_assignment_receipts(changed, formulas)
    assert "variable_reference_invalid:0" in invalid["errors"]
    assert invalid["rows"][0]["sat_extendible"] is False


def test_scenario_report_7395_validation_plan_freezes_eight_checks(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-VALIDATION freezes scope and coverage environment."""

    commands = mod.build_validation_plan(ROOT, tmp_path / "private")
    assert tuple(command.name for command in commands) == mod.AFFECTED_CHECK_NAMES
    assert len(commands) == 8
    assert mod.validate_validation_plan(ROOT, commands) == []
    assert all(
        dict(getattr(command, "command_environment", ())).get("COVERAGE_FILE")
        for command in commands
    )
    broad = [*commands, CommandSpec("full_python_suite", ("pytest", "tests/python"), "broad")]
    errors = mod.validate_validation_plan(ROOT, broad)
    assert "unexpected_command:full_python_suite" in errors
    assert "full_python_suite_forbidden" in errors


def test_scenario_report_7395_contract_retains_stale_markdown_authority() -> None:
    """SCENARIO-REPORT-7395-CONTRACT preserves the actual advisory mismatch."""

    comparison = mod.build_contract_comparison(ROOT)
    assert comparison["yaml_milestone"] == mod.MILESTONE
    assert comparison["yaml_task_count"] == 14
    assert comparison["markdown_milestone"] == "2026.09.648"
    assert comparison["contract_complete_score"] == 0
    assert len(comparison["contract_rows"]) == 14
    assert any(row["failures"] for row in comparison["contract_rows"])
    controls = mod.run_contract_mutation_controls(comparison["yaml_contract"])
    assert [row["mutation"] for row in controls] == [
        "missing_row",
        "reordered_rows",
        "wrong_field",
    ]
    assert all(row["passed"] for row in controls)


def test_scenario_report_7395_current_fixture_passes_unchanged_readers(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-MUTATIONS exercises both unchanged artifact readers."""

    helper_receipt = _build_receipt([])
    fixture = mod.build_reader_fixture(helper_receipt)
    path = tmp_path / "clean.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    report = verify_artifact(path)
    assert report["flag_count"] == 0
    assert check_artifact(path) == ("ok", [])
    assert fixture["model_invoked"] is False
    assert fixture["fresh_model_evidence"] is False


def test_scenario_report_7395_artifact_cold_reducer_detects_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-ARTIFACT recomputes scores, hashes, and provenance."""

    artifact = mod.build_artifact_for_test(
        ROOT,
        sidecar_dir=tmp_path / "sidecars",
        validation_receipts=_passing_receipts(),
    )
    assert (
        mod.validate_artifact(
            artifact, root=ROOT, sidecar_root=tmp_path / "sidecars", require_terminal=True
        )
        == []
    )
    assert artifact["assignment_reducer_ready_score"] == 1
    assert artifact["receipt_protocol_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert isinstance(artifact["inference_substrate"], str)
    assert artifact["receipt_sidecars"]
    assert all(set(row) == {"path", "sha256", "scope"} for row in artifact["receipt_sidecars"])
    assert len(artifact["receipt_mutation_rows"]) == 6

    forged = deepcopy(artifact)
    forged["receipt_protocol_ready_score"] = 0
    forged["reproducibility_checksum"] = mod.artifact_checksum(forged)
    assert "terminal_state_mismatch" in mod.validate_artifact(
        forged, root=ROOT, sidecar_root=tmp_path / "sidecars", require_terminal=True
    )

    forged = deepcopy(artifact)
    forged["inference_substrate"] = {"device": "cpu"}
    forged["reproducibility_checksum"] = mod.artifact_checksum(forged)
    assert "current_work_receipt_mismatch" in mod.validate_artifact(
        forged, root=ROOT, sidecar_root=tmp_path / "sidecars", require_terminal=True
    )

    forged = deepcopy(artifact)
    forged["source_artifact_hashes"]["missing.json"] = "sha256:bad"
    forged["reproducibility_checksum"] = mod.artifact_checksum(forged)
    assert "source_hash_mismatch:missing.json" in mod.validate_artifact(
        forged, root=ROOT, sidecar_root=tmp_path / "sidecars", require_terminal=True
    )


def test_req_report_7395_blocked_artifact_names_exact_failed_gate(tmp_path: Path) -> None:
    """REQ-REPORT-7395 maps absent unchanged inputs to an exact blocked receipt."""

    preconditions, hashes = mod.collect_preconditions(tmp_path)
    artifact = mod.build_blocked_artifact(
        preconditions,
        source_hashes=hashes,
        started_at_utc="2026-09-18T00:00:00Z",
        duration_s=0.01,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["check"]
    assert artifact["gate_check_summary"]["upstream"]
    assert artifact["gate_check_summary"]["artifact_field"]
    assert artifact["gate_check_summary"]["expected"] is not None
    assert artifact["gate_check_summary"]["observed"] is None
    assert artifact["assignment_reducer_ready_score"] == 0
    assert artifact["receipt_protocol_ready_score"] == 0


def test_req_report_7395_helper_rejects_shape_and_time_errors() -> None:
    """REQ-REPORT-7395 rejects non-string substrate and invalid time bounds."""

    with pytest.raises(ValueError, match="inference_substrate_must_be_string"):
        receipt.build_current_work_receipt(
            run_id="run",
            owner_pid=os.getpid(),
            events=[],
            inference_substrate={"device": "cpu"},  # type: ignore[arg-type]
            inference_substrate_details={},
            inference_substrate_class="aggregation",
            execution_venue="host",
            started_monotonic_ns=2,
            ended_monotonic_ns=1,
        )
    invalid = _build_receipt([])
    invalid["receipt_sidecars"] = [{"path": "x", "sha256": "bad", "scope": "current"}]
    errors = receipt.validate_current_work_receipt(invalid, [], root=ROOT)
    assert "sidecar_scope_invalid:0" in errors
    assert "sidecar_hash_invalid:0" in errors


def test_req_report_7395_parsers_and_contract_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7395 covers malformed JSON, YAML, and both contract parsers."""

    assert mod.load_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[1]", encoding="utf-8")
    assert mod.load_json(bad_json) == {}
    written = tmp_path / "written.json"
    mod.atomic_json(written, {"value": 1})
    assert mod.load_json(written) == {"value": 1}

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("[", encoding="utf-8")
    assert mod.load_yaml(bad_yaml) == {}
    roadmap = mod.load_yaml(ROOT / mod.ROADMAP_PATH)
    assert len(mod.yaml_contract_rows(roadmap)) == 14
    broken = mod.compare_contract_authorities("not a contract", {})
    assert any(error.startswith("markdown_parse_error:") for error in broken["errors"])
    assert any(error.startswith("yaml_parse_error:") for error in broken["errors"])
    assert "markdown_task_count_mismatch" in broken["errors"]
    assert "yaml_task_count_mismatch" in broken["errors"]

    (tmp_path / mod.ROADMAP_PATH).write_text("tasks: bad\n", encoding="utf-8")
    design = tmp_path / mod.DESIGN_PATH
    design.parent.mkdir(parents=True, exist_ok=True)
    design.write_text("not a contract\n", encoding="utf-8")
    comparison = mod.build_contract_comparison(tmp_path)
    assert comparison["markdown_contract"]["tasks"] == []
    assert comparison["yaml_contract"]["tasks"] == []


def test_req_report_7395_artifact_classification_and_validator_branches(tmp_path: Path) -> None:
    """REQ-REPORT-7395 separates blocked inputs from failed owned validation."""

    checks, hashes = mod.collect_preconditions(ROOT)
    blocked_checks = deepcopy(checks)
    blocked_checks[0]["passed"] = False
    blocked = mod.build_artifact(
        root=ROOT,
        sidecar_dir=tmp_path / "blocked",
        preconditions=blocked_checks,
        source_hashes=hashes,
        receipts=_passing_receipts(),
        started_at="2026-09-18T00:00:00Z",
        ended_at="2026-09-18T00:00:02Z",
        duration_s=2.0,
        phase_spans=[{"phase": "test", "start_s": 0.0, "end_s": 1.0}],
    )
    assert blocked["verdict_class"] == "blocked"

    failed_receipts = _passing_receipts()
    failed_receipts[0]["passed"] = False
    failed_receipts[0]["exit_code"] = 1
    failed = mod.build_artifact(
        root=ROOT,
        sidecar_dir=tmp_path / "failed",
        preconditions=checks,
        source_hashes=hashes,
        receipts=failed_receipts,
        started_at="2026-09-18T00:00:00Z",
        ended_at="2026-09-18T00:00:02Z",
        duration_s=2.0,
        phase_spans=[{"phase": "test", "start_s": 0.0, "end_s": 1.0}],
    )
    assert failed["verdict_class"] == "disqualified"

    assert mod.validate_artifact([]) == ["artifact_not_object"]
    assert mod.validate_artifact({})[0].startswith("missing_required_field:")
    artifact = mod.build_artifact_for_test(ROOT, tmp_path / "valid", _passing_receipts())
    changed = deepcopy(artifact)
    changed.update(
        schema="wrong",
        MODEL_SPECS=["unexpected"],
        inference_substrate_class="no_model_load",
        source_artifact_hashes={"missing.json": "sha256:bad"},
        field_principles={},
        assignment_replay={},
        promotion_score=1,
        assignment_reducer_ready_score=0,
        receipt_protocol_ready_score=0,
        reproducibility_checksum="sha256:bad",
    )
    errors = mod.validate_artifact(
        changed, root=ROOT, sidecar_root=tmp_path / "valid", require_terminal=True
    )
    assert {
        "identity_invalid",
        "current_invocation_declaration_invalid",
        "current_substrate_declaration_invalid",
        "source_hash_mismatch:missing.json",
        "field_principles_incomplete",
        "assignment_replay_mismatch",
        "assignment_reducer_ready_score_mismatch",
        "receipt_protocol_ready_score_mismatch",
        "promotion_score_nonzero",
        "reproducibility_checksum_mismatch",
    }.issubset(errors)

    blocked_artifact = mod.build_blocked_artifact(
        [{"passed": False, "observed": None}],
        started_at="2026-09-18T00:00:00Z",
        duration_s=0.1,
    )
    blocked_artifact["assignment_reducer_ready_score"] = 1
    blocked_artifact["reproducibility_checksum"] = mod.artifact_checksum(blocked_artifact)
    assert "blocked_scores_nonzero" in mod.validate_artifact(blocked_artifact, root=ROOT)

    terminal = mod.terminal_command_specs(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(mod.TERMINAL_CHECK_NAMES)
