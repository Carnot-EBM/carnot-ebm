"""Tests for REQ-CL-7613 and REQ-HW-7613.

The tests require raw exclusive spans. Aggregate speed cannot substitute for
the missing service-placement evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_7613_v664_service_attribution as exp
from carnot.pipeline.calibrated_decision_service import CalibratedDecisionService


ROOT = Path(__file__).resolve().parents[2]
RUST_BINARY = ROOT / "target/release/portable-recalibration-service"


@pytest.fixture(scope="module", autouse=True)
def built_worker() -> None:
    """Build the changed worker because telemetry is a process-boundary feature."""

    completed = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "carnot-core",
            "--bin",
            "portable-recalibration-service",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr


def test_scenario_cl_7613_telemetry_is_default_off_and_explicit_opt_in(
    tmp_path: Path,
) -> None:
    """REQ-CL-7613; telemetry changes no public default."""

    with CalibratedDecisionService(
        state_path=tmp_path / "off.json",
        binary_path=RUST_BINARY,
    ) as service:
        decision = service.predict("off", 0.25)
        ack = service.release_feedback("off", 1)
        assert decision.exclusive_stage_ns == {}
        assert ack.exclusive_stage_ns == {}

    with CalibratedDecisionService(
        state_path=tmp_path / "on.json",
        binary_path=RUST_BINARY,
        telemetry_enabled=True,
    ) as service:
        decision = service.predict("on", 0.25)
        ack = service.release_feedback("on", 1)
        assert service.telemetry_enabled is True
        assert decision.available and ack.durable
        assert set(ack.exclusive_stage_ns) == set(exp.WORKER_STAGE_NAMES)
        assert all(value >= 0 for value in ack.exclusive_stage_ns.values())
        assert ack.caller_stage_ns["caller_encode"] >= 0
        assert ack.caller_stage_ns["caller_wait_decode"] >= 0


def test_scenario_cl_7613_exclusive_rows_reconcile() -> None:
    """SCENARIO-CL-7613-ATTRIBUTION uses durations, not cross-clock stamps."""

    rows = exp.synthetic_stage_rows()
    reduction = exp.reduce_stage_rows(rows)

    assert reduction["stage_attribution_ready_score"] == 1
    assert set(reduction["strata"]) == {"cold:1", "cold:8", "warm:1", "warm:8"}
    for row in rows:
        assert row["clock_domains"] == {
            "caller": "caller_process_monotonic_duration",
            "worker": "worker_process_monotonic_duration",
        }
        assert row["caller_remainder_label"] == "ipc_scheduling_and_unobserved_caller_work"
        assert row["reconciliation_error_ns"] <= max(1_000, row["whole_service_ns"] * 0.01)
        assert row["cross_clock_timestamp_subtraction"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "negative",
        "double_count",
        "wrong_durability",
        "stage_names",
        "denominator",
        "clock",
        "parity",
        "pair_count",
    ],
)
def test_req_cl_7613_rejects_invalid_attribution(mutation: str) -> None:
    """REQ-CL-7613 rejects negative, overlapping, or unequal service evidence."""

    rows = deepcopy(exp.synthetic_stage_rows())
    if mutation == "negative":
        rows[0]["worker_stage_ns"]["update_arithmetic"] = -1
    elif mutation == "double_count":
        rows[0]["caller_remainder_ns"] += 50_000
    elif mutation == "wrong_durability":
        rows[0]["durability_policy"] = "memory_only"
    elif mutation == "stage_names":
        rows[0]["worker_stage_ns"].pop("encoding")
    elif mutation == "denominator":
        rows[0]["whole_service_ns"] = 0
    elif mutation == "clock":
        rows[0]["cross_clock_timestamp_subtraction"] = True
    elif mutation == "parity":
        rows[0]["parity"] = False
    else:
        rows.pop()

    with pytest.raises(ValueError, match="stage_|durability"):
        exp.reduce_stage_rows(rows)


def test_scenario_cl_7613_overhead_controls_never_select_report() -> None:
    """SCENARIO-CL-7613-OVERHEAD fixes ten off blocks per stratum."""

    rows = exp.synthetic_overhead_rows()
    overhead = exp.reduce_instrumentation_overhead(rows)

    assert set(overhead) == {"cold:1", "cold:8", "warm:1", "warm:8"}
    assert all(item["telemetry_on_blocks"] == 10 for item in overhead.values())
    assert all(item["telemetry_off_blocks"] == 10 for item in overhead.values())
    assert all(item["report_selection_authority"] is False for item in overhead.values())

    with pytest.raises(ValueError, match="overhead_block_count"):
        exp.reduce_instrumentation_overhead(rows[:-1])


def test_req_cl_7613_percentile_guards() -> None:
    """REQ-CL-7613 does not reduce an empty uncertainty sample."""

    assert exp._percentile([4.0], 0.5) == 4.0
    with pytest.raises(ValueError, match="percentile_requires_values"):
        exp._percentile([], 0.5)


def test_req_cl_7613_amdahl_uses_whole_durable_denominator() -> None:
    """SCENARIO-CL-7613-TERMINAL computes a bound and its uncertainty."""

    reduction = exp.reduce_stage_rows(exp.synthetic_stage_rows())
    row = reduction["strata"]["warm:8"]

    assert math.isclose(
        row["amdahl_upper_bound"]["estimate"], 1 / (1 - row["arithmetic_fraction"]["estimate"])
    )
    assert row["amdahl_upper_bound"]["kind"] == "upper_bound_not_measured_speedup"
    assert row["denominator"] == "whole_durable_request"
    assert row["projected_hardware_latency"] is None


def test_req_hw_7613_preserves_all_board_scopes_without_operations(tmp_path: Path) -> None:
    """REQ-HW-7613; SCENARIO-HW-7613-REPORTING."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    boards = {row["board"]: row for row in artifact["board_rows"]}

    assert set(boards) == {"KV260", "PolarFire", "GateMate"}
    assert boards["KV260"]["future_access"] == "ssh kria"
    assert boards["KV260"]["k_max"] == 5
    assert boards["PolarFire"]["fpga_sampling_measured"] is False
    assert boards["GateMate"]["disposition"] == "blocked_unchanged_physical_prerequisite"
    assert artifact["hardware_operations_issued"] == []
    assert artifact["acquisition_decision"]["purchase_authorized"] is False


def test_req_cl_7613_terminal_artifact_preserves_null_and_required_fields(
    tmp_path: Path,
) -> None:
    """REQ-CL-7613 preserves Exp7598 and separates every gate category."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)

    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "null"
    assert (
        artifact["prior_exp7598_honest_verdict"]
        == "complete_null_rust_consumer_ready_speed_gate_failed"
    )
    assert artifact["prior_exp7598_aggregate_verdict_preserved"] is True
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["no_model_load"] is True
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["stage_attribution_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["readiness"] is None
    assert artifact["flagged_adversarial"] is False
    assert {gate["category"] for gate in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }
    assert exp.validate_artifact(artifact, root=ROOT) == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("MODEL_SPECS", [{"name": "forbidden"}], "model_specs"),
        ("no_model_load", False, "invocation_counts"),
        ("inference_substrate_class", "live_llm_inference", "inference_substrate_class"),
        ("hardware_operations_issued", ["probe"], "hardware_operations"),
        ("prior_exp7598_honest_verdict", "changed", "prior_verdict"),
        ("prior_exp7598_aggregate_verdict_preserved", False, "prior_verdict_preservation"),
        ("public_client_opt_in_unchanged", False, "production_default"),
        ("projected_hardware_latency", 1.0, "claim_boundary"),
        ("stage_attribution_ready_score", 0, "stage_attribution_ready_score"),
        ("acceptance_gate_results", [], "acceptance_gate_results"),
        ("validation_receipts", [], "validation_receipts"),
        ("field_principles", {}, "field_principles"),
        ("reproducibility_checksum", "changed", "reproducibility_checksum"),
    ],
)
def test_req_cl_7613_validator_rejects_claim_and_custody_drift(
    tmp_path: Path, field: str, value: object, expected: str
) -> None:
    """REQ-CL-7613 verifier guards remain effective under direct mutation."""

    artifact = exp.build_test_artifact(ROOT, tmp_path / field)
    artifact[field] = value
    assert expected in exp.validate_artifact(artifact)


def test_req_cl_7613_validator_rejects_row_board_and_source_drift(tmp_path: Path) -> None:
    """REQ-CL-7613 recomputes raw rows, board scope, and source hashes."""

    artifact = exp.build_test_artifact(ROOT, tmp_path / "base")
    changed = deepcopy(artifact)
    changed["consumer_stage_rows"][0]["whole_service_ns"] += 1
    assert "stage_reduction" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    for row in changed["instrumentation_overhead_rows"]:
        if row["mode"] == "cold" and row["batch_size"] == 1 and row["telemetry_enabled"]:
            row["whole_service_ns"] += 100
    assert "instrumentation_overhead" in exp.validate_artifact(changed)

    for board_name, field, value, expected in (
        ("KV260", "future_access", "block-device", "kv260_scope"),
        ("PolarFire", "fpga_sampling_measured", True, "polarfire_scope"),
        ("GateMate", "disposition", "ready", "gatemate_scope"),
    ):
        changed = deepcopy(artifact)
        next(row for row in changed["board_rows"] if row["board"] == board_name)[field] = value
        assert expected in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"][exp.MODULE_PATH.as_posix()]["sha256"] = "sha256:bad"
    assert any(
        error.startswith("source_hash:") for error in exp.validate_artifact(changed, root=ROOT)
    )


def test_req_cl_7613_readers_recompute_exact_candidate(tmp_path: Path) -> None:
    """SCENARIO-CL-7613-TERMINAL exercises cold and independent readers."""

    artifact = exp.build_test_artifact(ROOT, tmp_path / "artifact")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(candidate, root=ROOT) == []
    assert exp.independent_replay(candidate) == []
    assert exp.independent_replay(tmp_path / "missing.json") == ["artifact_unreadable"]


def test_req_cl_7613_external_blocker_is_complete_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7613 emits complete_blocked without partial measurements."""

    context = exp.collect_preconditions(ROOT)
    context["blocker"] = {
        "check": "missing_input",
        "upstream": "fixture",
        "path": "missing.json",
        "field": "ready_score",
        "op": "eq",
        "expected": 1,
        "observed": None,
    }
    monkeypatch.setattr(exp, "collect_preconditions", lambda _root: context)
    artifact = exp.build_artifact(ROOT, [], [])

    assert artifact["honest_verdict"] == "complete_blocked_missing_input"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"][0] == context["blocker"]
    assert artifact["stage_attribution_ready_score"] == 0
    assert exp.validate_artifact(artifact) == []

    changed = deepcopy(artifact)
    changed["stage_attribution_ready_score"] = 1
    assert "stage_attribution_ready_score" in exp.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["gate_check_summary"] = []
    assert "blocked_gate_check_summary" in exp.validate_artifact(changed)


def test_req_cl_7613_reader_and_validator_defensive_paths(tmp_path: Path) -> None:
    """REQ-CL-7613 fails closed for malformed rows, sources, and summaries."""

    artifact = exp.build_test_artifact(ROOT, tmp_path / "base")
    with pytest.raises(ValueError, match="raw_rows_missing"):
        exp.independent_reduce({})
    assert exp._verify_sources({"source_artifact_hashes": []}, ROOT) == ["source_artifact_hashes"]
    errors = exp._verify_sources({"source_artifact_hashes": {"bad": "not-a-receipt"}}, ROOT)
    assert errors == ["source_receipt:bad"]

    changed = deepcopy(artifact)
    changed.pop("honest_verdict")
    assert any(error.startswith("required_fields:") for error in exp.validate_artifact(changed))
    changed = deepcopy(artifact)
    changed.pop("consumer_stage_rows")
    assert any(error.startswith("reduction:") for error in exp.validate_artifact(changed))
    changed = deepcopy(artifact)
    changed["board_rows"].pop()
    errors = exp.validate_artifact(changed)
    assert "board_count" in errors and "board_identity" in errors
    assert exp.cold_replay(tmp_path / "missing.json", root=ROOT) == ["artifact_unreadable"]

    for field, value, expected in (
        ("stage_reduction", {}, "stage_reduction"),
        ("instrumentation_overhead", {}, "instrumentation_overhead"),
        ("board_rows", [], "board_count"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        candidate = tmp_path / f"{field}.json"
        candidate.write_text(json.dumps(changed), encoding="utf-8")
        assert expected in exp.independent_replay(candidate)

    changed = deepcopy(artifact)
    changed.pop("consumer_stage_rows")
    candidate = tmp_path / "bad-reduction.json"
    candidate.write_text(json.dumps(changed), encoding="utf-8")
    assert exp.independent_replay(candidate) == ["raw_rows_missing"]
