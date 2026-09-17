"""Tests for the V646 read-only board disposition.

Spec refs: REQ-REPORT-7367 and SCENARIO-REPORT-7367-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7367_v646_board_disposition as experiment
from carnot.reporting.experiment_7303_validation_scope import (
    REQUIRED_CHECK_NAMES,
    validation_outcome,
)


ROOT = Path(__file__).resolve().parents[2]


def _validation(*, failed: str | None = None) -> dict[str, Any]:
    """Create exact affected-check receipts without nested test processes."""

    receipts = [
        {
            "name": name,
            "command": f"check {name}",
            "command_argv": ["check", name],
            "scope": "explicit_exp7367_scope",
            "exit_code": int(name == failed),
            "duration_s": 0.01,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "1" * 64,
            "passed": name != failed,
            "timed_out": False,
        }
        for name in REQUIRED_CHECK_NAMES
    ]
    return validation_outcome(receipts, [])


def _operator_receipt(path: Path) -> Path:
    """Write a private receipt that can enable only a future bounded task."""

    path.write_text(
        json.dumps(
            {
                "exists": True,
                "receipt_date": "20260917",
                "source": "operator directive 2026-09-17T12:00:00Z: GateMate JTAG cable changed",
                "operator_authored": True,
                "board": "Cologne Chip GateMate A1-EVB-2M",
                "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
                "jtag": "operator-confirmed changed cable path",
                "changes": [
                    {
                        "field": "jtag",
                        "description": "operator changed the GateMate JTAG cable path",
                    }
                ],
                "action": "future bounded detect task",
            }
        ),
        encoding="utf-8",
    )
    return path


# REQ-REPORT-7367 / SCENARIO-REPORT-7367-SOURCES
def test_authenticates_board_and_native_cost_without_promoting_them(tmp_path: Path) -> None:
    checks, hashes, context = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )

    assert all(row["passed"] for row in checks)
    assert context["board_source"]["experiment_id"] == 7355
    assert context["board_source"]["status"] == "blocked"
    assert context["board_source"]["diagnostic_only"] is True
    assert context["board_source"]["authorizes_promotion"] is False
    assert context["cost_source"]["experiment_id"] == 7340
    assert context["cost_source"]["status"] == "complete"
    assert context["cost_source"]["verdict_class"] == "null"
    assert context["cost_source"]["native_ten_x_score"] == 0
    assert context["validation_boundary"]["validation_contract_ready_score"] == 1
    assert hashes[experiment.BOARD_SOURCE_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.COST_SOURCE_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.HISTORICAL_MODEL_SIDECAR_PATH.as_posix()].startswith("sha256:")


# REQ-REPORT-7367 / SCENARIO-REPORT-7367-ROWS
def test_three_rows_retain_venue_denominator_date_and_claim_boundary(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )
    rows = {row["board"]: row for row in artifact["board_rows"]}

    assert set(rows) == {"KV260", "PolarFire", "GateMate"}
    assert rows["KV260"]["execution_venue"] == "kv260_fpga_fabric"
    assert rows["KV260"]["denominator"] == {"unit": "fabric_samples", "count": 32}
    assert rows["KV260"]["evidence_date"] == "20260915"
    assert rows["KV260"]["availability_class"] == "graduated_historical"
    assert rows["PolarFire"]["execution_venue"] == "polarfire_linux_cpu"
    assert rows["PolarFire"]["denominator"] == {"unit": "cpu_dispatches", "count": 1}
    assert rows["PolarFire"]["fpga_sampling_claimed"] is False
    assert rows["GateMate"]["execution_venue"] == "none_read_only"
    assert rows["GateMate"]["denominator"] == {"unit": "new_hardware_runs", "count": 0}
    assert rows["GateMate"]["availability_class"] == "blocked"
    assert all(row["native_tenfold_speed_gate"] is None for row in rows.values())
    assert artifact["native_tenfold_speed_gate"] is None
    assert artifact["native_cost_context"]["complete_boundary_gate_passed"] is False
    assert artifact["native_cost_context"]["inner_kernel_combined_with_service"] is False
    assert artifact["board_disposition_complete_score"] == 1
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0


# REQ-REPORT-7367 / SCENARIO-REPORT-7367-GATEMATE
def test_missing_operator_receipt_is_a_terminal_external_block(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_changed_physical_state:")
    assert artifact["changed_state_receipt"] is None
    assert artifact["gate_check_summary"]["first_failure"] == {
        "upstream": "physical_state_receipt",
        "check": "gatemate_changed_physical_state_receipt",
        "field": "receipt_date/operator_authored/provenance/changed_field",
        "expected_value": experiment.PHYSICAL_RECEIPT_CONTRACT,
        "observed_value": {
            "accepted_receipt_count": 0,
            "selected_source_path": None,
            "absence": experiment.MISSING_RECEIPT,
        },
    }
    assert artifact["hardware_operations"] == {
        "usb": 0,
        "ssh": 0,
        "jtag": 0,
        "fpga": 0,
        "rocm": 0,
        "thermodynamic_devices": 0,
        "purchases": 0,
        "vendor_contacts": 0,
    }
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"]["current"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7367 / SCENARIO-REPORT-7367-WISHLIST
def test_wishlist_separates_owned_history_block_and_vendor_announcement(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )
    devices = {row["device"]: row for row in artifact["device_reconciliation"]}

    assert devices["KV260"]["class"] == "owned_graduated_historical"
    assert devices["PolarFire"]["class"] == "owned_graduated_historical"
    assert devices["GateMate"]["class"] == "owned_blocked"
    assert devices["Extropic Z1"]["class"] == "vendor_announced"
    assert devices["Extropic Z1"]["owned_by_carnot"] is False
    assert devices["Extropic Z1"]["availability"] == "vendor early access targeted for 2027"
    assert devices["Extropic Z1"]["source_date"] == "20260917"
    assert devices["Extropic Z1"]["runtime_claimed"] is False
    assert devices["Extropic Z1"]["speed_claimed"] is False


# REQ-REPORT-7367 / SCENARIO-REPORT-7367-GATEMATE
def test_new_receipt_only_changes_future_eligibility(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "artifact"),
        _validation(),
        candidate_paths=[_operator_receipt(tmp_path / "operator.json")],
    )
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")

    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["changed_state_receipt"]["receipt_date"] == "20260917"
    assert gate["availability_class"] == "future_bounded_task_eligible"
    assert gate["new_hardware_execution_claimed"] is False
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert set(artifact["hardware_operations"].values()) == {0}
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7367 / SCENARIO-REPORT-7367-ARTIFACT
def test_validation_failure_disqualifies_and_cold_replay_detects_mutation(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    failed = experiment.build_artifact(
        ROOT,
        paths,
        _validation(failed="focused_pytest"),
        candidate_paths=[tmp_path / "missing.json"],
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["board_disposition_complete_score"] == 0
    assert failed["hardware_ready_score"] == 0
    assert experiment.validate_artifact(failed) == []

    artifact = experiment.build_artifact(
        ROOT,
        paths,
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )
    experiment.atomic_json(paths.terminal_candidate, artifact)
    assert experiment.cold_validate_candidate(paths) == []
    receipt = experiment.write_artifact(paths.artifact, artifact)
    assert receipt["sha256"] == experiment.sha256_file(paths.artifact)

    changed = deepcopy(artifact)
    changed["board_rows"][0]["execution_venue"] = "host"
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    experiment.atomic_json(paths.terminal_candidate, changed)
    assert "candidate_rows_mismatch" in experiment.cold_validate_candidate(paths)

    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    assert "scores" in experiment.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "checksum" in experiment.validate_artifact(changed)

    with pytest.raises(ValueError, match="artifact_validation_failed"):
        experiment.write_artifact(tmp_path / "invalid.json", {})


# REQ-REPORT-7367 / SCENARIO-REPORT-7367-ARTIFACT
def test_exp7358_plan_is_explicit_and_entrypoint_is_thin(tmp_path: Path) -> None:
    commands = experiment.build_validation_plan(ROOT, tmp_path / "private")
    terminal = experiment._terminal_commands(
        ROOT, experiment.ExperimentPaths.under(tmp_path / "terminal")
    )

    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert experiment.validate_validation_plan(ROOT, commands) == []
    assert all(
        "tests/python/test_experiment_7367_v646_board_disposition.py" in command.argv
        for command in commands
        if command.name in {"focused_pytest", "changed_module_coverage"}
    )
    assert [row.spec.name for row in terminal] == [
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    assert [row.spec.scope for row in terminal] == [
        "independent_replay",
        "safety",
        "safety",
    ]
    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7367_v646_board_disposition import main" in source
    assert source.count("main()") == 1


# REQ-REPORT-7367 / SCENARIO-REPORT-7367-SOURCES
def test_invalid_source_identity_and_row_shapes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    board = json.loads((ROOT / experiment.BOARD_SOURCE_PATH).read_text(encoding="utf-8"))
    cost = json.loads((ROOT / experiment.COST_SOURCE_PATH).read_text(encoding="utf-8"))
    physical = {"exists": False, "accepted_receipt_count": 0}

    assert experiment.build_board_rows({"board_rows": [None]}, cost, physical) == []
    invalid_board = deepcopy(board)
    invalid_board["experiment_id"] = 1
    assert "board_source_identity" in experiment.source_errors(invalid_board, cost)
    invalid_cost = deepcopy(cost)
    invalid_cost["native_ten_x_score"] = 1
    assert "native_tenfold_boundary" in experiment.source_errors(board, invalid_cost)

    malformed = tmp_path / "list.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required JSON object"):
        experiment._load_object(malformed)

    paths = experiment.ExperimentPaths.under(tmp_path / "blocked")
    checks, hashes, context = experiment.collect_preconditions(ROOT, paths)
    checks[0]["passed"] = False
    monkeypatch.setattr(
        experiment,
        "collect_preconditions",
        lambda _root, _paths: (checks, hashes, context),
    )
    blocked = experiment.build_artifact(
        ROOT,
        paths,
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )
    assert blocked["honest_verdict"].startswith("blocked_source_precondition:")


# REQ-REPORT-7367 / SCENARIO-REPORT-7367-ARTIFACT
def test_cold_replay_rejects_score_and_source_hash_drift(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.build_artifact(
        ROOT,
        paths,
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )

    changed = deepcopy(artifact)
    changed["board_disposition_complete_score"] = 0
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    experiment.atomic_json(paths.terminal_candidate, changed)
    assert "candidate_score_mismatch" in experiment.cold_validate_candidate(paths)

    experiment.atomic_json(paths.terminal_candidate, artifact)
    raw = json.loads(paths.raw_rows.read_text(encoding="utf-8"))
    raw["source_hashes"][experiment.BOARD_SOURCE_PATH.as_posix()] = "sha256:bad"
    experiment.atomic_json(paths.raw_rows, raw)
    assert "candidate_source_hashes_mismatch" in experiment.cold_validate_candidate(paths)
