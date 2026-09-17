"""Tests for the V647 host-only hardware placement envelope.

Spec refs: REQ-REPORT-7379 and SCENARIO-REPORT-7379-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7379_v647_hardware_envelope as experiment
from carnot.reporting.experiment_7303_validation_scope import (
    REQUIRED_CHECK_NAMES,
    validation_outcome,
)


ROOT = Path(__file__).resolve().parents[2]


def _validation(*, failed: str | None = None) -> dict[str, Any]:
    """Build exact scoped receipts without starting nested child processes."""

    receipts = [
        {
            "name": name,
            "command": f"check {name}",
            "command_argv": ["check", name],
            "scope": "explicit_exp7379_scope",
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
    """Write one private operator receipt for the future-eligibility branch."""

    path.write_text(
        json.dumps(
            {
                "exists": True,
                "receipt_date": "20260917",
                "source": "operator directive 2026-09-17T12:00:00Z: GateMate cable changed",
                "operator_authored": True,
                "board": "Cologne Chip GateMate A1-EVB-2M",
                "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
                "jtag": "operator-confirmed changed cable path",
                "changes": [
                    {
                        "field": "cable",
                        "description": "operator changed the GateMate JTAG cable",
                    }
                ],
                "action": "future bounded detect task",
            }
        ),
        encoding="utf-8",
    )
    return path


# REQ-REPORT-7379 / SCENARIO-REPORT-7379-SOURCES
def test_authenticates_board_rows_receipts_and_complete_cost_null(tmp_path: Path) -> None:
    checks, hashes, context = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )

    hard_checks = [row for row in checks if row["category"] == "required_source"]
    assert hard_checks and all(row["passed"] for row in hard_checks)
    assert context["board_source"]["experiment_id"] == 7367
    assert context["board_source"]["verdict_class"] == "blocked"
    assert context["board_source"]["board_disposition_complete_score"] == 1
    assert context["cost_source"] == {
        "experiment_id": 7340,
        "status": "complete",
        "verdict_class": "null",
        "native_cost_complete_score": 1,
        "native_ten_x_score": 0,
        "complete_boundary_gate_passed": False,
    }
    assert context["receipt_authentication"] == {
        "KV260": True,
        "GateMate": True,
        "PolarFire": True,
    }
    assert hashes[experiment.BOARD_SOURCE_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.COST_SOURCE_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.ISING_SOURCE_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.MEMORY_SOURCE_PATH.as_posix()] is None


# REQ-REPORT-7379 / SCENARIO-REPORT-7379-PLACEMENT
def test_missing_and_disqualified_inputs_emit_exact_unavailable_rows() -> None:
    rows = experiment.load_placement_envelope(ROOT)
    by_source = {row["source_experiment"]: row for row in rows}

    assert set(by_source) == {"Exp7374", "Exp7378"}
    assert by_source["Exp7374"]["outcome"] == "placement_input_unavailable"
    assert by_source["Exp7374"]["source_path"] == experiment.MEMORY_SOURCE_PATH.as_posix()
    assert by_source["Exp7374"]["source_class"] == "missing"
    assert by_source["Exp7374"]["failed_field"] == "path"
    assert by_source["Exp7374"]["observed_value"] == "missing"
    assert by_source["Exp7378"]["source_class"] == "disqualified"
    assert by_source["Exp7378"]["failed_field"] == "verdict_class"
    assert by_source["Exp7378"]["observed_value"] == "disqualified"
    assert all(row["performance_evidence_eligible"] is False for row in rows)
    assert all(row["measured_replaceable_fraction"] is None for row in rows)
    assert all(row["bounded_full_service_speedup"] is None for row in rows)
    assert all(row["full_path_version_upload_count"] is None for row in rows)
    assert all(row["update_cadence_queries"] is None for row in rows)
    assert all(row["recommendation"] == "retain_cpu" for row in rows)


# REQ-REPORT-7379 / SCENARIO-REPORT-7379-AMDAHL
def test_measured_service_rows_get_finite_and_infinite_amdahl_bounds() -> None:
    source = {
        "experiment_id": "eligible-fixture",
        "status": "complete",
        "verdict_class": "positive",
        "flagged_adversarial": False,
        "required_checks_passed": True,
        "placement_service_rows": [
            {
                "unit_id": "query-1",
                "solve_time_ns": 50,
                "certificate_discovery_time_ns": 20,
                "certificate_check_time_ns": 10,
                "certificate_update_time_ns": 5,
                "serialization_time_ns": 5,
                "host_overhead_time_ns": 10,
                "total_service_time_ns": 100,
                "replaceable_components": [
                    "certificate_discovery_time_ns",
                    "certificate_check_time_ns",
                ],
                "full_path_version_upload_count": 2,
                "update_cadence_queries": 8,
            }
        ],
    }
    rows = experiment.assess_placement_source(
        "Fixture", Path("fixture.json"), "eligible-fixture", source, kernel_rate=100.0
    )
    row = rows[0]

    assert row["service_time_decomposition_ns"] == {
        "solve": 50,
        "certificate_discovery": 20,
        "certificate_check": 10,
        "certificate_update": 5,
        "serialization": 5,
        "host_overhead": 10,
        "total": 100,
    }
    assert row["measured_replaceable_fraction"] == pytest.approx(0.3)
    assert row["unaccelerated_fraction"] == pytest.approx(0.7)
    assert row["bounded_full_service_speedup"] == pytest.approx(1 / 0.703)
    assert row["infinite_kernel_upper_bound"] == pytest.approx(1 / 0.7)
    assert row["hundred_x_necessary_condition"]["unaccelerated_fraction_max"] == 0.01
    assert row["hundred_x_necessary_condition"]["passed"] is False
    assert row["full_path_version_upload_count"] == 2
    assert row["update_cadence_queries"] == 8
    assert row["recommendation"] == "retain_cpu"

    bound = experiment.amdahl_bounds(0.995, 100.0)
    assert bound["bounded_full_service_speedup"] == pytest.approx(1 / 0.01495)
    assert bound["infinite_kernel_upper_bound"] == pytest.approx(200.0)
    assert bound["hundred_x_necessary_condition"]["passed"] is True
    with pytest.raises(ValueError, match="replaceable_fraction"):
        experiment.amdahl_bounds(1.1, 100.0)
    with pytest.raises(ValueError, match="kernel_rate"):
        experiment.amdahl_bounds(0.5, 0.0)


# REQ-REPORT-7379 / SCENARIO-REPORT-7379-PLACEMENT
def test_incomplete_or_inconsistent_service_rows_fail_closed() -> None:
    base = {
        "experiment_id": "fixture",
        "status": "complete",
        "verdict_class": "positive",
        "flagged_adversarial": False,
        "required_checks_passed": True,
        "placement_service_rows": [{}],
    }
    missing = experiment.assess_placement_source("Fixture", Path("fixture.json"), "fixture", base)[
        0
    ]
    assert missing["outcome"] == "placement_input_unavailable"
    assert missing["failed_field"] == "placement_service_rows[0].solve_time_ns"

    inconsistent = deepcopy(base)
    inconsistent["placement_service_rows"] = [
        {
            "unit_id": "bad-total",
            "solve_time_ns": 1,
            "certificate_discovery_time_ns": 1,
            "certificate_check_time_ns": 1,
            "certificate_update_time_ns": 1,
            "serialization_time_ns": 1,
            "host_overhead_time_ns": 1,
            "total_service_time_ns": 99,
            "replaceable_components": ["solve_time_ns"],
            "full_path_version_upload_count": 1,
            "update_cadence_queries": 1,
        }
    ]
    bad = experiment.assess_placement_source(
        "Fixture", Path("fixture.json"), "fixture", inconsistent
    )[0]
    assert bad["failed_field"] == "placement_service_rows[0].total_service_time_ns"
    assert bad["expected_value"] == 6
    assert bad["observed_value"] == 99

    no_rows = deepcopy(base)
    no_rows["placement_service_rows"] = []
    assert (
        experiment.assess_placement_source("Fixture", Path("fixture.json"), "fixture", no_rows)[0][
            "failed_field"
        ]
        == "placement_service_rows"
    )

    complete_row = deepcopy(inconsistent["placement_service_rows"][0])
    complete_row["total_service_time_ns"] = 6
    defensive_cases = (
        ("solve_time_ns", -1, "placement_service_rows[0].solve_time_ns"),
        (
            "replaceable_components",
            ["not_a_stage"],
            "placement_service_rows[0].replaceable_components",
        ),
        (
            "full_path_version_upload_count",
            -1,
            "placement_service_rows[0].full_path_version_upload_count",
        ),
        ("update_cadence_queries", 0, "placement_service_rows[0].update_cadence_queries"),
    )
    for field, value, expected_field in defensive_cases:
        changed = deepcopy(base)
        changed["placement_service_rows"] = [{**complete_row, field: value}]
        assert (
            experiment.assess_placement_source("Fixture", Path("fixture.json"), "fixture", changed)[
                0
            ]["failed_field"]
            == expected_field
        )

    assert experiment._receipt_authentication(ROOT, {"board_rows": [None]}) == {}
    assert experiment.build_board_rows({"board_rows": [None]}, {"exists": False}) == []


# REQ-REPORT-7379 / SCENARIO-REPORT-7379-BOARDS
def test_blocked_gatemate_finishes_board_accounting_without_hardware_claim(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )
    boards = {row["board"]: row for row in artifact["board_rows"]}

    assert set(boards) == {"KV260", "GateMate", "PolarFire"}
    assert boards["KV260"]["last_authenticated_venue"] == "kv260_fpga_fabric"
    assert boards["KV260"]["future_access"] == "ssh_only"
    assert boards["KV260"]["architecture_limit"] == "k_max<=5"
    assert boards["PolarFire"]["last_authenticated_venue"] == "polarfire_linux_cpu"
    assert boards["PolarFire"]["fpga_sampling_claimed"] is False
    assert boards["GateMate"]["terminal_state"] == "blocked_changed_physical_state"
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_changed_physical_state:")
    assert artifact["changed_state_receipt"] is None
    assert artifact["board_disposition_complete_score"] == 1
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["native_cost_boundary"]["native_ten_x_score"] == 0
    assert artifact["native_cost_boundary"]["complete_boundary_gate_passed"] is False
    assert artifact["placement_recommendation"] == "retain_cpu"
    assert artifact["amdahl_protocol"]["hundred_x_unaccelerated_fraction_max"] == 0.01
    assert set(artifact["invocation_counts"]["current"]["loads"].values()) == {0}
    assert set(artifact["invocation_counts"]["current"]["generations"].values()) == {0}
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "aggregation"
    assert set(artifact["hardware_operations"].values()) == {0}
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7379 / SCENARIO-REPORT-7379-BOARDS
def test_wishlist_and_changed_receipt_never_create_readiness(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "artifact"),
        _validation(),
        candidate_paths=[_operator_receipt(tmp_path / "operator.json")],
    )
    devices = {row["device"]: row for row in artifact["device_reconciliation"]}
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")

    assert artifact["changed_state_receipt"]["receipt_date"] == "20260917"
    assert artifact["honest_verdict"].startswith("blocked_placement_input_unavailable:")
    assert gate["terminal_state"] == "future_bounded_task_eligible"
    assert gate["new_hardware_execution_claimed"] is False
    assert devices["Extropic Z1T"]["class"] == "vendor_announcement"
    assert devices["Extropic Z1T"]["vendor_source_date"] == "20260904"
    assert devices["Extropic Z1T"]["measured_carnot_gain"] is None
    assert devices["TSU"]["access_assumed"] is False
    assert devices["NPU"]["access_assumed"] is False
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7379 / SCENARIO-REPORT-7379-ARTIFACT
def test_validation_failure_and_cold_replay_fail_closed(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    failed = experiment.build_artifact(
        ROOT,
        paths,
        _validation(failed="focused_pytest"),
        candidate_paths=[tmp_path / "missing.json"],
    )
    assert failed["status"] == "complete"
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
    changed["board_rows"][0]["last_authenticated_venue"] = "host"
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    experiment.atomic_json(paths.terminal_candidate, changed)
    assert "candidate_board_rows_mismatch" in experiment.cold_validate_candidate(paths)

    changed = deepcopy(artifact)
    changed["promotion_score"] = 1
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    assert "scores" in experiment.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "checksum" in experiment.validate_artifact(changed)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        experiment.write_artifact(tmp_path / "invalid.json", {})


# REQ-REPORT-7379 / SCENARIO-REPORT-7379-PLACEMENT
def test_source_block_and_fully_measured_terminal_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    monkeypatch.undo()
    source = {
        "experiment_id": "fixture",
        "status": "complete",
        "verdict_class": "positive",
        "flagged_adversarial": False,
        "required_checks_passed": True,
        "placement_service_rows": [
            {
                "unit_id": "query",
                "solve_time_ns": 1,
                "certificate_discovery_time_ns": 1,
                "certificate_check_time_ns": 1,
                "certificate_update_time_ns": 1,
                "serialization_time_ns": 1,
                "host_overhead_time_ns": 1,
                "total_service_time_ns": 6,
                "replaceable_components": ["solve_time_ns"],
                "full_path_version_upload_count": 0,
                "update_cadence_queries": 1,
            }
        ],
    }
    row = experiment.assess_placement_source("FixtureA", Path("fixture-a.json"), "fixture", source)[
        0
    ]
    other = deepcopy(row)
    other["unit_id"] = "placement:FixtureB:query"
    other["source_experiment"] = "FixtureB"
    other["source_path"] = "fixture-b.json"
    other["row_sha256"] = experiment.canonical_hash(
        {key: value for key, value in other.items() if key != "row_sha256"}
    )
    monkeypatch.setattr(experiment, "load_placement_envelope", lambda _root: [row, other])
    complete = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "complete"),
        _validation(),
        candidate_paths=[_operator_receipt(tmp_path / "complete-operator.json")],
    )
    assert complete["status"] == "complete"
    assert complete["verdict_class"] == "null"
    assert complete["honest_verdict"].startswith("complete_null:")


# REQ-REPORT-7379 / SCENARIO-REPORT-7379-ARTIFACT
def test_exp7358_plan_is_exact_and_entrypoint_is_thin(tmp_path: Path) -> None:
    commands = experiment.build_validation_plan(ROOT, tmp_path / "private")
    terminal = experiment.terminal_commands(
        ROOT, experiment.ExperimentPaths.under(tmp_path / "terminal")
    )

    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert experiment.validate_validation_plan(ROOT, commands) == []
    assert all(
        "tests/python/test_experiment_7379_v647_hardware_envelope.py" in command.argv
        for command in commands
        if command.name in {"focused_pytest", "changed_module_coverage"}
    )
    assert not any("full_python_suite" in command.name for command in commands)
    assert [row.spec.name for row in terminal] == [
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7379_v647_hardware_envelope import main" in source
    assert source.count("main()") == 1
