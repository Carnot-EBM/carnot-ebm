"""Tests for the V648 host-only hardware placement reducer.

Spec refs: REQ-REPORT-7393 and SCENARIO-REPORT-7393-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7393_v648_hardware_placement as experiment
from carnot.reporting.experiment_7303_validation_scope import (
    REQUIRED_CHECK_NAMES,
    validation_outcome,
)


ROOT = Path(__file__).resolve().parents[2]


def _validation(*, failed: str | None = None) -> dict[str, Any]:
    """Build exact scoped receipts without starting child processes."""

    receipts = [
        {
            "name": name,
            "command": f"check {name}",
            "command_argv": ["check", name],
            "command_environment": {"COVERAGE_FILE": "/tmp/exp7393.coverage"},
            "scope": "explicit_exp7393_scope",
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


# REQ-REPORT-7393 / SCENARIO-REPORT-7393-SOURCES
def test_authenticates_hardware_and_current_stage_sources(tmp_path: Path) -> None:
    checks, hashes, context = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )

    hard = [row for row in checks if row["category"] == "required_source"]
    assert hard and all(row["passed"] for row in hard)
    assert context["hardware_source"] == {
        "experiment_id": 7379,
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "board_disposition_complete_score": 1,
        "authenticated_board_receipts": {
            "GateMate": True,
            "KV260": True,
            "PolarFire": True,
        },
    }
    assert context["stage_sources"]["Exp7385"]["verdict_class"] == "null"
    assert context["stage_sources"]["Exp7386"]["verdict_class"] == "disqualified"
    assert context["stage_sources"]["Exp7386"]["flagged_adversarial"] is True
    assert context["stage_sources"]["Exp7389"]["available"] is False
    assert hashes[experiment.HARDWARE_SOURCE_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.TRAINING_SOURCE_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.ONLINE_SOURCE_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.PROOF_SOURCE_PATH.as_posix()] is None


# REQ-REPORT-7393 / SCENARIO-REPORT-7393-SOURCES
def test_stage_reduction_keeps_invalid_and_missing_inputs_unavailable() -> None:
    rows = experiment.load_placement_envelope(ROOT)
    by_source = {row["source_experiment"]: row for row in rows}

    assert set(by_source) == {"Exp7385", "Exp7386", "Exp7389"}
    training = by_source["Exp7385"]
    assert training["source_evidence_eligible"] is True
    assert training["stage_costs_s"]["prediction"] > 0
    assert training["stage_costs_s"]["update"] is None
    assert training["stage_costs_s"]["certificate"] is None
    assert training["stage_costs_s"]["serialization"] is None
    assert training["stage_costs_s"]["orchestration"] is None
    assert training["complete_boundary_eligible"] is False
    assert set(training["stage_fractions"].values()) == {None}
    assert training["failed_field"] == "complete_service_time_s"
    assert training["bounded_full_service_speedup"] is None

    online = by_source["Exp7386"]
    assert online["source_evidence_eligible"] is False
    assert online["source_class"] == "disqualified"
    assert online["failed_field"] == "verdict_class"
    assert online["observed_value"] == "disqualified"
    assert online["diagnostic_measurements_used_for_readiness"] is False

    proof = by_source["Exp7389"]
    assert proof["source_class"] == "missing"
    assert proof["failed_field"] == "path"
    assert proof["observed_value"] == "missing"
    assert proof["stage_costs_s"] == experiment.EMPTY_STAGE_COSTS


# REQ-REPORT-7393 / SCENARIO-REPORT-7393-CHECKPOINT
def test_actual_numeric_checkpoint_fixes_state_and_transfer_size() -> None:
    measurement = experiment.measure_selector_checkpoint(ROOT)

    assert measurement["source_experiment"] == "Exp7385"
    assert measurement["arm"] == "natural_prevalence_bernoulli_gibbs"
    assert measurement["seed"] == 7382001
    assert measurement["architecture"] == {
        "input_dim": 2,
        "hidden_dims": [4],
        "output_dim": 1,
    }
    assert measurement["numeric_parameter_count"] == 17
    assert (
        measurement["checkpoint_file_bytes"]
        == (ROOT / measurement["checkpoint_path"]).stat().st_size
    )
    assert measurement["checkpoint_sha256"] == experiment.sha256_file(
        ROOT / measurement["checkpoint_path"]
    )
    assert measurement["assumed_device_numeric_format"] == "float32"
    assert measurement["estimated_minimum_transfer_bytes"] == 68
    assert measurement["transfer_is_measured"] is False


# REQ-REPORT-7393 / SCENARIO-REPORT-7393-PLACEMENT
def test_complete_service_fixture_gets_stage_fractions_and_amdahl_bounds() -> None:
    row = experiment.reduce_complete_service(
        {
            "prediction": 10.0,
            "update": 20.0,
            "certificate": 30.0,
            "serialization": 5.0,
            "orchestration": 35.0,
        },
        assumed_device_rate=100.0,
    )

    assert row["complete_service_time_s"] == 100.0
    assert row["stage_fractions"] == {
        "prediction": pytest.approx(0.10),
        "update": pytest.approx(0.20),
        "certificate": pytest.approx(0.30),
        "serialization": pytest.approx(0.05),
        "orchestration": pytest.approx(0.35),
    }
    assert row["measured_replaceable_fraction"] == pytest.approx(0.65)
    assert row["unaccelerated_fraction"] == pytest.approx(0.35)
    assert row["bounded_full_service_speedup"] == pytest.approx(1 / 0.3565)
    assert row["infinite_device_upper_bound"] == pytest.approx(1 / 0.35)
    assert row["hundred_x_necessary_condition"]["passed"] is False
    assert row["recommendation"] == "retain_cpu_or_batch_before_port"

    with pytest.raises(ValueError, match="stage costs"):
        experiment.reduce_complete_service({"prediction": 1.0})
    with pytest.raises(ValueError, match="nonnegative finite"):
        experiment.reduce_complete_service({name: -1.0 for name in experiment.STAGE_NAMES})
    with pytest.raises(ValueError, match="positive finite"):
        experiment.reduce_complete_service(
            {name: 1.0 for name in experiment.STAGE_NAMES}, assumed_device_rate=0.0
        )
    with pytest.raises(ValueError, match="complete service time"):
        experiment.reduce_complete_service({name: 0.0 for name in experiment.STAGE_NAMES})


# REQ-REPORT-7393 / SCENARIO-REPORT-7393-BOARDS
def test_unchanged_gatemate_blocks_without_erasing_three_board_rows(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing-receipt.json"],
    )
    boards = {row["board"]: row for row in artifact["board_rows"]}
    devices = {row["device"]: row for row in artifact["device_paths"]}

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
    assert artifact["gate_check_summary"]["first_failure"]["check"] == (
        "gatemate_changed_physical_state_receipt"
    )
    assert artifact["board_disposition_complete_score"] == 1
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["placement_recommendation"] == "retain_cpu_or_batch_before_port"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert set(artifact["hardware_operations"].values()) == {0}
    assert devices["CPU"]["available_for_current_task"] is True
    assert devices["GPU"]["operation_performed"] is False
    assert devices["NPU"]["driver_installed"] is False
    assert devices["sparse proof-check FPGA"]["operation_performed"] is False
    assert devices["Extropic Z1T"]["available_for_current_task"] is False
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7393 / SCENARIO-REPORT-7393-ARTIFACT
def test_validation_failure_and_cold_replay_fail_closed(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    failed = experiment.build_artifact(
        ROOT,
        paths,
        _validation(failed="focused_pytest"),
        candidate_paths=[tmp_path / "missing-receipt.json"],
    )
    assert failed["status"] == "complete"
    assert failed["verdict_class"] == "disqualified"
    assert failed["board_disposition_complete_score"] == 0
    assert experiment.validate_artifact(failed) == []

    artifact = experiment.build_artifact(
        ROOT,
        paths,
        _validation(),
        candidate_paths=[tmp_path / "missing-receipt.json"],
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
    changed["placement_envelope_rows"][0]["source_class"] = "positive"
    changed["reproducibility_checksum"] = experiment.reproducibility_checksum(changed)
    assert "placement_rows" in experiment.validate_artifact(changed)

    with pytest.raises(ValueError, match="artifact_validation_failed"):
        experiment.write_artifact(tmp_path / "invalid.json", {})


# REQ-REPORT-7393 / SCENARIO-REPORT-7393-SOURCES
def test_defensive_source_and_checkpoint_failures_are_exact(tmp_path: Path) -> None:
    assert (
        experiment._authenticate_hardware_receipts(ROOT, {"board_rows": [None, {"board": None}]})
        == {}
    )
    assert (
        experiment.build_board_rows(
            {"board_rows": [None, {"board": "not-a-board"}]}, {"exists": False}
        )
        == []
    )
    assert experiment._measured_prediction_cost({"rows": [None]}) is None

    missing = experiment.assess_stage_source(
        "Fixture", Path("missing.json"), "fixture-id", None, capture_field="capture"
    )
    assert missing["failed_field"] == "path"

    invalid = experiment.assess_stage_source(
        "Fixture",
        Path("fixture.json"),
        "fixture-id",
        {
            "experiment_id": "fixture-id",
            "status": "complete",
            "verdict_class": "positive",
            "flagged_adversarial": False,
            "capture": 1,
            "acceptance_gate_results": [{"category": "required_validation", "passed": True}],
            "rows": [{"measured_cost": {"cpu_scoring_duration_s": "bad"}}],
        },
        capture_field="capture",
    )
    assert invalid["failed_field"] == "rows.measured_cost.cpu_scoring_duration_s"

    malformed = tmp_path / "checkpoint.json"
    malformed.write_text(json.dumps({"weights": {"w": [1.0, "bad"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="numeric checkpoint"):
        experiment.measure_numeric_checkpoint(
            malformed,
            expected_hash=experiment.sha256_file(malformed),
            arm="fixture",
            seed=1,
        )

    for payload, message in (
        ({"weights": {"w": [True]}}, "boolean"),
        ({"weights": {"w": [float("inf")]}}, "non-finite"),
        ({"not_weights": []}, "missing weights"),
    ):
        path = tmp_path / f"{message}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            experiment.measure_numeric_checkpoint(
                path,
                expected_hash=experiment.sha256_file(path),
                arm="fixture",
                seed=1,
            )
    with pytest.raises(ValueError, match="hash mismatch"):
        experiment.measure_numeric_checkpoint(
            malformed,
            expected_hash="sha256:" + "0" * 64,
            arm="fixture",
            seed=1,
        )


# REQ-REPORT-7393 / SCENARIO-REPORT-7393-ARTIFACT
def test_exp7358_plan_is_exact_and_entrypoint_is_thin(tmp_path: Path) -> None:
    commands = experiment.build_validation_plan(ROOT, tmp_path / "private")
    terminal = experiment.terminal_commands(
        ROOT, experiment.ExperimentPaths.under(tmp_path / "terminal")
    )

    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert experiment.validate_validation_plan(ROOT, commands) == []
    assert all(
        "tests/python/test_experiment_7393_v648_hardware_placement.py" in command.argv
        for command in commands
        if command.name in {"focused_pytest", "changed_module_coverage"}
    )
    assert not any("full_python_suite" in command.name for command in commands)
    coverage_report = next(
        command for command in commands if command.name == "changed_module_coverage_report"
    )
    assert "COVERAGE_FILE" in dict(coverage_report.command_environment)
    assert [row.spec.name for row in terminal] == [
        "independent_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7393_v648_hardware_placement import main" in source
    assert source.count("main()") == 1
