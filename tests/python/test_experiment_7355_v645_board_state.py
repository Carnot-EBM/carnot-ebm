"""Tests for the V645 read-only board-state record.

Spec refs: REQ-REPORT-7355 and SCENARIO-REPORT-7355-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7355_v645_board_state as experiment
from carnot.reporting.experiment_7303_validation_scope import (
    REQUIRED_CHECK_NAMES,
    validation_outcome,
)


ROOT = Path(__file__).resolve().parents[2]


def _validation(*, failed: str | None = None) -> dict[str, Any]:
    """Build current scoped-check receipts without starting nested subprocesses."""

    receipts = [
        {
            "name": name,
            "command": f"check {name}",
            "command_argv": ["check", name],
            "scope": "explicit_test_scope",
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
    """Write one private receipt that makes only a future action eligible."""

    path.write_text(
        json.dumps(
            {
                "exists": True,
                "receipt_date": "20260916",
                "source": (
                    "operator directive 2026-09-16T12:00:00Z: GateMate DirtyJTAG cable changed"
                ),
                "operator_authored": True,
                "board": "Cologne Chip GateMate A1-EVB-2M",
                "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
                "dirtyjtag": "operator-confirmed changed cable path",
                "changes": [
                    {
                        "field": "dirtyjtag",
                        "description": "operator changed the DirtyJTAG cable path",
                    }
                ],
                "action": "detect",
            }
        ),
        encoding="utf-8",
    )
    return path


# REQ-REPORT-7355 / SCENARIO-REPORT-7355-SOURCES
def test_preconditions_authenticate_v644_rows_and_exp6559_cutoff(tmp_path: Path) -> None:
    checks, hashes, context = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )

    assert all(row["passed"] for row in checks)
    assert context["prior_receipt"]["experiment_id"] == 7341
    assert context["prior_receipt"]["verdict_class"] == "blocked"
    assert context["prior_receipt"]["board_disposition_complete_score"] == 1
    assert context["prior_used_as_readiness_gate"] is False
    assert context["reference_observation"]["all_match"] is True
    assert context["scope_observation"] == {
        "kv260_fabric_execution": True,
        "kv260_processor_class": "fpga_fabric",
        "polarfire_cpu_dispatch": True,
        "polarfire_processor_class": "cpu",
        "polarfire_fpga_sampling": False,
    }
    assert context["downstream_science_gate_references"] == []
    assert hashes[experiment.PRIOR_PATH.as_posix()].startswith("sha256:")
    assert (
        hashes[experiment.CUTOFF_PATH.as_posix()]
        == (context["prior_receipt"]["physical_state_receipt"]["cutoff_source_hash"])
    )


# REQ-REPORT-7355 / SCENARIO-REPORT-7355-GATEMATE
def test_missing_receipt_is_terminal_with_three_complete_dispositions(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )
    rows = {row["board"]: row for row in artifact["board_rows"]}

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_changed_physical_state:")
    assert set(rows) == {"KV260", "GateMate", "PolarFire"}
    assert rows["KV260"]["processor_class"] == "fpga_fabric"
    assert rows["KV260"]["present_availability_asserted"] is False
    assert rows["PolarFire"]["processor_class"] == "cpu"
    assert rows["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert rows["GateMate"]["disposition"] == "blocked_changed_physical_state"
    assert rows["GateMate"]["accepted_receipt_count"] == 0
    assert all(row["hardware_operations_issued"] == [] for row in rows.values())
    assert artifact["board_disposition_complete_score"] == 1
    for field in experiment.ZERO_SCORE_FIELDS:
        assert artifact[field] == 0
    assert artifact["hardware_operations_issued_count"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"]["loads"].values()) == {0}
    assert set(artifact["invocation_counts"]["generations"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["gate_check_summary"]["first_failure"]["check"] == (
        "gatemate_changed_physical_state_receipt"
    )
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7355 / SCENARIO-REPORT-7355-CONTEXT
def test_v645_context_keeps_extropic_z1t_and_kan_external(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )

    assert artifact["deployment_relevance"]["scan"] == "V645"
    assert artifact["deployment_relevance"]["extropic"]["device"] == "Z1T"
    assert artifact["deployment_relevance"]["extropic"]["carnot_speedup_claimed"] is False
    assert artifact["deployment_relevance"]["kan"]["deployment_context_only"] is True
    assert artifact["deployment_relevance"]["purchase_required"] is False
    assert artifact["hardware_path"]["kv260_future_access"] == "ssh kria only"
    assert artifact["hardware_path"]["polarfire_scope"] == "cpu_dispatch_not_fpga_sampling"
    assert artifact["next_hardware_conditions"]["gatemate"]["satisfied_by"] == "operator"


# REQ-REPORT-7355 / SCENARIO-REPORT-7355-GATEMATE
def test_new_receipt_only_enables_a_future_bounded_operation(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "artifact"),
        _validation(),
        candidate_paths=[_operator_receipt(tmp_path / "operator.json")],
    )

    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["board_disposition_complete_score"] == 1
    assert artifact["hardware_operations_issued_count"] == 0
    assert artifact["physical_state_receipt"]["accepted_receipt_count"] == 1
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")
    assert gate["disposition"] == "changed_physical_state_future_experiment_eligible"
    assert gate["exact_next_condition"] == experiment.GATEMATE_FUTURE_ACTION
    assert gate["hardware_command_count"] == 0
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7355 / SCENARIO-REPORT-7355-ARTIFACT
def test_validation_failure_disqualifies_without_erasing_rows(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(failed="focused_pytest"),
        candidate_paths=[tmp_path / "missing.json"],
    )

    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified:")
    assert artifact["board_disposition_complete_score"] == 0
    assert len(artifact["board_rows"]) == 3
    assert artifact["failed_required_commands"] == ["focused_pytest"]
    for field in experiment.ZERO_SCORE_FIELDS:
        assert artifact[field] == 0
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7355 / SCENARIO-REPORT-7355-ARTIFACT
def test_cold_reduction_and_atomic_write_fail_closed(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.build_artifact(
        ROOT,
        paths,
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )
    experiment._atomic_json(paths.terminal_candidate, artifact)

    assert experiment.cold_validate_candidate(paths) == []
    receipt = experiment.write_artifact(paths.artifact, artifact)
    assert json.loads(paths.artifact.read_text(encoding="utf-8")) == artifact
    assert receipt["sha256"] == experiment.sha256_file(paths.artifact)

    for mutation, expected in (
        ({"MODEL_SPECS": ["model"]}, "model_declaration"),
        ({"hardware_operations_issued_count": 1}, "operations"),
        ({"hardware_readiness_score": 1}, "readiness_value_or_promotion"),
        ({"hardware_value_score": 1}, "readiness_value_or_promotion"),
        ({"board_disposition_complete_score": 0}, "disposition_score"),
        ({"reproducibility_checksum": "sha256:bad"}, "checksum"),
    ):
        changed = deepcopy(artifact)
        changed.update(mutation)
        if "reproducibility_checksum" not in mutation:
            changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
        assert expected in experiment.validate_artifact(changed)

    changed_candidate = deepcopy(artifact)
    changed_candidate["board_rows"][0]["processor_class"] = "cpu"
    changed_candidate["reproducibility_checksum"] = experiment.artifact_checksum(changed_candidate)
    experiment._atomic_json(paths.terminal_candidate, changed_candidate)
    assert "candidate_rows_mismatch" in experiment.cold_validate_candidate(paths)

    experiment._atomic_json(paths.raw_rows, {"rows": []})
    assert "raw_row_reduction" in experiment.cold_validate_candidate(paths)

    with pytest.raises(ValueError, match="artifact_validation_failed"):
        experiment.write_artifact(tmp_path / "invalid.json", {})

    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7355_v645_board_state import main" in source
    assert source.count("main()") == 1


# REQ-REPORT-7355 / SCENARIO-REPORT-7355-SOURCES
def test_downstream_gate_and_malformed_row_are_rejected(tmp_path: Path) -> None:
    roadmap = tmp_path / experiment.ROADMAP_PATH
    roadmap.write_text(
        "tasks:\n"
        "  - not-a-mapping\n"
        f"  - id: {experiment.TASK_ID}\n"
        "  - id: dependent-science\n"
        f"    gated_on: {{producer: {experiment.TASK_ID}, field: score}}\n",
        encoding="utf-8",
    )

    references = experiment._downstream_science_gate_references(tmp_path)

    assert references == [
        {
            "task_id": "dependent-science",
            "gate": {"producer": experiment.TASK_ID, "field": "score"},
        }
    ]
    assert experiment.build_board_rows(ROOT, {"board_rows": [None]}, {}) == []
