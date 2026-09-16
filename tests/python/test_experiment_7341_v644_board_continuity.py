"""Tests for the V644 read-only board disposition.

Spec refs: REQ-REPORT-7341 and SCENARIO-REPORT-7341-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7341_v644_board_continuity as experiment
from carnot.reporting.experiment_7303_validation_scope import (
    REQUIRED_CHECK_NAMES,
    validation_outcome,
)


ROOT = Path(__file__).resolve().parents[2]


def _validation(*, failed: str | None = None) -> dict[str, Any]:
    """Build realistic validator input without nesting pytest subprocesses."""

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
    """Write one private operator receipt that changes a permitted field."""

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


# REQ-REPORT-7341 / SCENARIO-REPORT-7341-SOURCES
def test_preconditions_authenticate_diagnostic_chain_without_consuming_blocked_score(
    tmp_path: Path,
) -> None:
    checks, hashes, context = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )

    assert all(row["passed"] for row in checks)
    assert context["diagnostic_receipt"]["experiment_id"] == 7327
    assert context["diagnostic_receipt"]["verdict_class"] == "blocked"
    assert context["diagnostic_receipt"]["board_continuity_complete_score"] == 0
    assert context["diagnostic_used_as_readiness_gate"] is False
    assert context["graduated_receipt"]["experiment_id"] == 7314
    assert context["reference_observation"]["all_match"] is True
    assert context["scope_observation"] == {
        "kv260_fabric_execution": True,
        "kv260_processor_class": "fpga_fabric",
        "polarfire_cpu_dispatch": True,
        "polarfire_processor_class": "cpu",
        "polarfire_fpga_sampling": False,
    }
    assert context["downstream_science_gate_references"] == []
    assert hashes[experiment.DIAGNOSTIC_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.GRADUATED_PATH.as_posix()].startswith("sha256:")
    assert hashes["results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"]
    assert hashes["results/raw/experiment_7231/polarfire_dispatch.json"]


# REQ-REPORT-7341 / SCENARIO-REPORT-7341-GATEMATE
def test_missing_receipt_records_three_terminal_dispositions(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, context = experiment.collect_preconditions(ROOT, paths)
    physical = experiment.search_physical_state_receipts(
        ROOT, paths.physical_state_search, candidate_paths=[tmp_path / "missing.json"]
    )

    rows = experiment.build_board_rows(context["diagnostic_receipt"], physical)
    by_board = {row["board"]: row for row in rows}

    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["processor_class"] == "fpga_fabric"
    assert by_board["KV260"]["fabric_execution_completed"] is True
    assert by_board["KV260"]["present_availability_asserted"] is False
    assert by_board["KV260"]["exact_next_condition"] == (
        "none; any future access uses ssh kria only"
    )
    assert by_board["PolarFire"]["processor_class"] == "cpu"
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert by_board["GateMate"]["disposition"] == "blocked_changed_physical_state"
    assert by_board["GateMate"]["accepted_receipt_count"] == 0
    assert by_board["GateMate"]["error"] == experiment.MISSING_RECEIPT
    assert all(row["hardware_operations_issued"] == [] for row in rows)
    assert experiment.reduce_board_rows(rows)["board_disposition_complete_score"] == 1


# REQ-REPORT-7341 / SCENARIO-REPORT-7341-ACCOUNTING
def test_external_block_keeps_accounting_complete_and_readiness_zero(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_changed_physical_state:")
    assert artifact["board_disposition_complete_score"] == 1
    assert artifact["hardware_readiness_score"] == 0
    assert artifact["hardware_promotion_score"] == 0
    assert artifact["board_execution_promotion_score"] == 0
    assert artifact["scientific_value_score"] == 0
    assert artifact["hardware_operations_issued_count"] == 0
    assert artifact["purchase_required"] is False
    assert artifact["physical_state_receipt"]["accepted_receipt_count"] == 0
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
    assert artifact["gate_check_summary"]["first_failure"]["observed_value"] == {
        "accepted_receipt_count": 0,
        "selected_source_path": None,
        "absence": experiment.MISSING_RECEIPT,
    }
    assert (
        artifact["source_artifact_states"][experiment.DIAGNOSTIC_PATH.as_posix()][
            "used_as_readiness_gate"
        ]
        is False
    )
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7341 / SCENARIO-REPORT-7341-CONTEXT
def test_v644_hardware_path_keeps_extropic_and_kan_external(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )

    assert artifact["deployment_relevance"]["scan"] == "V644"
    assert artifact["deployment_relevance"]["extropic"]["carnot_speedup_claimed"] is False
    assert artifact["deployment_relevance"]["kan"]["replacement_authorized"] is False
    assert artifact["deployment_relevance"]["purchase_required"] is False
    assert artifact["hardware_path"]["current_execution"] == "host_read_only_aggregation"
    assert artifact["hardware_path"]["kv260_future_access"] == "ssh kria only"
    assert artifact["hardware_path"]["polarfire_scope"] == "cpu_dispatch_not_fpga_sampling"
    assert artifact["hardware_path"]["gatemate_state"] == (
        "blocked_pending_dated_operator_physical_change"
    )
    assert artifact["next_hardware_conditions"]["gatemate"]["satisfied_by"] == "operator"


# REQ-REPORT-7341 / SCENARIO-REPORT-7341-GATEMATE
def test_new_receipt_records_eligibility_but_issues_no_operation(tmp_path: Path) -> None:
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
    assert artifact["hardware_readiness_score"] == 0
    assert artifact["physical_state_receipt"]["accepted_receipt_count"] == 1
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")
    assert gate["disposition"] == "changed_physical_state_future_experiment_eligible"
    assert gate["exact_next_condition"] == experiment.GATEMATE_FUTURE_ACTION
    assert gate["hardware_command_count"] == 0
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7341 / SCENARIO-REPORT-7341-ARTIFACT
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
    assert artifact["hardware_readiness_score"] == 0
    assert artifact["hardware_promotion_score"] == 0
    assert len(artifact["board_rows"]) == 3
    assert artifact["failed_required_commands"] == ["focused_pytest"]
    assert experiment.validate_artifact(artifact) == []


# REQ-REPORT-7341 / SCENARIO-REPORT-7341-ARTIFACT
def test_cold_reducer_atomic_writer_and_entrypoint_fail_closed(tmp_path: Path) -> None:
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
        ({"hardware_readiness_score": 1}, "readiness_or_promotion"),
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

    unserializable = deepcopy(artifact)
    unserializable["validation_receipts"] = {object()}
    assert "checksum" in experiment.validate_artifact(unserializable)

    with pytest.raises(ValueError, match="artifact_validation_failed"):
        experiment.write_artifact(tmp_path / "invalid.json", {})

    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7341_v644_board_continuity import main" in source
    assert source.count("main()") == 1


# REQ-REPORT-7341 / SCENARIO-REPORT-7341-SOURCES
def test_malformed_contract_and_external_precondition_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert experiment.read_task_contract(tmp_path / "missing") is None
    roadmap = tmp_path / experiment.ROADMAP_PATH
    roadmap.write_text("[]\n", encoding="utf-8")
    assert experiment.read_task_contract(tmp_path) is None
    roadmap.write_text("tasks: []\n", encoding="utf-8")
    assert experiment.read_task_contract(tmp_path) is None

    assert experiment._downstream_science_gate_references(tmp_path / "missing") == [
        {"task_id": "unreadable_roadmap", "gate": None}
    ]
    roadmap.write_text("[]\n", encoding="utf-8")
    assert experiment._downstream_science_gate_references(tmp_path) == [
        {"task_id": "malformed_roadmap", "gate": None}
    ]
    roadmap.write_text(
        "tasks:\n"
        "  - not-a-mapping\n"
        "  - id: before\n"
        f"  - id: {experiment.TASK_ID}\n"
        "  - id: after\n"
        f"    gated_on: {{producer: {experiment.TASK_ID}, field: score}}\n",
        encoding="utf-8",
    )
    references = experiment._downstream_science_gate_references(tmp_path)
    assert references[0]["task_id"] == "after"

    assert experiment.build_board_rows({"board_rows": [None]}, {}) == []
    assert experiment._normalized_validation({"validation_receipts": [None]}) == []

    failure = {
        "check": "diagnostic_identity",
        "upstream": experiment.DIAGNOSTIC_PATH.as_posix(),
        "field": "producer identity and terminal class",
        "expected_value": {"verdict_class": "blocked"},
        "observed_value": {"verdict_class": "disqualified"},
        "passed": False,
    }
    monkeypatch.setattr(
        experiment,
        "collect_preconditions",
        lambda _root, _paths: (
            [failure],
            {},
            {
                "diagnostic_receipt": {},
                "graduated_receipt": {},
                "diagnostic_quarantine": {},
                "graduated_quarantine": {},
                "diagnostic_checksum_matches": False,
                "graduated_checksum_matches": False,
                "diagnostic_used_as_readiness_gate": False,
                "downstream_science_gate_references": [],
            },
        ),
    )

    artifact = experiment.build_artifact(
        ROOT, experiment.ExperimentPaths.under(tmp_path / "artifact"), _validation()
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_external_precondition:")
    assert artifact["board_rows"] == []
    assert artifact["board_disposition_complete_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"]["observed_value"] == {
        "verdict_class": "disqualified"
    }
    assert experiment.validate_artifact(artifact) == []
