"""Tests for the V643 read-only board-continuity receipt.

Spec refs: REQ-ISING-7327 and SCENARIO-ISING-7327-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7327_v643_board_continuity as experiment
from carnot.reporting.experiment_7303_validation_scope import (
    REQUIRED_CHECK_NAMES,
    validation_outcome,
)


ROOT = Path(__file__).resolve().parents[2]


def _validation(*, failed: str | None = None) -> dict[str, Any]:
    """Build real reducer input without running child pytest from a unit test."""

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


def _valid_operator_receipt(path: Path) -> Path:
    """Write one private operator record that meets the shipped receipt grammar."""

    path.write_text(
        json.dumps(
            {
                "exists": True,
                "receipt_date": "20260915",
                "source": (
                    "operator directive 2026-09-15T12:00:00Z: GateMate DirtyJTAG cable changed"
                ),
                "operator_authored": True,
                "board": "Cologne Chip GateMate A1-EVB-2M",
                "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
                "dirtyjtag": "new operator-confirmed cable path",
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


# REQ-ISING-7327 / SCENARIO-ISING-7327-PREFLIGHT
def test_preconditions_authenticate_exp7314_and_original_transcripts(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)

    checks, hashes, context = experiment.collect_preconditions(ROOT, paths)

    assert all(row["passed"] for row in checks)
    assert context["receipt"]["experiment_id"] == 7314
    assert context["upstream_checksum_matches"] is True
    assert context["reference_observation"]["all_match"] is True
    assert context["scope_observation"] == {
        "kv260_fabric_execution": True,
        "kv260_processor_class": "fpga_fabric",
        "polarfire_cpu_dispatch": True,
        "polarfire_processor_class": "cpu",
        "polarfire_fpga_sampling": False,
    }
    assert hashes[experiment.UPSTREAM_PATH.as_posix()].startswith("sha256:")
    assert hashes["results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"]
    assert hashes["results/raw/experiment_7231/polarfire_dispatch.json"]


# REQ-ISING-7327 / SCENARIO-ISING-7327-BOARDS
def test_missing_receipt_preserves_three_scopes_without_current_availability(
    tmp_path: Path,
) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, context = experiment.collect_preconditions(ROOT, paths)
    physical = experiment.search_physical_state_receipts(
        ROOT, paths.physical_state_search, candidate_paths=[tmp_path / "missing.json"]
    )

    rows = experiment.build_board_rows(ROOT, context["receipt"], physical)
    by_board = {row["board"]: row for row in rows}

    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["processor_class"] == "fpga_fabric"
    assert by_board["KV260"]["fabric_execution_completed"] is True
    assert by_board["KV260"]["exact_next_condition"] == (
        "none; any future access uses ssh kria only"
    )
    assert by_board["PolarFire"]["processor_class"] == "cpu"
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert by_board["GateMate"]["disposition"] == "blocked_changed_physical_state"
    assert all(row["present_availability_asserted"] is False for row in rows)
    assert all(row["hardware_operations_issued"] == [] for row in rows)
    assert experiment.reduce_board_rows(rows)["board_disposition_complete_score"] == 1


# REQ-ISING-7327 / SCENARIO-ISING-7327-GATEMATE
def test_operator_receipt_only_enables_one_future_action(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, context = experiment.collect_preconditions(ROOT, paths)
    physical = experiment.search_physical_state_receipts(
        ROOT,
        paths.physical_state_search,
        candidate_paths=[_valid_operator_receipt(tmp_path / "operator.json")],
    )

    rows = experiment.build_board_rows(ROOT, context["receipt"], physical)
    gate = next(row for row in rows if row["board"] == "GateMate")

    assert gate["disposition"] == "changed_physical_state_future_action_enabled"
    assert gate["exact_next_condition"] == experiment.GATEMATE_FUTURE_ACTION
    assert physical["date_evidence"] == "20260915"
    assert physical["hardware_operations_issued"] == []
    assert gate["hardware_command_count"] == 0

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "complete"),
        _validation(),
        candidate_paths=[_valid_operator_receipt(tmp_path / "complete-operator.json")],
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["board_continuity_complete_score"] == 1


# REQ-ISING-7327 / SCENARIO-ISING-7327-ARTIFACT
def test_missing_gatemate_receipt_is_terminal_blocked_with_zero_score(tmp_path: Path) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_changed_physical_state:")
    assert artifact["board_continuity_complete_score"] == 0
    assert artifact["hardware_operations_issued_count"] == 0
    assert artifact["hardware_operations_issued"] == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"]["loads"].values()) == {0}
    assert set(artifact["invocation_counts"]["generations"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    failure = artifact["gate_check_summary"]["first_failure"]
    assert failure == {
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
    assert artifact["deployment_relevance"]["extropic"]["local_tsu_authority"] is False
    assert artifact["deployment_relevance"]["sparsekan"]["useful_target_predictor"] is False
    assert artifact["next_hardware_conditions"]["gatemate"]["satisfied_by"] == "operator"
    assert experiment.validate_artifact(artifact) == []


# REQ-ISING-7327 / SCENARIO-ISING-7327-VALIDATION
def test_affected_validation_failure_disqualifies_and_cannot_erase_rows(
    tmp_path: Path,
) -> None:
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        _validation(failed="focused_pytest"),
        candidate_paths=[_valid_operator_receipt(tmp_path / "operator.json")],
    )

    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified:")
    assert artifact["board_continuity_complete_score"] == 0
    assert len(artifact["board_rows"]) == 3
    assert artifact["failed_required_commands"] == ["focused_pytest"]
    assert experiment.validate_artifact(artifact) == []


# REQ-ISING-7327 / SCENARIO-ISING-7327-ARTIFACT
def test_validator_atomic_writer_and_entrypoint_are_fail_closed(tmp_path: Path) -> None:
    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.build_artifact(
        ROOT,
        paths,
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )

    receipt = experiment.write_artifact(paths.artifact, artifact)
    assert json.loads(paths.artifact.read_text(encoding="utf-8")) == artifact
    assert receipt["sha256"] == experiment.sha256_file(paths.artifact)

    for mutation, expected in (
        ({"MODEL_SPECS": ["model"]}, "model_declaration"),
        ({"hardware_operations_issued_count": 1}, "hardware_operations"),
        ({"board_continuity_complete_score": 1}, "blocked_score"),
        ({"reproducibility_checksum": "sha256:bad"}, "checksum"),
    ):
        changed = deepcopy(artifact)
        changed.update(mutation)
        if "reproducibility_checksum" not in mutation:
            changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
        assert expected in experiment.validate_artifact(changed)

    with pytest.raises(ValueError, match="artifact_validation_failed"):
        experiment.write_artifact(tmp_path / "invalid.json", {})

    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7327_v643_board_continuity import main" in source
    assert source.count("main()") == 1


# REQ-ISING-7327 / SCENARIO-ISING-7327-PREFLIGHT
def test_task_contract_and_malformed_evidence_fail_closed(tmp_path: Path) -> None:
    roadmap = tmp_path / experiment.ROADMAP_PATH
    roadmap.write_text("[]\n", encoding="utf-8")
    assert experiment.read_task_contract(tmp_path) is None
    roadmap.write_text("tasks: []\n", encoding="utf-8")
    assert experiment.read_task_contract(tmp_path) is None

    observation = experiment.original_reference_observation(
        tmp_path, {"board_rows": [{"board": "KV260"}]}
    )
    assert observation["all_match"] is False
    assert observation["rows"] == []

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path / "artifact"),
        _validation(),
        candidate_paths=[tmp_path / "missing.json"],
    )
    changed = deepcopy(artifact)
    changed["board_rows"][0]["processor_class"] = "cpu"
    changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert "board_scope" in experiment.validate_artifact(changed)

    malformed = deepcopy(artifact)
    malformed["validation_receipts"] = {object()}
    assert "checksum" in experiment.validate_artifact(malformed)


# REQ-ISING-7327 / SCENARIO-ISING-7327-PREFLIGHT
def test_external_precondition_failure_is_terminal_and_has_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = {
        "check": "exp7314_terminal_identity",
        "upstream": experiment.UPSTREAM_PATH.as_posix(),
        "field": "producer identity and terminal class",
        "expected_value": {"verdict_class": "positive"},
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
                "receipt": {},
                "quarantine": {},
                "upstream_checksum_matches": False,
            },
        ),
    )

    artifact = experiment.build_artifact(
        ROOT, experiment.ExperimentPaths.under(tmp_path), _validation()
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_external_precondition:")
    assert artifact["board_rows"] == []
    assert artifact["board_continuity_complete_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"]["observed_value"] == {
        "verdict_class": "disqualified"
    }
    assert experiment.validate_artifact(artifact) == []
