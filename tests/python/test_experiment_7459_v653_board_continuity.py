"""Tests for the V653 read-only board-continuity reducer.

Spec refs: REQ-HW-7459 and SCENARIO-HW-7459-UNCHANGED,
SCENARIO-HW-7459-CHANGED, and SCENARIO-HW-7459-DURABILITY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7459_v653_board_continuity as experiment


ROOT = Path(__file__).resolve().parents[2]


def _source(path: Path) -> dict[str, Any]:
    """Read an immutable source fixture without invoking a producer."""

    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _unchanged_state() -> dict[str, Any]:
    """Describe the observed zero-receipt branch without a board operation."""

    return {
        "exists": False,
        "accepted_receipt_count": 0,
        "latest_receipt_date": "20260911",
        "search_receipt_path": "results/raw/experiment_7459/gatemate_changed_state.json",
        "search_receipt_hash": "sha256:" + "1" * 64,
        "eligibility_contract": {
            "operator_authored": True,
            "receipt_date": ">20260823",
            "changed_field_any_of": [
                "cable",
                "port",
                "power",
                "board",
                "dirtyjtag",
            ],
        },
        "hardware_operations_issued": [],
    }


def test_req_hw_7459_authenticates_original_sources_independently() -> None:
    """REQ-HW-7459: preserve producer identity, flags, and optional absence."""

    hardware = _source(experiment.HARDWARE_SOURCE_PATH)
    cutoff = _source(experiment.EXP6559_PATH)
    durable = _source(experiment.DURABLE_SOURCE_PATH)

    gates = experiment.authenticate_sources(hardware, cutoff, durable)
    assert gates
    assert all(row["passed"] for row in gates)
    assert hardware["honest_verdict"] == (
        "complete_null_hardware_envelope_requires_persistence_orchestration_redesign"
    )
    assert cutoff["honest_verdict"].startswith("blocked_missing_new_physical_receipt:")
    assert durable["honest_verdict"] == "complete_null_durable_delta_speed_gate_not_met"

    missing = experiment.authenticate_sources(hardware, cutoff, None)
    optional = next(row for row in missing if row["check"] == "durable_update_optional")
    assert optional["passed"] is True
    assert optional["observed"] == "absent"

    broken = deepcopy(hardware)
    broken["flagged_adversarial"] = True
    failed = experiment.authenticate_sources(broken, cutoff, durable)
    assert (
        next(row for row in failed if row["check"] == "hardware_flagged_adversarial")["passed"]
        is False
    )


def test_scenario_hw_7459_unchanged_preserves_three_narrow_claims() -> None:
    """SCENARIO-HW-7459-UNCHANGED: reduce three independent claim scopes."""

    source = _source(experiment.HARDWARE_SOURCE_PATH)
    rows = experiment.reduce_board_rows(source, _unchanged_state())
    by_board = {row["board"]: row for row in rows}

    assert list(by_board) == ["KV260", "PolarFire", "GateMate"]
    assert by_board["KV260"]["future_access"] == "ssh_only"
    assert by_board["KV260"]["exact_claim_scope"] == ("historical_kv260_fpga_fabric_sampling_only")
    assert by_board["PolarFire"]["exact_claim_scope"] == (
        "historical_hash_matched_cpu_dispatch_only_no_fpga_sampling"
    )
    assert by_board["PolarFire"]["fpga_sampling_claimed"] is False
    assert by_board["GateMate"]["honest_verdict"] == ("blocked_unchanged_physical_prerequisite")
    assert by_board["GateMate"]["disposition"] == "blocked"
    assert by_board["GateMate"]["hardware_operations_issued"] == []
    assert all(row["evidence_sha256"].startswith("sha256:") for row in rows)
    assert all(row["row_sha256"].startswith("sha256:") for row in rows)

    reduced = experiment.reduce_rows(rows)
    assert reduced == {
        "board_count": 3,
        "board_names_exact": True,
        "graduated_count": 2,
        "blocked_count": 1,
        "hardware_operation_count": 0,
        "kv260_ssh_only": True,
        "polarfire_cpu_not_fpga": True,
        "gatemate_changed_state": False,
        "hardware_ready_score": 0,
        "hardware_value_score": 0,
    }


def test_scenario_hw_7459_changed_records_only_future_eligibility() -> None:
    """SCENARIO-HW-7459-CHANGED: bind evidence but do not touch hardware."""

    changed = {
        **_unchanged_state(),
        "exists": True,
        "accepted_receipt_count": 1,
        "receipt_timestamp": "2026-09-20T15:00:00Z",
        "changed_conditions": {
            "cable": True,
            "port": False,
            "power": True,
            "board": False,
            "dirtyjtag": False,
        },
        "evidence_path": "ops/operator-followup.md",
        "evidence_hash": "sha256:" + "2" * 64,
    }
    rows = experiment.reduce_board_rows(_source(experiment.HARDWARE_SOURCE_PATH), changed)
    gate = next(row for row in rows if row["board"] == "GateMate")

    assert gate["honest_verdict"] == (
        "complete_changed_physical_prerequisite_future_bounded_task_only"
    )
    assert gate["disposition"] == "complete"
    assert gate["changed_state_evidence"]["evidence_path"] == ("ops/operator-followup.md")
    assert gate["bounded_next_bringup_plan"]["maximum_hardware_actions"] == 1
    assert gate["bounded_next_bringup_plan"]["current_task_authorized"] is False
    assert gate["hardware_operations_issued"] == []
    assert experiment.reduce_rows(rows)["gatemate_changed_state"] is True


def test_scenario_hw_7459_durability_drives_only_future_evidence() -> None:
    """SCENARIO-HW-7459-DURABILITY: host timing cannot become board value."""

    durable = _source(experiment.DURABLE_SOURCE_PATH)
    value = experiment.build_hardware_wishlist_disposition(durable)
    options = {row["option"]: row for row in value["routes"]}

    assert value["measurement_available"] is True
    assert value["residual_host_fraction"] == pytest.approx(0.6120927319090697)
    assert value["service_ratio_ci95_upper"] == pytest.approx(1.0046567702574078)
    assert value["hardware_value_score"] == 0
    assert options["NPU"]["currently_justified"] is False
    assert options["larger FPGA"]["currently_justified"] is False
    assert options["authenticated Extropic run"]["local_access_authenticated"] is False
    assert all(row["vendor_report_is_local_timing"] is False for row in options.values())

    absent = experiment.build_hardware_wishlist_disposition(None)
    assert absent["measurement_available"] is False
    assert absent["residual_host_fraction"] is None
    assert absent["independent_board_audit_continues"] is True


def test_req_hw_7459_fixture_cold_reduction_and_mutations(tmp_path: Path) -> None:
    """REQ-HW-7459: raw reduction and identity mutations fail closed."""

    artifact = experiment.build_fixture_artifact(ROOT, tmp_path)
    assert experiment.validate_artifact(artifact, root=ROOT, require_terminal=False) == []
    assert experiment.independent_reduce(artifact)["matches_declared"] is True
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["hardware_ready_score"] == 0
    assert artifact["hardware_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["gate_check_summary"]["first_failure"]["check"] == (
        "gatemate_changed_physical_prerequisite"
    )

    mutations = (
        ("schema", "bad", "identity_mismatch"),
        ("MODEL_SPECS", ["not-current"], "current_model_boundary_invalid"),
        ("hardware_ready_score", 1, "hardware_scores_nonzero"),
        ("board_rows", artifact["board_rows"][:2], "independent_reduction_mismatch"),
        ("field_principles", {}, "field_principles_mismatch"),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum_mismatch"),
    )
    for field, replacement, expected in mutations:
        candidate = deepcopy(artifact)
        candidate[field] = replacement
        assert expected in experiment.validate_artifact(
            candidate, root=ROOT, require_terminal=False
        )


def test_req_hw_7459_scoped_plan_includes_affected_shared_tests(
    tmp_path: Path,
) -> None:
    """REQ-HW-7459: freeze exact new and affected shared validation targets."""

    commands = experiment.build_validation_plan(ROOT, tmp_path / "private")
    assert experiment.validate_validation_plan(ROOT, commands) == []
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert set(experiment.AFFECTED_MANIFEST.test_paths).issubset(set(focused.argv))
    assert "tests/python" not in focused.argv
    assert all("full_python_suite" not in command.name for command in commands)

    malformed = [command for command in commands if command.name != "ruff_check"]
    assert "missing_command:ruff_check" in experiment.validate_validation_plan(ROOT, malformed)

    wrapper = (ROOT / experiment.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "experiment_7459_v653_board_continuity import main" in wrapper
    assert "run_experiment" not in wrapper


def test_req_hw_7459_terminal_receipts_are_required_only_at_publish(
    tmp_path: Path,
) -> None:
    """REQ-HW-7459: a measured candidate and final publication have distinct gates."""

    artifact = experiment.build_fixture_artifact(ROOT, tmp_path)
    assert experiment.validate_artifact(artifact, root=ROOT, require_terminal=False) == []
    assert "terminal_receipts_invalid" in experiment.validate_artifact(
        artifact, root=ROOT, require_terminal=True
    )

    missing_raw = deepcopy(artifact)
    missing_raw["raw_evidence_reference"]["path"] = str(tmp_path / "missing.json")
    assert "raw_evidence_hash_mismatch" in experiment.validate_artifact(
        missing_raw, root=ROOT, require_terminal=False
    )


def test_req_hw_7459_defensive_reducer_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-7459: malformed rows, sources, and validation fail closed."""

    assert experiment.utc_now().endswith("+00:00")
    assert experiment._load_object(tmp_path / "missing.json") == {}
    malformed_json = tmp_path / "malformed.json"
    malformed_json.write_text("{", encoding="utf-8")
    assert experiment._load_object(malformed_json) == {}

    with pytest.raises(ValueError, match="hardware_board_rows_missing"):
        experiment.reduce_board_rows({}, _unchanged_state())
    monkeypatch.setattr(
        experiment.hardware_history,
        "build_board_rows",
        lambda _rows, _changed: [{"board": "KV260"}],
    )
    with pytest.raises(ValueError, match="board_rows_invalid"):
        experiment.reduce_board_rows({"board_rows": [{"board": "KV260"}]}, _unchanged_state())
    monkeypatch.undo()

    artifact = experiment.build_fixture_artifact(ROOT, tmp_path / "fixture")
    ignored_source = deepcopy(artifact)
    ignored_source["source_artifact_hashes"]["ignored"] = "not-a-record"
    ignored_source["reproducibility_checksum"] = experiment.reproducibility_checksum(ignored_source)
    assert experiment.validate_artifact(ignored_source, root=ROOT, require_terminal=False) == []

    bad_source = deepcopy(artifact)
    source = next(iter(bad_source["source_artifact_hashes"].values()))
    source["sha256"] = "sha256:bad"
    assert any(
        error.startswith("source_hash_mismatch:")
        for error in experiment.validate_artifact(bad_source, root=ROOT, require_terminal=False)
    )

    invalid_raw = deepcopy(artifact)
    invalid_raw["raw_evidence_reference"] = None
    assert "raw_evidence_reference_invalid" in experiment.validate_artifact(
        invalid_raw, root=ROOT, require_terminal=False
    )

    bad_board = deepcopy(artifact)
    bad_board["board_rows"][0]["future_access"] = "host_block_device"
    assert "board_claim_boundary_invalid" in experiment.validate_artifact(
        bad_board, root=ROOT, require_terminal=False
    )

    bad_wishlist = deepcopy(artifact)
    bad_wishlist["hardware_wishlist_disposition"] = None
    assert "hardware_wishlist_boundary_invalid" in experiment.validate_artifact(
        bad_wishlist, root=ROOT, require_terminal=False
    )

    required = deepcopy(artifact)
    required["validation_required"] = True
    assert "affected_receipts_invalid" in experiment.validate_artifact(
        required, root=ROOT, require_terminal=False
    )
