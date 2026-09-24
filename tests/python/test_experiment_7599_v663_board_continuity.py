"""Tests for REQ-HW-7599 and SCENARIO-HW-7599-*.

These tests keep historical board scope separate from optional host placement.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7599_v663_board_continuity as exp


ROOT = Path(__file__).resolve().parents[2]


def _eligible_consumer() -> dict[str, object]:
    """Return one synthetic whole-client decomposition with clear arithmetic."""

    return {
        "schema": "synthetic.exp7598.eligible.v1",
        "honest_verdict": "complete_null_fixture",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "consumer_ready_score": 1,
        "public_client_stage_timings": {
            "measured": True,
            "scope": "whole_public_client",
            "unit": "ns",
            "whole_client": 100.0,
            "arithmetic": 20.0,
            "ipc": 30.0,
            "persistence": 40.0,
            "other": 10.0,
        },
    }


def test_req_hw_7599_preserves_three_dated_board_scopes(tmp_path: Path) -> None:
    """REQ-HW-7599; SCENARIO-HW-7599-BOARD-SCOPES."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    rows = {row["board"]: row for row in artifact["board_rows"]}

    assert set(rows) == {"KV260", "PolarFire", "GateMate"}
    assert artifact["board_continuity_complete_score"] == 1
    assert artifact["hardware_operations_issued"] == []
    assert rows["KV260"]["processor_class"] == "fpga_fabric"
    assert rows["KV260"]["future_access"] == "ssh kria"
    assert rows["KV260"]["k_max"] == 5
    assert rows["KV260"]["current_reachability"] == "unknown_not_probed"
    assert rows["PolarFire"]["processor_class"] == "linux_cpu"
    assert rows["PolarFire"]["fpga_sampling_measured"] is False
    assert rows["GateMate"]["disposition"] == "blocked_unchanged_physical_prerequisite"
    assert rows["GateMate"]["last_diagnostic"]["observed"] == "0xffffffff"
    assert all(row["source_sha256"].startswith("sha256:") for row in rows.values())
    assert exp.validate_artifact(artifact, root=ROOT) == []


def test_scenario_hw_7599_real_exp7598_is_placement_unmeasured() -> None:
    """SCENARIO-HW-7599-PLACEMENT-UNMEASURED keeps missing stages explicit."""

    source = exp.load_object(ROOT / exp.CONSUMER_PATH)
    placement = exp.reduce_placement(source)

    assert placement["placement_scope"] == "placement_unmeasured"
    assert placement["amdahl_upper_bound"] is None
    assert placement["fractions"] == {}
    assert placement["failed_check"]["field"] == "public_client_stage_timings"
    assert "IPC" in placement["failed_check"]["observed"]
    assert "persistence" in placement["failed_check"]["observed"]


@pytest.mark.parametrize(
    ("consumer", "field"),
    [
        ({}, "path"),
        ({"flagged_adversarial": True}, "flagged_adversarial"),
        ({"flagged_adversarial": False, "consumer_ready_score": 0}, "consumer_ready_score"),
    ],
)
def test_scenario_hw_7599_ineligible_consumer_fails_closed(
    consumer: dict[str, object], field: str
) -> None:
    """SCENARIO-HW-7599-PLACEMENT-UNMEASURED names early eligibility failures."""

    placement = exp.reduce_placement(consumer)
    assert placement["placement_scope"] == "placement_unmeasured"
    assert placement["failed_check"]["field"] == field


def test_scenario_hw_7599_eligible_stage_formula_is_transparent() -> None:
    """SCENARIO-HW-7599-PLACEMENT computes fractions and an Amdahl bound."""

    placement = exp.reduce_placement(_eligible_consumer())

    assert placement["placement_scope"] == "whole_public_client_measured_decomposition"
    assert placement["fractions"] == {
        "arithmetic": 0.2,
        "ipc": 0.3,
        "persistence": 0.4,
        "other": 0.1,
    }
    assert math.isclose(placement["amdahl_upper_bound"], 1.25)
    assert placement["formula"] == "whole_client / (whole_client - arithmetic)"
    assert placement["projected_latency_is_measurement"] is False


def test_scenario_hw_7599_eligible_consumer_keeps_null_hardware_claim(tmp_path: Path) -> None:
    """SCENARIO-HW-7599-PLACEMENT records host placement without hardware benefit."""

    artifact = exp.build_artifact(
        ROOT,
        tmp_path / "raw",
        consumer_override=_eligible_consumer(),
        receipt_candidates=[],
    )
    assert artifact["placement_scope"] == "whole_public_client_measured_decomposition"
    assert artifact["honest_verdict"] == "complete_null_board_continuity_host_placement_only"
    assert artifact["verdict_class"] == "null"
    assert artifact["host_speed_reported_as_board_or_tsu_speed"] is False


@pytest.mark.parametrize(
    "timings",
    [
        {"measured": True, "scope": "whole_public_client", "unit": "ns"},
        {
            "measured": True,
            "scope": "whole_public_client",
            "unit": "ns",
            "whole_client": 10.0,
            "arithmetic": 10.0,
            "ipc": 0.0,
            "persistence": 0.0,
            "other": 0.0,
        },
        {
            "measured": True,
            "scope": "kernel_only",
            "unit": "ns",
            "whole_client": 100.0,
            "arithmetic": 20.0,
            "ipc": 30.0,
            "persistence": 40.0,
            "other": 10.0,
        },
    ],
)
def test_req_hw_7599_rejects_ineligible_stage_decompositions(
    timings: dict[str, object],
) -> None:
    """REQ-HW-7599 rejects incomplete, unbounded, or mismatched timings."""

    consumer = _eligible_consumer()
    consumer["public_client_stage_timings"] = timings
    assert exp.reduce_placement(consumer)["placement_scope"] == "placement_unmeasured"


def test_req_hw_7599_acquisition_note_authorizes_no_purchase(tmp_path: Path) -> None:
    """REQ-HW-7599 keeps acquisition contingent on a measured workload."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    decision = artifact["acquisition_decision"]

    assert decision["purchase_authorized"] is False
    assert decision["new_accelerator_justified_by_small_host_update"] is False
    assert decision["extropic_thrml"]["status"] == "compatibility_or_future_access"
    assert decision["xdna"]["required_evidence"]
    assert decision["larger_fpga"]["required_evidence"]
    assert decision["thermodynamic_sampler_required"] is False
    assert artifact["host_speed_reported_as_board_or_tsu_speed"] is False


def test_req_hw_7599_gate_summary_names_exact_gatemate_failure(tmp_path: Path) -> None:
    """REQ-HW-7599 records every operand of the external GateMate blocker."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    gate = next(row for row in artifact["gate_check_summary"] if row["check"] == "gatemate_receipt")

    assert gate["upstream"] == "operator physical-change receipt"
    assert gate["path"].endswith("gatemate_receipt_search.json")
    assert gate["field"] == "accepted_receipt_count"
    assert gate["op"] == "gt"
    assert gate["expected"] == 0
    assert gate["observed"] == 0
    assert gate["passed"] is False


def test_scenario_hw_7599_new_receipt_only_enables_separate_task(tmp_path: Path) -> None:
    """SCENARIO-HW-7599-BOARD-SCOPES records a receipt without probing."""

    receipt = {
        "exists": True,
        "receipt_date": "20260924",
        "source": "operator directive: GateMate power cable changed",
        "operator_authored": True,
        "board": "Cologne Chip GateMate A1-EVB-2M",
        "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
        "power": "operator changed the GateMate power cable",
        "changes": [{"field": "power", "description": "power cable changed"}],
    }
    artifact = exp.build_test_artifact(ROOT, tmp_path, receipt_candidates=[receipt])
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")

    assert gate["disposition"] == "changed_physical_state_separate_task_eligible"
    assert gate["operator_receipt"]["receipt_date"] == "20260924"
    assert gate["current_hardware_execution_authorized"] is False
    assert artifact["hardware_operations_issued"] == []
    assert not any(row["check"] == "gatemate_receipt" for row in artifact["gate_check_summary"])


def test_req_hw_7599_independent_reduction_and_checksum(tmp_path: Path) -> None:
    """REQ-HW-7599 independently reproduces counts and placement status."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    reduced = exp.independent_reduce(artifact)

    assert reduced == {
        "board_count": 3,
        "authenticated_board_count": 3,
        "blocked_board_count": 1,
        "hardware_operation_count": 0,
        "placement_scope": "placement_unmeasured",
        "fractions": {},
        "amdahl_upper_bound": None,
    }
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)


def test_req_hw_7599_defensive_readers_and_incomplete_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-HW-7599 treats malformed bytes and incomplete board sets as absent."""

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert exp.load_object(invalid) == {}
    assert exp.load_object(tmp_path / "missing.json") == {}
    assert exp._prior_rows({"board_rows": "wrong"}) == {}
    assert exp.build_board_rows(ROOT, {"board_rows": []}, {}) == []

    monkeypatch.setattr(exp, "CONSUMER_PATH", Path("results/missing-exp7598.json"))
    artifact = exp.build_test_artifact(ROOT, tmp_path / "missing-consumer")
    source = artifact["source_artifact_hashes"]["results/missing-exp7598.json"]
    assert source["missing_producer"] is True
    assert source["sha256"] is None
    assert artifact["placement_scope"] == "placement_unmeasured"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"MODEL_SPECS": [{"name": "model"}]}, "model_declaration"),
        ({"hardware_operations_issued": ["ssh kria true"]}, "hardware_operations"),
        ({"board_continuity_complete_score": 0}, "board_continuity_score"),
        ({"host_speed_reported_as_board_or_tsu_speed": True}, "host_speed_claim"),
        ({"rows": []}, "rows"),
        ({"amdahl_upper_bound": 99.0}, "placement_reduction"),
        ({"reproducibility_checksum": "sha256:bad"}, "reproducibility_checksum"),
    ],
)
def test_req_hw_7599_validator_rejects_claim_broadening(
    tmp_path: Path, mutation: dict[str, object], error: str
) -> None:
    """REQ-HW-7599 rejects model, hardware, placement, and custody drift."""

    artifact = exp.build_test_artifact(ROOT, tmp_path)
    changed = deepcopy(artifact)
    changed.update(mutation)
    if "reproducibility_checksum" not in mutation:
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert error in exp.validate_artifact(changed, root=ROOT)


def test_req_hw_7599_validator_rejects_each_board_scope_mutation(tmp_path: Path) -> None:
    """REQ-HW-7599 keeps each venue claim narrow under nested mutations."""

    mutations = [
        ("KV260", "processor_class", "host_cpu", "kv260_scope"),
        ("KV260", "future_access", "host block device", "kv260_access"),
        ("KV260", "k_max", 6, "kv260_k_max"),
        ("PolarFire", "processor_class", "fpga_fabric", "polarfire_scope"),
        ("PolarFire", "fpga_sampling_measured", True, "polarfire_fpga_claim"),
        ("GateMate", "current_hardware_execution_authorized", True, "gatemate_authorization"),
    ]
    for board, field, value, error in mutations:
        changed = deepcopy(exp.build_test_artifact(ROOT, tmp_path / error))
        row = next(item for item in changed["board_rows"] if item["board"] == board)
        row[field] = value
        changed["rows"] = deepcopy(changed["board_rows"])
        changed["independent_reduction"] = exp.independent_reduce(changed)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert error in exp.validate_artifact(changed)

    changed = deepcopy(exp.build_test_artifact(ROOT, tmp_path / "identity"))
    changed["board_rows"][0]["board"] = "Other"
    changed["rows"] = deepcopy(changed["board_rows"])
    changed["independent_reduction"] = exp.independent_reduce(changed)
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "board_identity" in exp.validate_artifact(changed)


def test_req_hw_7599_source_reader_rejects_receipt_and_hash_drift(tmp_path: Path) -> None:
    """REQ-HW-7599 authenticates source receipt structure and exact bytes."""

    artifact = exp.build_test_artifact(ROOT, tmp_path / "artifact")
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["bad-receipt"] = "not-a-map"
    changed["source_artifact_hashes"]["missing-allowed"] = {
        "path": "missing-allowed",
        "sha256": None,
        "missing_producer": True,
    }
    changed["source_artifact_hashes"]["wrong-hash"] = {
        "path": exp.CONSUMER_PATH.as_posix(),
        "sha256": "sha256:bad",
    }
    errors = exp._verify_sources(changed, ROOT)
    assert "source_receipt:bad-receipt" in errors
    assert "source_hash:wrong-hash" in errors
    assert not any("missing-allowed" in error for error in errors)
    assert exp._verify_sources({}, ROOT) == ["source_artifact_hashes"]


def test_scenario_hw_7599_missing_source_is_complete_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-7599-BOARD-SCOPES fails closed on source custody loss."""

    missing = tmp_path / "missing-kv260.json"
    monkeypatch.setitem(exp.BOARD_EVIDENCE_PATHS, "KV260", missing)
    artifact = exp.build_test_artifact(ROOT, tmp_path / "scratch")

    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["board_rows"] == []
    gate = artifact["gate_check_summary"][0]
    assert gate["upstream"] == "KV260 dated disposition"
    assert gate["path"] == str(missing)
    assert gate["field"] == "path"
    assert gate["expected"] == "readable nonempty file"
    assert gate["observed"] == "missing"


def test_req_hw_7599_scoped_commands_and_thin_cli(tmp_path: Path) -> None:
    """REQ-HW-7599 freezes private validation and keeps the CLI thin."""

    commands = exp.build_validation_commands(ROOT, tmp_path / "validation")
    rendered = [" ".join(command.argv) for command in commands]

    assert len(commands) == 7
    assert all("tests/python " not in command for command in rendered)
    assert any("--no-cov" in command and "-n 0" in command for command in rendered)
    assert any("--fail-under=100" in command for command in rendered)
    assert any("check_spec_coverage.py" in command for command in rendered)
    assert all(path.startswith("/tmp/") for path in exp.private_basetemps(commands))

    terminal = exp.terminal_commands(tmp_path / "candidate.json", ROOT)
    assert [command.name for command in terminal] == list(exp.TERMINAL_NAMES)

    source = (ROOT / exp.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "experiment_7599_v663_board_continuity import main" in source
    assert source.count("main()") == 1


def test_req_hw_7599_atomic_publish_and_read_only_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-HW-7599 publishes stable bytes and replays in a fresh mode."""

    artifact = exp.build_test_artifact(ROOT, tmp_path / "scratch")
    path = tmp_path / "artifact.json"
    exp.atomic_publish(path, artifact, root=ROOT)

    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert exp.main(["--root", str(ROOT), "--cold-replay", str(path)]) == 0
    assert exp.main(["--root", str(ROOT), "--independent-reduce", str(path)]) == 0
    assert "validation_passed" in capsys.readouterr().out
    assert exp.cold_replay(tmp_path / "missing.json", root=ROOT) == ["artifact_unreadable"]
    assert exp.independent_replay(tmp_path / "missing.json") == ["artifact_unreadable"]
    with pytest.raises(ValueError, match="invalid Exp7599 artifact"):
        exp.atomic_publish(tmp_path / "bad.json", {}, root=ROOT)


def test_req_hw_7599_main_dispatches_declared_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-HW-7599 sends the fixed date and root to the producer."""

    calls: list[tuple[Path, str, Path | None]] = []
    monkeypatch.setattr(exp, "progress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root, date, output_path=None: calls.append((root, date, output_path)),
    )
    assert exp.main(["--root", str(ROOT), "--date", exp.RUN_DATE]) == 0
    assert calls == [(ROOT.resolve(), exp.RUN_DATE, None)]
