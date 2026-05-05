"""Tests for Exp 1348 p-bit update dynamics and dual-BRAM packet.

Spec refs: REQ-HW-047, SCENARIO-HW-047.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.analysis import pbit_update_dynamics_dual_bram as exp1348


def test_req_hw_047_artifact_schema_and_claim_gates() -> None:
    """REQ-HW-047: packet has the required fields and disallows hardware claims."""
    artifact = exp1348.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        synthesis_performed=False,
        board_executed=False,
    )

    exp1348.validate_artifact(artifact)
    assert exp1348.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["metadata"]["run_date"] == "20260505"
    assert artifact["metadata"]["synthesis_performed"] is False
    assert artifact["metadata"]["board_executed"] is False
    assert artifact["kv260_claim_allowed"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"] == exp1348.CPU_ONLY_HONEST_VERDICT


def test_req_hw_047_sync_async_regime_covers_update_dynamics_references() -> None:
    """REQ-HW-047: update-dynamics grid covers sync, async, and delayed regimes."""
    regimes = exp1348.build_sync_async_regime()
    names = {row["name"] for row in regimes}

    assert {
        "synchronous_snapshot_parallel",
        "asynchronous_single_site_gibbs_like",
        "phase_serialized_delayed_snapshot",
    } <= names
    assert all(row["hardware_verified"] is False for row in regimes)
    assert all("reference_basis" in row for row in regimes)
    assert any("Scientific Reports 2026" in item for row in regimes for item in row["reference_basis"])
    assert any("arXiv 2602.16143" in item for row in regimes for item in row["reference_basis"])


def test_req_hw_047_reuse_factor_grid_inherits_tiny_ising_cpu_evidence() -> None:
    """REQ-HW-047: reuse grid records physical p-bit reuse and CPU KL evidence."""
    artifact = exp1348.build_artifact(run_date="20260505")
    grid = artifact["reuse_factor_grid"]

    assert [row["reuse_factor"] for row in grid] == [1, 2, 4]
    assert [row["logical_spins"] for row in grid] == [4, 4, 4]
    assert [row["physical_pbits"] for row in grid] == [4, 2, 1]
    assert [row["parallel_update_width"] for row in grid] == [4, 2, 1]
    assert all(row["cpu_kl_to_gibbs"] >= 0.0 for row in grid)
    assert grid[-1]["regime_name"] == "asynchronous_single_site_gibbs_like"
    assert grid[-1]["cpu_kl_to_gibbs"] < grid[0]["cpu_kl_to_gibbs"]


def test_req_hw_047_bram_dac_and_finite_delay_assumptions_are_explicit() -> None:
    """REQ-HW-047: BRAM layout, DAC precision, and finite delay are RTL-reviewable."""
    artifact = exp1348.build_artifact(run_date="20260505")
    layout = artifact["bram_layout"]
    dac = artifact["dac_precision_assumption"]
    delay = artifact["finite_delay_assumption"]

    assert layout["bank_count"] == 2
    assert layout["bank_a"]["role"] == "snapshot_read"
    assert layout["bank_b"]["role"] == "delayed_write_next_snapshot"
    assert "ising_coupling_rows" in layout["bank_a"]["contents"]
    assert "kan_spline_lut_segments" in layout["bank_a"]["contents"]
    assert "quantized_local_field_cache" in layout["bank_b"]["contents"]
    assert layout["bank_swap_rule"] == "swap BRAM_A and BRAM_B only at phase boundary"

    assert dac["selected_bits"] == 6
    assert dac["bit_widths_to_sweep"] == [4, 6, 8]
    assert dac["quantization_rule"] == "signed clipped uniform local-field ladder"
    assert dac["analog_dac_validated"] is False

    assert delay["delay_cycles_grid"] == [0, 1, 2, 4]
    assert delay["selected_delay_cycles"] == 1
    assert delay["local_delay_measurement_available"] is False
    assert delay["acceptance_gate"] == "KL and energy-rank drift must be remeasured after RTL timing"


def test_scenario_hw_047_next_rtl_requirements_are_concrete_interfaces() -> None:
    """SCENARIO-HW-047: next RTL requirements name files and interfaces."""
    requirements = exp1348.build_next_rtl_requirements()
    paths = {row["path"] for row in requirements}
    interfaces = " ".join(row["interface"] for row in requirements)

    assert "hardware/kv260/ising_sampler_v7_pbit_dual_bram.v" in paths
    assert "hardware/kv260/pbit_dual_bram_pkg.sv" in paths
    assert "hardware/kv260/synth_pbit_dual_bram.tcl" in paths
    assert "python/carnot/hardware/pbit_dual_bram_driver.py" in paths
    assert "tests/python/test_pbit_dual_bram_rtl_contract.py" in paths
    assert "AXI-Lite" in interfaces
    assert "BRAM_A/BRAM_B" in interfaces
    assert all(row["claim_gate"] == "required_before_hardware_claim" for row in requirements)


def test_scenario_hw_047_write_packet_json(tmp_path: Path) -> None:
    """SCENARIO-HW-047: writer persists the complete honest packet."""
    out_path = tmp_path / "experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json"
    artifact = exp1348.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
    )

    written = exp1348.write_artifact(out_path, artifact)
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert payload == artifact
    assert payload["status"] == "complete"
    assert payload["kv260_claim_allowed"] is False
    assert payload["hardware_claim_allowed"] is False


def test_req_hw_047_validation_rejects_incomplete_or_dishonest_packets() -> None:
    """REQ-HW-047: validator rejects missing fields and unsupported hardware claims."""
    artifact = exp1348.build_artifact(run_date="20260505")

    missing = dict(artifact)
    missing.pop("sync_async_regime")
    with pytest.raises(ValueError, match="missing"):
        exp1348.validate_artifact(missing)

    in_progress = dict(artifact)
    in_progress["status"] = "in_progress"
    with pytest.raises(ValueError, match="status"):
        exp1348.validate_artifact(in_progress)

    too_few_regimes = dict(artifact)
    too_few_regimes["sync_async_regime"] = too_few_regimes["sync_async_regime"][:2]
    with pytest.raises(ValueError, match="sync_async_regime"):
        exp1348.validate_artifact(too_few_regimes)

    dishonest_kv260 = dict(artifact)
    dishonest_kv260["kv260_claim_allowed"] = True
    with pytest.raises(ValueError, match="kv260_claim_allowed"):
        exp1348.validate_artifact(dishonest_kv260)

    dishonest_hardware = dict(artifact)
    dishonest_hardware["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        exp1348.validate_artifact(dishonest_hardware)

    too_few_reuse_rows = dict(artifact)
    too_few_reuse_rows["reuse_factor_grid"] = too_few_reuse_rows["reuse_factor_grid"][:2]
    with pytest.raises(ValueError, match="reuse_factor_grid"):
        exp1348.validate_artifact(too_few_reuse_rows)

    wrong_bank_count = dict(artifact)
    wrong_bank_count["bram_layout"] = {**wrong_bank_count["bram_layout"], "bank_count": 1}
    with pytest.raises(ValueError, match="two banks"):
        exp1348.validate_artifact(wrong_bank_count)

    wrong_bank_a = dict(artifact)
    wrong_bank_a["bram_layout"] = {
        **wrong_bank_a["bram_layout"],
        "bank_a": {**wrong_bank_a["bram_layout"]["bank_a"], "role": "write_first"},
    }
    with pytest.raises(ValueError, match="bank_a"):
        exp1348.validate_artifact(wrong_bank_a)

    wrong_bank_b = dict(artifact)
    wrong_bank_b["bram_layout"] = {
        **wrong_bank_b["bram_layout"],
        "bank_b": {**wrong_bank_b["bram_layout"]["bank_b"], "role": "read_only"},
    }
    with pytest.raises(ValueError, match="bank_b"):
        exp1348.validate_artifact(wrong_bank_b)

    missing_kan_layout = dict(artifact)
    missing_kan_layout["bram_layout"] = {
        **missing_kan_layout["bram_layout"],
        "bank_a": {**missing_kan_layout["bram_layout"]["bank_a"], "contents": []},
    }
    with pytest.raises(ValueError, match="kan_spline_lut_segments"):
        exp1348.validate_artifact(missing_kan_layout)

    unknown_verdict = dict(artifact)
    unknown_verdict["honest_verdict"] = "kv260_hardware_working"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1348.validate_artifact(unknown_verdict)


def test_req_hw_047_validation_rejects_bad_status_regime_layout_and_verdict() -> None:
    """REQ-HW-047: validator rejects malformed packet status, regimes, banks, and verdict."""
    artifact = exp1348.build_artifact(run_date="20260505")

    bad_status = dict(artifact)
    bad_status["status"] = "in_progress"
    with pytest.raises(ValueError, match="status"):
        exp1348.validate_artifact(bad_status)

    too_few_regimes = dict(artifact)
    too_few_regimes["sync_async_regime"] = too_few_regimes["sync_async_regime"][:2]
    with pytest.raises(ValueError, match="sync_async_regime"):
        exp1348.validate_artifact(too_few_regimes)

    bad_bank_count = dict(artifact)
    bad_bank_count["bram_layout"] = {**bad_bank_count["bram_layout"], "bank_count": 1}
    with pytest.raises(ValueError, match="exactly two banks"):
        exp1348.validate_artifact(bad_bank_count)

    bad_bank_a = dict(artifact)
    bad_bank_a["bram_layout"] = {
        **bad_bank_a["bram_layout"],
        "bank_a": {**bad_bank_a["bram_layout"]["bank_a"], "role": "write_first"},
    }
    with pytest.raises(ValueError, match="snapshot_read"):
        exp1348.validate_artifact(bad_bank_a)

    bad_bank_b = dict(artifact)
    bad_bank_b["bram_layout"] = {
        **bad_bank_b["bram_layout"],
        "bank_b": {**bad_bank_b["bram_layout"]["bank_b"], "role": "read_first"},
    }
    with pytest.raises(ValueError, match="delayed_write_next_snapshot"):
        exp1348.validate_artifact(bad_bank_b)

    unknown_verdict = dict(artifact)
    unknown_verdict["honest_verdict"] = "claimed_kv260_without_a_local_run"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1348.validate_artifact(unknown_verdict)
