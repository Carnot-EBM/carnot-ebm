"""Tests for Exp 1361 p-dit certificate-state mapping.

Spec refs: REQ-HW-048, SCENARIO-HW-048.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.analysis import pdit_certificate_state_mapping as exp1361


def _prior_exp1348_packet() -> dict[str, object]:
    return {
        "metadata": {
            "experiment_id": 1348,
            "schema": "pbit_update_dynamics_dual_bram_packet_v2",
            "run_date": "20260505",
        },
        "status": "complete",
        "reuse_factor_grid": [{"reuse_factor": 1}, {"reuse_factor": 2}, {"reuse_factor": 4}],
        "hardware_claim_allowed": False,
        "kv260_claim_allowed": False,
        "honest_verdict": "cpu_only_update_dynamics_dual_bram_packet_ready_hardware_not_run",
    }


def test_req_hw_048_artifact_schema_and_claim_gates() -> None:
    """REQ-HW-048: artifact has all fields and refuses hardware claims."""
    artifact = exp1361.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        prior_pbit_packet=_prior_exp1348_packet(),
    )

    exp1361.validate_artifact(artifact)
    assert exp1361.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["metadata"]["run_date"] == "20260505"
    assert artifact["metadata"]["hardware_executed"] is False
    assert artifact["metadata"]["local_hardware_synthesis_or_board_run"] is False
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["kv260_claim_allowed"] is False
    assert artifact["honest_verdict"] == exp1361.CPU_ONLY_HONEST_VERDICT


def test_req_hw_048_certificate_alphabet_and_variable_counts() -> None:
    """REQ-HW-048: four certificate states collapse from four p-bits to one p-dit."""
    artifact = exp1361.build_artifact(
        run_date="20260505",
        prior_pbit_packet=_prior_exp1348_packet(),
    )

    assert artifact["certificate_states_mapped"] == ["SAT", "UNSAT", "UNKNOWN", "REPAIR"]
    assert artifact["binary_spin_count"] == 4
    assert artifact["pdit_variable_count"] == 1
    assert artifact["state_expansion_ratio"] == 4.0
    assert artifact["state_space_proxy"]["binary_raw_configurations"] == 16
    assert artifact["state_space_proxy"]["valid_certificate_states"] == 4
    assert artifact["state_space_proxy"]["raw_binary_to_valid_state_ratio"] == 4.0


def test_req_hw_048_binary_pbit_and_pdit_mappings_are_consistent() -> None:
    """REQ-HW-048: one-hot p-bit states and q=4 p-dit/p-int codes agree."""
    alphabet = exp1361.certificate_state_alphabet()
    binary_mapping = exp1361.build_binary_one_hot_mapping(alphabet)
    pdit_mapping = exp1361.build_pdit_mapping(alphabet)

    assert binary_mapping["SAT"] == [1, -1, -1, -1]
    assert binary_mapping["UNSAT"] == [-1, 1, -1, -1]
    assert binary_mapping["UNKNOWN"] == [-1, -1, 1, -1]
    assert binary_mapping["REPAIR"] == [-1, -1, -1, 1]
    assert pdit_mapping["SAT"]["pdit_code"] == 0
    assert pdit_mapping["REPAIR"]["pint_value"] == 3
    assert {row["alphabet_size"] for row in pdit_mapping.values()} == {4}


def test_req_hw_048_energy_equivalence_proxy_is_exact_for_valid_states() -> None:
    """REQ-HW-048: valid one-hot energies equal p-dit table energies."""
    alphabet = exp1361.certificate_state_alphabet()
    binary_mapping = exp1361.build_binary_one_hot_mapping(alphabet)
    pdit_mapping = exp1361.build_pdit_mapping(alphabet)
    energy_table = exp1361.build_certificate_energy_table(alphabet)

    assert (
        exp1361.compute_energy_equivalence_error(binary_mapping, pdit_mapping, energy_table) == 0.0
    )
    assert exp1361.binary_one_hot_energy([1, 1, -1, -1], binary_mapping, energy_table) > 10.0


def test_req_hw_048_pbit_packet_delta_names_exp1348_and_cpu_only_scope() -> None:
    """REQ-HW-048: pbit_packet_delta records the delta from Exp 1348."""
    artifact = exp1361.build_artifact(
        run_date="20260505",
        prior_pbit_packet=_prior_exp1348_packet(),
    )
    delta = artifact["pbit_packet_delta"]

    assert delta["prior_experiment_id"] == 1348
    assert delta["prior_schema"] == "pbit_update_dynamics_dual_bram_packet_v2"
    assert delta["prior_hardware_claim_allowed"] is False
    assert delta["prior_kv260_claim_allowed"] is False
    assert delta["new_mapping"] == "q4_pdit_or_pint_certificate_state"
    assert "4 binary p-bit spins" in delta["variable_delta"]
    assert delta["hardware_scope_delta"] == "still_cpu_only_no_vivado_fpga_kv260_tsu_analog_run"


def test_scenario_hw_048_write_packet_json(tmp_path: Path) -> None:
    """SCENARIO-HW-048: writer persists a complete honest mapping artifact."""
    out_path = tmp_path / "experiment_1361_pdit_certificate_state_hardware_mapping.json"
    artifact = exp1361.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        prior_pbit_packet=_prior_exp1348_packet(),
    )

    written = exp1361.write_artifact(out_path, artifact)
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert payload == artifact
    assert payload["status"] == "complete"
    assert payload["hardware_claim_allowed"] is False
    assert payload["kv260_claim_allowed"] is False


def test_req_hw_048_validation_rejects_missing_states_or_dishonest_claims() -> None:
    """REQ-HW-048: validator rejects missing fields, states, and unsupported claims."""
    artifact = exp1361.build_artifact(
        run_date="20260505",
        prior_pbit_packet=_prior_exp1348_packet(),
    )

    missing = dict(artifact)
    missing.pop("pdit_variable_count")
    with pytest.raises(ValueError, match="missing"):
        exp1361.validate_artifact(missing)

    missing_state = dict(artifact)
    missing_state["certificate_states_mapped"] = ["SAT", "UNSAT", "UNKNOWN"]
    with pytest.raises(ValueError, match="certificate_states_mapped"):
        exp1361.validate_artifact(missing_state)

    bad_ratio = dict(artifact)
    bad_ratio["state_expansion_ratio"] = 1.0
    with pytest.raises(ValueError, match="state_expansion_ratio"):
        exp1361.validate_artifact(bad_ratio)

    bad_energy = dict(artifact)
    bad_energy["energy_equivalence_error"] = 0.25
    with pytest.raises(ValueError, match="energy_equivalence_error"):
        exp1361.validate_artifact(bad_energy)

    dishonest_hardware = dict(artifact)
    dishonest_hardware["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        exp1361.validate_artifact(dishonest_hardware)

    dishonest_kv260 = dict(artifact)
    dishonest_kv260["kv260_claim_allowed"] = True
    with pytest.raises(ValueError, match="kv260_claim_allowed"):
        exp1361.validate_artifact(dishonest_kv260)
