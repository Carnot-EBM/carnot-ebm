"""Tests for Exp 1320 p-bit sampler portability packet.

Spec refs: REQ-HW-046, SCENARIO-HW-046.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.analysis import pbit_sampler_portability as exp1320


def test_req_hw_046_cpu_gibbs_matches_exact_boltzmann() -> None:
    """REQ-HW-046: CPU sequential Gibbs baseline matches the tiny Ising target."""
    case = exp1320.tiny_ising_case()
    states = exp1320.enumerate_spin_states(case.n_spins)
    boltzmann = exp1320.exact_boltzmann_distribution(case, states)
    gibbs = exp1320.cpu_gibbs_distribution(case, states, sweeps=180)

    assert case.n_spins == 4
    assert np.isclose(float(gibbs.sum()), 1.0)
    assert exp1320.kl_divergence(gibbs, boltzmann) < 1e-8
    assert exp1320.kl_divergence(gibbs, gibbs) == pytest.approx(0.0)


def test_req_hw_046_reuse_and_dac_sweeps_report_cpu_kl() -> None:
    """REQ-HW-046: reuse-factor and DAC-bit sweeps both report KL to CPU Gibbs."""
    case = exp1320.tiny_ising_case()
    states = exp1320.enumerate_spin_states(case.n_spins)
    baseline = exp1320.cpu_gibbs_distribution(case, states, sweeps=180)

    reuse_sweep = exp1320.reuse_factor_sweep(
        case,
        states,
        baseline,
        reuse_factors=(1, 2, 4),
        dac_bits=6,
        sweeps=140,
    )
    dac_sweep = exp1320.dac_bits_sweep(
        case,
        states,
        baseline,
        dac_bits_values=(2, 3, 4, 6),
        reuse_factor=4,
        sweeps=140,
    )

    assert [row["reuse_factor"] for row in reuse_sweep] == [1, 2, 4]
    assert [row["physical_pbits"] for row in reuse_sweep] == [4, 2, 1]
    assert all(row["kl_to_cpu_gibbs"] >= 0.0 for row in reuse_sweep)
    assert reuse_sweep[-1]["update_policy"] == "single_site_gibbs_like"
    assert reuse_sweep[-1]["kl_to_cpu_gibbs"] < reuse_sweep[0]["kl_to_cpu_gibbs"]

    assert [row["dac_bits"] for row in dac_sweep] == [2, 3, 4, 6]
    assert [row["dac_levels"] for row in dac_sweep] == [4, 8, 16, 64]
    assert all(row["kl_to_cpu_gibbs"] >= 0.0 for row in dac_sweep)
    assert dac_sweep[-1]["kl_to_cpu_gibbs"] < 0.01


def test_req_hw_046_dual_bram_mapping_ready_gate() -> None:
    """REQ-HW-046: dual-BRAM readiness requires read snapshot and write paths."""
    case = exp1320.tiny_ising_case()
    sketch = exp1320.dual_bram_mapping_sketch(case, reuse_factors=(1, 2, 4))

    assert exp1320.is_dual_bram_mapping_ready(sketch) is True
    assert sketch["bank_a"]["role"] == "read_snapshot"
    assert sketch["bank_b"]["role"] == "delayed_write_update"
    assert sketch["spin_serial_schedule"]["reuse_factors"] == [1, 2, 4]

    missing_read_path = dict(sketch)
    missing_read_path["read_snapshot_path"] = False
    assert exp1320.is_dual_bram_mapping_ready(missing_read_path) is False

    missing_bank = dict(sketch)
    missing_bank["bank_b"] = {}
    assert exp1320.is_dual_bram_mapping_ready(missing_bank) is False


def test_req_hw_046_hardware_claim_gate_requires_actual_execution() -> None:
    """REQ-HW-046: tools on PATH alone do not authorize a hardware claim."""
    no_tools = exp1320.detect_fpga_environment(which=lambda _name: None, env={})
    with_tool = exp1320.detect_fpga_environment(
        which=lambda name: f"/tools/{name}" if name == "vivado" else None,
        env={"CARNOT_KV260_BITFILE": "/tmp/fake.bit"},
    )

    assert no_tools["vivado_available"] is False
    assert with_tool["vivado_available"] is True
    assert with_tool["kv260_bitfile_configured"] is True
    assert exp1320.hardware_claim_allowed(with_tool, synthesis_performed=False) is False
    assert exp1320.hardware_claim_allowed(with_tool, synthesis_performed=True) is True
    assert exp1320.vivado_required_for_next_step(synthesis_performed=False) is True
    assert exp1320.vivado_required_for_next_step(synthesis_performed=True) is False


def test_scenario_hw_046_artifact_schema_and_honest_verdict() -> None:
    """SCENARIO-HW-046: complete artifact has all required packet fields."""
    detection = exp1320.detect_fpga_environment(which=lambda _name: None, env={})

    artifact = exp1320.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        fpga_environment=detection,
        synthesis_performed=False,
    )

    exp1320.validate_artifact(artifact)
    assert exp1320.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["metadata"]["run_date"] == "20260505"
    assert artifact["dual_bram_mapping_ready"] is True
    assert len(artifact["reuse_factor_sweep"]) >= 3
    assert len(artifact["dac_bits_sweep"]) >= 3
    assert artifact["kl_to_cpu_gibbs"] == pytest.approx(
        artifact["selected_cpu_equivalence"]["kl_to_cpu_gibbs"]
    )
    assert artifact["vivado_required_for_next_step"] is True
    assert artifact["hardware_claim_allowed"] is False
    assert artifact["honest_verdict"] == "cpu_portability_packet_ready_hardware_not_run"


def test_scenario_hw_046_write_packet_json(tmp_path: Path) -> None:
    """SCENARIO-HW-046: writer persists the validated JSON portability packet."""
    out_path = tmp_path / "experiment_1320_pbit_sampler_portability_packet.json"
    artifact = exp1320.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        fpga_environment=exp1320.detect_fpga_environment(which=lambda _name: None, env={}),
    )

    written = exp1320.write_artifact(out_path, artifact)
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert payload == artifact
    assert payload["dual_bram_mapping_ready"] is True
    assert payload["hardware_claim_allowed"] is False


def test_req_hw_046_validation_rejects_incomplete_or_dishonest_packets() -> None:
    """REQ-HW-046: validator rejects missing fields and hardware-claim drift."""
    artifact = exp1320.build_artifact(
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
        run_date="20260505",
        fpga_environment=exp1320.detect_fpga_environment(which=lambda _name: None, env={}),
    )

    missing = dict(artifact)
    missing.pop("dac_bits_sweep")
    with pytest.raises(ValueError, match="missing"):
        exp1320.validate_artifact(missing)

    too_short = dict(artifact)
    too_short["reuse_factor_sweep"] = too_short["reuse_factor_sweep"][:2]
    with pytest.raises(ValueError, match="reuse"):
        exp1320.validate_artifact(too_short)

    dishonest = dict(artifact)
    dishonest["hardware_claim_allowed"] = True
    with pytest.raises(ValueError, match="hardware_claim_allowed"):
        exp1320.validate_artifact(dishonest)

    too_few_dac_bits = dict(artifact)
    too_few_dac_bits["dac_bits_sweep"] = too_few_dac_bits["dac_bits_sweep"][:2]
    with pytest.raises(ValueError, match="dac_bits"):
        exp1320.validate_artifact(too_few_dac_bits)

    mapping_not_ready = dict(artifact)
    mapping_not_ready["dual_bram_mapping_ready"] = False
    with pytest.raises(ValueError, match="dual_bram"):
        exp1320.validate_artifact(mapping_not_ready)

    impossible_vivado_gate = dict(artifact)
    impossible_vivado_gate["vivado_required_for_next_step"] = False
    with pytest.raises(ValueError, match="vivado_required"):
        exp1320.validate_artifact(impossible_vivado_gate)

    unknown_verdict = dict(artifact)
    unknown_verdict["honest_verdict"] = "made_a_hardware_claim_without_a_run"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1320.validate_artifact(unknown_verdict)


def test_req_hw_046_math_helpers_reject_invalid_inputs() -> None:
    """REQ-HW-046: helper failures are explicit for malformed tiny-case inputs."""
    with pytest.raises(ValueError, match="n_spins"):
        exp1320.enumerate_spin_states(0)

    with pytest.raises(ValueError, match="dac_bits"):
        exp1320.quantize_field(0.1, dac_bits=0)

    with pytest.raises(ValueError, match="same shape"):
        exp1320.kl_divergence(np.array([1.0, 0.0]), np.array([1.0]))
