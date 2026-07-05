"""Tests for Exp 5278 solver fixture to factor-graph boundary.

Spec refs: REQ-VERIFY-5278, SCENARIO-VERIFY-5278.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from carnot import experiment_5278_constraint_factor_graph_boundary_v482 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def test_req_verify_5278_spec_declares_boundary_contract() -> None:
    """REQ-VERIFY-5278: OpenSpec anchors the deterministic boundary artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5278") : spec.index("### REQ-VERIFY-5263")
    ]

    for marker in (
        "REQ-VERIFY-5278",
        "SCENARIO-VERIFY-5278",
        str(mod.RESULT_RELATIVE_PATH),
        "offline_deterministic_certificate_no_llm",
        "hardware_speedup_claimed.value=false",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5278_solver_assignment_roundtrips_to_zero_energy() -> None:
    """REQ-VERIFY-5278: the Exp 5273 witness maps through bits and energy."""

    source = mod.select_tiny_fixture()
    boundary = mod.build_boundary(source)
    witness = {"a": 2, "b": 3}

    bits = boundary.assignment_to_bits(witness)
    decoded = boundary.bits_to_assignment(bits)
    evaluation = boundary.evaluate_assignment(witness)
    qubo_energy = boundary.qubo_energy(bits)
    roundtrip = mod.roundtrip_assignment(boundary, witness)

    assert source["fixture_id"] == "small_pair_sum"
    assert decoded == witness
    assert boundary.bit_order == (
        "a_0",
        "a_1",
        "a_2",
        "a_3",
        "b_0",
        "b_1",
        "b_2",
        "b_3",
    )
    assert bits == (0, 0, 1, 0, 0, 0, 0, 1)
    assert evaluation["total_violation"] == 0
    assert evaluation["constraint_violations"] == {
        "a_domain": 0,
        "b_domain": 0,
        "sum_is_five": 0,
        "a_less_than_b": 0,
    }
    assert qubo_energy == 0.0
    assert roundtrip["passed"] is True
    assert roundtrip["decoded_assignment"] == witness
    assert roundtrip["energy"] == 0.0


def test_req_verify_5278_false_assignment_is_rejected_and_enumeration_is_exact() -> None:
    """REQ-VERIFY-5278: rejecting fixture assignment has positive violation."""

    boundary = mod.build_boundary(mod.select_tiny_fixture())
    false_assignment = {"a": 3, "b": 2}
    false_bits = boundary.assignment_to_bits(false_assignment)
    false_eval = boundary.evaluate_assignment(false_assignment)
    false_energy = boundary.qubo_energy(false_bits)
    rejection = mod.reject_false_assignment(boundary, false_assignment)
    enumeration = mod.enumerate_boundary(boundary)

    assert false_eval["constraint_violations"]["sum_is_five"] == 0
    assert false_eval["constraint_violations"]["a_less_than_b"] == 1
    assert false_energy > 0.0
    assert rejection["rejected"] is True
    assert rejection["energy"] == false_energy
    assert enumeration["state_count"] == 256
    assert enumeration["valid_onehot_state_count"] == 16
    assert enumeration["best_energy"] == 0.0
    assert enumeration["best_assignments"] == [{"a": 2, "b": 3}]
    assert enumeration["autocorrelation_metric"] is None


def test_scenario_verify_5278_writes_no_speedup_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5278: artifact fields prove boundary, not speedup."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        result_path=result_path,
        tests_run=[{"command": "unit exp5278", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "factor-graph boundary is usable" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["factor_graph_boundary_ready"]["value"] is True
    assert artifact["sampler_interface_ready"]["value"] is True
    assert artifact["mapping_roundtrip_passed"]["value"] is True
    assert artifact["false_assignment_rejected"]["value"] is True
    assert artifact["autocorrelation_metric"]["value"] is None
    assert artifact["hardware_speedup_claimed"]["value"] is False
    assert artifact["tests_run"] == [{"command": "unit exp5278", "outcome": "passed"}]

    sampler = artifact["sampler_interface"]
    assert sampler["backend_protocol"] == "SamplerBackend"
    assert sampler["bias_shape"] == [8]
    assert sampler["coupling_shape"] == [8, 8]
    assert sampler["cpu_enumerator_state_count"] == 256
    assert sampler["hardware_board_command_run"] is False
    assert sampler["speedup_claimed"] is False
    assert np.asarray(sampler["biases"], dtype=float).shape == (8,)
    assert np.asarray(sampler["couplings"], dtype=float).shape == (8, 8)
