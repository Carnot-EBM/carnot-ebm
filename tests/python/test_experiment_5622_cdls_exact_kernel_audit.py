"""Tests for Exp 5622 cDLS exact kernel audit.

Spec refs: REQ-SAMPLE-5622, SCENARIO-SAMPLE-5622.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_5622_cdls_exact_kernel_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5622_cdls_exact_kernel_audit.py")


def test_req_sample_5622_spec_declares_exact_kernel_audit_contract() -> None:
    """REQ-SAMPLE-5622: OpenSpec anchors exact targets, controls, and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5622") : spec.index("### REQ-SAMPLE-1746")]
    normalized = " ".join(section.split())

    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "discrete_dls_heat_bath",
        "uncorrected_cdls_projection",
        "corrected_cdls_projection_mh",
        "Metropolis-Hastings projection correction",
        "at least five independent seeds",
        "`deterministic_verifier`",
        "`kernel_audit_ready_score=1.0`",
        "SCENARIO-SAMPLE-5622",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_sample_5622_exact_transition_matrices_accept_and_reject_kernels() -> None:
    """REQ-SAMPLE-5622: exact matrices prove corrected target parity and reject controls."""

    system = mod.exact_ising_systems()[1]
    states = mod.enumerate_states(system.n_spins)
    target = mod.target_distribution(system, states)
    matrices = mod.transition_matrices(system, states, target)

    discrete_audit = mod.audit_transition_matrix(
        system=system,
        states=states,
        target=target,
        matrix=matrices["discrete_dls_heat_bath"],
        model_id="discrete_dls_heat_bath",
    )
    uncorrected_audit = mod.audit_transition_matrix(
        system=system,
        states=states,
        target=target,
        matrix=matrices["uncorrected_cdls_projection"],
        model_id="uncorrected_cdls_projection",
    )
    corrected_audit = mod.audit_transition_matrix(
        system=system,
        states=states,
        target=target,
        matrix=matrices["corrected_cdls_projection_mh"],
        model_id="corrected_cdls_projection_mh",
    )
    broken_audit = mod.audit_transition_matrix(
        system=system,
        states=states,
        target=target,
        matrix=mod.broken_proposal_control_matrix(len(states)),
        model_id="broken_zero_support_control",
    )

    assert discrete_audit["passes_exact_target_gate"] is True
    assert corrected_audit["passes_exact_target_gate"] is True
    assert corrected_audit["row_sum_error_max"] <= mod.EXACT_ROW_SUM_TOLERANCE
    assert corrected_audit["probability_min"] >= 0.0
    assert corrected_audit["irreducible"] is True
    assert corrected_audit["detailed_balance_residual_max"] <= mod.EXACT_BALANCE_TOLERANCE
    assert corrected_audit["stationary_distribution_tv"] <= mod.EXACT_TV_TOLERANCE
    assert corrected_audit["energy_histogram_tv"] <= mod.EXACT_TV_TOLERANCE
    assert uncorrected_audit["passes_exact_target_gate"] is False
    assert uncorrected_audit["stationary_distribution_tv"] > mod.BIASED_CONTROL_TV_FLOOR
    assert broken_audit["passes_exact_target_gate"] is False
    assert broken_audit["irreducible"] is False


def test_req_sample_5622_empirical_seed_replay_is_deterministic() -> None:
    """REQ-SAMPLE-5622: empirical checks replay exactly for a fixed seed."""

    system = mod.exact_ising_systems()[0]
    states = mod.enumerate_states(system.n_spins)
    target = mod.target_distribution(system, states)
    matrix = mod.transition_matrices(system, states, target)["corrected_cdls_projection_mh"]

    first = mod.empirical_distribution(matrix, seed=5622, retained_samples=512, burn_in_steps=64)
    second = mod.empirical_distribution(matrix, seed=5622, retained_samples=512, burn_in_steps=64)
    other = mod.empirical_distribution(matrix, seed=5623, retained_samples=512, burn_in_steps=64)
    interval = mod.empirical_tv_interval(
        system_id=system.system_id,
        model_id="corrected_cdls_projection_mh",
        matrix=matrix,
        target=target,
        seeds=mod.DEFAULT_RANDOM_SEEDS,
        retained_samples=512,
        burn_in_steps=64,
    )

    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert interval["seed_count"] == len(mod.DEFAULT_RANDOM_SEEDS)
    assert interval["seed_replay_match"] is True
    assert interval["tv_interval_95"][0] <= interval["tv_mean"] <= interval["tv_interval_95"][1]
    assert interval["tv_max"] < 0.20


def test_scenario_sample_5622_builds_valid_terminal_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5622: exact audit writes gated JSON evidence."""

    artifact = mod.build_artifact(
        retained_samples=1024,
        burn_in_steps=128,
        tests_added_or_reused=[TEST_PATH.as_posix()],
    )
    output_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert saved["schema"] == mod.SCHEMA
    assert saved["spec_refs"] == list(mod.SPEC_REFS)
    assert saved["target_descriptors"]
    assert {row["model_id"] for row in saved["models_tested"]} == {
        "discrete_dls_heat_bath",
        "uncorrected_cdls_projection",
        "corrected_cdls_projection_mh",
    }
    assert saved["state_space_sizes"]["max_exact_enumerated_states"] == 64
    assert saved["transition_row_sum_error_max"] <= mod.EXACT_ROW_SUM_TOLERANCE
    assert saved["detailed_balance_residual_max"] <= mod.EXACT_BALANCE_TOLERANCE
    assert saved["exact_distribution_tv_max"] <= mod.EXACT_TV_TOLERANCE
    assert saved["energy_histogram_tv_max"] <= mod.EXACT_TV_TOLERANCE
    assert saved["broken_kernel_controls_rejected"] is True
    assert saved["correction_applied"] is True
    assert saved["correction_spec"]["large_n_timing_tuned"] is False
    assert saved["quality_gate_specified_count"] >= 3
    assert saved["kernel_audit_ready_score"] == 1.0
    assert saved["inference_substrate"] == "deterministic_verifier"
    assert saved["random_seeds"] == list(mod.DEFAULT_RANDOM_SEEDS)
    assert "biased uncorrected kernel blocks timing" in saved["honest_verdict"]
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_sample_5622_validation_fails_closed_on_manual_gates() -> None:
    """REQ-SAMPLE-5622: validation rejects unsupported readiness and bad controls."""

    artifact = mod.build_artifact(retained_samples=512, burn_in_steps=64)
    mutations = [
        ("missing required field", lambda data: data.pop("field_principles")),
        ("field_principles", lambda data: data["field_principles"].update({"x": "y"})),
        ("inference_substrate", lambda data: data.__setitem__("inference_substrate", "llm")),
        ("correction_applied", lambda data: data.__setitem__("correction_applied", False)),
        (
            "large_n_timing_tuned",
            lambda data: data["correction_spec"].__setitem__("large_n_timing_tuned", True),
        ),
        (
            "quality_gate_specified_count",
            lambda data: data.__setitem__("quality_gate_specified_count", 2),
        ),
        (
            "broken_kernel_controls_rejected",
            lambda data: data.__setitem__("broken_kernel_controls_rejected", False),
        ),
        ("kernel_audit_ready_score", lambda data: data.__setitem__("kernel_audit_ready_score", 0.5)),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "bad")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"reproducibility_checksum", "missing required field"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)

    blocked = deepcopy(artifact)
    blocked["exact_distribution_tv_max"] = mod.EXACT_TV_TOLERANCE * 10.0
    blocked["kernel_audit_ready_score"] = mod.ready_score(blocked)
    blocked["honest_verdict"] = mod.honest_verdict(blocked)
    blocked["reproducibility_checksum"] = mod.payload_checksum(blocked)
    assert blocked["kernel_audit_ready_score"] == 0.0
    assert blocked["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(blocked)
