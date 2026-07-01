"""Tests for Exp 5116 HUBO/p-spin 2D parallel tempering.

Spec refs: REQ-SAMPLE-5116, SCENARIO-SAMPLE-5116.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5116_hubo_2dpt_sampling_reference_v469 as exp
from carnot.samplers.hubo_2dpt import (
    Hubo2DPTConfig,
    Hubo2DParallelTemperingSampler,
    build_synthetic_hubo_families,
    evaluate_hubo_energy,
    exact_enumerate,
)
from scripts import experiment_5116_hubo_2dpt_sampling_reference_v469 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
ARTIFACT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_sample_5116_spec_declares_hubo_2dpt_contract() -> None:
    """REQ-SAMPLE-5116: OpenSpec declares the CPU HUBO 2D PT contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5116") :]

    for marker in (
        "REQ-SAMPLE-5116",
        "SCENARIO-SAMPLE-5116",
        "HUBO/p-spin",
        "beta/penalty parallel tempering",
        exp.RESULT_RELATIVE_PATH,
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_sample_5116_exact_enumeration_matches_direct_energy_distribution() -> None:
    """REQ-SAMPLE-5116: exact enumeration covers optima and energy distributions."""

    problems = build_synthetic_hubo_families()
    families = {problem.family for problem in problems}

    assert len(families) >= 2
    for problem in problems:
        exact = exact_enumerate(problem, penalty=4.0)
        brute_force = {
            state: evaluate_hubo_energy(problem, state, penalty=4.0)
            for state in exact.all_states
        }

        assert sum(exact.energy_distribution.values()) == 2**problem.n_vars
        assert exact.optimum_energy == min(brute_force.values())
        assert set(exact.optimal_states) == {
            state for state, energy in brute_force.items() if energy == exact.optimum_energy
        }
        assert all(count > 0 for count in exact.energy_distribution.values())


def test_scenario_sample_5116_swap_bookkeeping_records_beta_and_penalty_axes() -> None:
    """SCENARIO-SAMPLE-5116: 2D PT records beta-axis and penalty-axis swaps."""

    problem = build_synthetic_hubo_families()[0]
    config = Hubo2DPTConfig(beta_grid=(0.4, 1.0), penalty_grid=(0.5, 2.0), sweeps=6)
    result = Hubo2DParallelTemperingSampler(config).run(problem, seed=5116)

    beta_stats = result.swap_stats["beta_axis"]
    penalty_stats = result.swap_stats["penalty_axis"]

    assert beta_stats.attempts > 0
    assert penalty_stats.attempts > 0
    assert beta_stats.accepted <= beta_stats.attempts
    assert penalty_stats.accepted <= penalty_stats.attempts
    assert 0.0 <= beta_stats.acceptance_rate <= 1.0
    assert 0.0 <= penalty_stats.acceptance_rate <= 1.0


def test_req_sample_5116_seed_reproducibility_for_cpu_sampler() -> None:
    """REQ-SAMPLE-5116: fixed seeds reproduce the same 2D PT run exactly."""

    problem = build_synthetic_hubo_families()[1]
    config = Hubo2DPTConfig(beta_grid=(0.3, 0.9, 1.7), penalty_grid=(0.5, 1.5), sweeps=8)
    sampler = Hubo2DParallelTemperingSampler(config)

    first = sampler.run(problem, seed=12345).as_dict()
    second = sampler.run(problem, seed=12345).as_dict()

    assert first == second


def test_req_sample_5116_artifact_fields_ready_gate_and_no_hardware_claim(tmp_path: Path) -> None:
    """REQ-SAMPLE-5116: artifact emits required fields and honest ready gate."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_hubo_2dpt_sampling_reference_5116.py"],
    )
    payload = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["exact_enumeration_checked"] is True
    assert artifact["hubo_2dpt_reference_ready"] is True
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["optimum_hit_rate"]["two_d_beta_penalty_pt"] >= artifact["optimum_hit_rate"]["unguided_gibbs"]
    assert artifact["best_energy_delta_vs_baselines"]["two_d_vs_unguided_gibbs"] <= 0.0
    assert artifact["best_energy_delta_vs_baselines"]["two_d_vs_beta_pt"] <= 0.0


def test_scenario_sample_5116_script_entrypoint_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5116: CLI wrapper writes the configured JSON artifact."""

    path = script_mod.main(
        root=tmp_path,
        date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_hubo_2dpt_sampling_reference_5116.py"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp.RESULT_RELATIVE_PATH
    exp.validate_artifact(payload)
    assert payload["hardware_speedup_claimed"] is False


def test_deliverable_file_validates_for_scenario_sample_5116() -> None:
    """SCENARIO-SAMPLE-5116: checked-in deliverable satisfies the terminal schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["hubo_2dpt_reference_ready"] is True
