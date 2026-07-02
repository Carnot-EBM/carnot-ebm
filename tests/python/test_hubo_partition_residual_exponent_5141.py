"""Tests for Exp 5141 partitioned HUBO/Ising 2D PT telemetry.

Spec refs: REQ-SAMPLE-5141, SCENARIO-SAMPLE-5141.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5141_hubo_partition_residual_exponent_v471 as exp
from scripts import experiment_5141_hubo_partition_residual_exponent_v471 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
ARTIFACT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_sample_5141_spec_declares_partition_telemetry_contract() -> None:
    """REQ-SAMPLE-5141: OpenSpec declares the partition telemetry contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5141") :]

    for marker in (
        "REQ-SAMPLE-5141",
        "SCENARIO-SAMPLE-5141",
        "partitioned 2D-PT variants",
        exp.RESULT_RELATIVE_PATH,
        "board-ready KV260, GateMate, and PolarFire workload descriptors",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in exp.FIELD_PRINCIPLES


def test_req_sample_5141_exact_enumerable_hubo_and_ising_instances() -> None:
    """REQ-SAMPLE-5141: generated HUBO and Ising instances exact-enumerate."""

    problems = exp.build_partition_telemetry_instances()
    families = {problem.family for problem in problems}

    assert "ising_pairwise_ring" in families
    assert any("hubo" in family or "parity" in family for family in families)
    for problem in problems:
        exact = exp.exact_enumerate(problem, penalty=exp.TARGET_PENALTY)

        assert exact.optimum_energy == min(
            exp.evaluate_hubo_energy(problem, state, penalty=exp.TARGET_PENALTY)
            for state in exact.all_states
        )
        assert sum(exact.energy_distribution.values()) == 2**problem.n_vars
        assert exact.optimal_states


def test_scenario_sample_5141_partitioned_sampler_records_boundary_and_residual_metrics() -> None:
    """SCENARIO-SAMPLE-5141: partitioned 2D PT records boundary and residual telemetry."""

    problem = exp.build_partition_telemetry_instances()[0]
    exact = exp.exact_enumerate(problem, penalty=exp.TARGET_PENALTY)
    partitions = exp.partition_layout_for_n_vars("contiguous_2", problem.n_vars)
    config = exp.Partitioned2DPTConfig(
        beta_grid=(0.35, 0.8),
        penalty_grid=(1.0, exp.TARGET_PENALTY),
        sweeps=10,
        partition_layout_id="contiguous_2",
        partitions=partitions,
        boundary_refresh_ratio=0.0,
    )

    result = exp.run_partitioned_2dpt(problem, config, seed=5141, exact_optimum_energy=exact.optimum_energy)
    residual = exp.residual_trace(result.energy_trace, exact.optimum_energy)
    exponent = exp.fit_residual_energy_exponent(residual, window=(1, len(residual)))

    assert result.algorithm == "partitioned_2dpt"
    assert result.boundary_mismatch_rate > 0.0
    assert result.swap_stats["beta_axis"].attempts > 0
    assert result.swap_stats["penalty_axis"].attempts > 0
    assert len(residual) == config.sweeps
    assert all(value >= 0.0 for value in residual)
    assert exponent >= 0.0


def test_req_sample_5141_detailed_balance_passes_or_records_blocker() -> None:
    """REQ-SAMPLE-5141: detailed balance is verified or blocked explicitly."""

    problem = exp.build_partition_telemetry_instances()[0]
    partitions = exp.partition_layout_for_n_vars("checkerboard_2", problem.n_vars)

    checked = exp.detailed_balance_evidence_for_variant(
        problem,
        partitions=partitions,
        beta=0.8,
        penalty=exp.TARGET_PENALTY,
        boundary_refresh_ratio=1.0,
    )
    blocked = exp.detailed_balance_evidence_for_variant(
        problem,
        partitions=partitions,
        beta=0.8,
        penalty=exp.TARGET_PENALTY,
        boundary_refresh_ratio=0.5,
    )

    assert checked["checked"] is True
    assert checked["passed"] is True
    assert checked["max_abs_probability_flow_error"] <= 1e-9
    assert blocked["checked"] is False
    assert blocked["passed"] is False
    assert "stale boundary" in blocked["blocker"]


def test_req_sample_5141_artifact_schema_ready_gate_and_no_hardware_claim(tmp_path: Path) -> None:
    """REQ-SAMPLE-5141: artifact emits required fields and honest ready gate."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260702",
        duration_s=1.0,
        tests_run=["tests/python/test_hubo_partition_residual_exponent_5141.py"],
    )
    payload = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["exp5129_baseline_loaded"] is True
    assert artifact["exact_enumeration_checked"] is True
    assert artifact["detailed_balance_evidence"]["all_unblocked_variants_passed"] is True
    assert artifact["partition_telemetry_ready"] is True
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["monolithic_reference"]["optimum_hit_rate"] >= artifact["unguided_baseline"]["optimum_hit_rate"]
    assert all(descriptor["workload_hash"] for descriptor in artifact["board_ready_workload_descriptors"])


def test_req_sample_5141_validation_rejects_missing_required_field(tmp_path: Path) -> None:
    """REQ-SAMPLE-5141: artifact validation rejects malformed terminal payloads."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260702",
        duration_s=1.0,
        tests_run=["tests/python/test_hubo_partition_residual_exponent_5141.py"],
    )
    malformed = dict(artifact)
    malformed.pop("partition_telemetry_ready")

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(malformed)


def test_scenario_sample_5141_script_entrypoint_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5141: CLI wrapper writes the configured JSON artifact."""

    path = script_mod.main(
        root=tmp_path,
        date="20260702",
        duration_s=1.0,
        tests_run=["tests/python/test_hubo_partition_residual_exponent_5141.py"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp.RESULT_RELATIVE_PATH
    exp.validate_artifact(payload)
    assert payload["hardware_speedup_claimed"] is False
    assert payload["conductor_modified"] is False


def test_deliverable_file_validates_for_scenario_sample_5141() -> None:
    """SCENARIO-SAMPLE-5141: checked-in deliverable satisfies terminal schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["partition_telemetry_ready"] is True
