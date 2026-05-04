"""Tests for Exp 1291 HardNet++ nonlinear repair benchmarking.

Spec refs: REQ-KONA-029, SCENARIO-KONA-029.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase3.nonlinear_repair import (
    hardnetpp_damped_projection,
    measure_nonlinear_violation,
    verified_span_reuse,
)
from scripts import experiment_1291_hardnetpp_nonlinear_repair_benchmark as experiment


def _two_basin_constraints(state: np.ndarray) -> np.ndarray:
    x, y, copy_left, copy_right = state
    left_disk = (x + 0.55) ** 2 + y**2 - 0.18**2
    right_disk = (x - 0.55) ** 2 + y**2 - 0.18**2
    basin_membership = left_disk * right_disk
    copy_consistency = 0.05 * (copy_left + copy_right) ** 2 - 0.02
    return np.array([basin_membership, copy_consistency], dtype=np.float64)


def _two_basin_jacobian(state: np.ndarray) -> np.ndarray:
    x, y, copy_left, copy_right = state
    left_disk = (x + 0.55) ** 2 + y**2 - 0.18**2
    right_disk = (x - 0.55) ** 2 + y**2 - 0.18**2
    basin_dx = (2.0 * (x + 0.55) * right_disk) + (
        2.0 * (x - 0.55) * left_disk
    )
    basin_dy = 2.0 * y * (left_disk + right_disk)
    copy_grad = 0.10 * (copy_left + copy_right)
    return np.array(
        [
            [basin_dx, basin_dy, 0.0, 0.0],
            [0.0, 0.0, copy_grad, copy_grad],
        ],
        dtype=np.float64,
    )


def test_hardnetpp_projection_repairs_nonlinear_two_basin_case() -> None:
    """REQ-KONA-029: damped relinearising projection repairs true nonlinear violations."""
    state = np.array([0.12, 0.10, 0.48, -0.52], dtype=np.float64)

    result = hardnetpp_damped_projection(
        state,
        _two_basin_constraints,
        _two_basin_jacobian,
        n_steps=32,
        damping=1e-3,
        step_size=0.8,
        anchor_weight=0.01,
        tolerance=1e-8,
        verified_span_indices=[2, 3],
    )

    initial_energy, initial_count = measure_nonlinear_violation(
        _two_basin_constraints(state)
    )

    assert result.initial_violation_energy == pytest.approx(initial_energy)
    assert initial_count == 1
    assert result.violation_energy < initial_energy * 0.01
    assert result.violation_count == 0
    assert result.converged is True
    assert result.convergence_steps > 0
    assert result.state.shape == state.shape
    assert np.all(result.state > -1.0)
    assert np.all(result.state < 1.0)
    assert result.distortion_l2 > 0.0
    assert result.verified_span_reuse == pytest.approx(1.0)


def test_hardnetpp_projection_rejects_malformed_inputs() -> None:
    """REQ-KONA-029: malformed nonlinear repair inputs fail before projection."""
    state = np.array([0.1, 0.2, 0.3, -0.3], dtype=np.float64)

    with pytest.raises(ValueError, match="state must be one-dimensional"):
        hardnetpp_damped_projection(
            np.zeros((1, 4)),
            _two_basin_constraints,
            _two_basin_jacobian,
        )

    with pytest.raises(ValueError, match="constraint_fn"):
        hardnetpp_damped_projection(
            state,
            lambda _: np.zeros((1, 1), dtype=np.float64),
            _two_basin_jacobian,
        )

    with pytest.raises(ValueError, match="jacobian_fn"):
        hardnetpp_damped_projection(
            state,
            _two_basin_constraints,
            lambda _: np.zeros((3, 4), dtype=np.float64),
        )

    with pytest.raises(ValueError, match="n_steps"):
        hardnetpp_damped_projection(
            state,
            _two_basin_constraints,
            _two_basin_jacobian,
            n_steps=-1,
        )

    with pytest.raises(ValueError, match="damping"):
        hardnetpp_damped_projection(
            state,
            _two_basin_constraints,
            _two_basin_jacobian,
            damping=0.0,
        )

    with pytest.raises(ValueError, match="step_size"):
        hardnetpp_damped_projection(
            state,
            _two_basin_constraints,
            _two_basin_jacobian,
            step_size=-0.1,
        )

    with pytest.raises(ValueError, match="anchor_weight"):
        hardnetpp_damped_projection(
            state,
            _two_basin_constraints,
            _two_basin_jacobian,
            anchor_weight=-0.1,
        )

    with pytest.raises(ValueError, match="tolerance"):
        hardnetpp_damped_projection(
            state,
            _two_basin_constraints,
            _two_basin_jacobian,
            tolerance=-1e-9,
        )


def test_hardnetpp_projection_covers_verified_span_and_dynamic_shape_guards() -> None:
    """REQ-KONA-029: projection diagnostics cover reusable spans and dynamic shapes."""
    state = np.array([0.1, 0.2], dtype=np.float64)

    assert verified_span_reuse(state, state.copy(), None) == pytest.approx(1.0)
    assert verified_span_reuse(state, state.copy(), []) == pytest.approx(1.0)

    call_count = {"constraint": 0}

    def malformed_later_constraint(_: np.ndarray) -> np.ndarray:
        call_count["constraint"] += 1
        if call_count["constraint"] == 1:
            return np.array([0.2], dtype=np.float64)
        return np.zeros((1, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="constraint_fn"):
        hardnetpp_damped_projection(
            state,
            malformed_later_constraint,
            lambda _: np.array([[1.0, 0.0]], dtype=np.float64),
        )

    call_count = {"jacobian": 0}

    def malformed_later_jacobian(_: np.ndarray) -> np.ndarray:
        call_count["jacobian"] += 1
        if call_count["jacobian"] == 1:
            return np.array([[1.0, 0.0]], dtype=np.float64)
        return np.zeros((2, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="jacobian_fn"):
        hardnetpp_damped_projection(
            state,
            lambda _: np.array([0.2], dtype=np.float64),
            malformed_later_jacobian,
        )

    near_feasible_scores = np.full(200, 0.009, dtype=np.float64)
    result = hardnetpp_damped_projection(
        np.array([0.0], dtype=np.float64),
        lambda _: near_feasible_scores,
        lambda _: np.zeros((200, 1), dtype=np.float64),
        tolerance=0.01,
    )

    assert result.violation_count == 0
    assert result.convergence_steps == 0
    assert result.distortion_l2 == pytest.approx(0.0)


def test_build_artifact_contains_required_exp1291_fields() -> None:
    """SCENARIO-KONA-029: artifact compares all nonlinear repair arms."""
    artifact = experiment.build_artifact()

    assert artifact["schema"] == "carnot.phase3.hardnetpp_nonlinear_repair.v1"
    assert artifact["experiment"] == "1291_hardnetpp_nonlinear_repair_benchmark"
    assert artifact["run_date"] == "20260504"
    assert artifact["status"] == "complete"
    assert artifact["spec_refs"] == ["REQ-KONA-029", "SCENARIO-KONA-029"]
    assert artifact["constraint_cases"]["valid_basin_count"] >= 2
    assert artifact["constraint_cases"]["misleading_local_basin_count"] >= 1
    assert set(artifact["arms"]) == {
        "raw_langevin",
        "fsnet_fixed_local_linear",
        "snarenet_fixed_local_linear",
        "hardnetpp_damped_projection",
    }
    for summary in artifact["arms"].values():
        assert "final_energy_mean" in summary
        assert "violation_count_mean" in summary
        assert "convergence_steps_mean" in summary
        assert "distortion_from_initial_mean" in summary
        assert "diversity_mean_pairwise_l2" in summary
        assert "verified_span_reuse_mean" in summary

    hardnetpp = artifact["arms"]["hardnetpp_damped_projection"]
    snarenet = artifact["arms"]["snarenet_fixed_local_linear"]
    assert artifact["hardnetpp_delta_over_snarenet"] > 0.0
    assert artifact["nonlinear_repair_viable"] is True
    assert artifact["construct_refine_iterations"] == pytest.approx(
        hardnetpp["convergence_steps_mean"]
    )
    assert artifact["copy_as_decode_verified_span_reuse"] == pytest.approx(
        hardnetpp["verified_span_reuse_mean"]
    )
    assert hardnetpp["violation_count_mean"] < snarenet["violation_count_mean"]
    assert artifact["honest_verdict"] == "hardnetpp_nonlinear_repair_viable"
    assert len(artifact["per_seed"]) == artifact["n_states"]
    json.dumps(artifact)


def test_hardnetpp_verdict_classification_covers_nonviable_branches() -> None:
    """SCENARIO-KONA-029: honest verdicts distinguish marginal and failed repairs."""
    viable_summary = {
        "violation_count_mean": 0.0,
        "diversity_ratio_vs_raw": 1.0,
        "verified_span_reuse_mean": 1.0,
    }
    marginal_summary = {
        "violation_count_mean": 1.0,
        "diversity_ratio_vs_raw": 1.0,
        "verified_span_reuse_mean": 1.0,
    }

    assert experiment._classify_hardnetpp_verdict(0.5, viable_summary) == (
        True,
        "hardnetpp_nonlinear_repair_viable",
    )
    assert experiment._classify_hardnetpp_verdict(0.5, marginal_summary) == (
        False,
        "hardnetpp_nonlinear_repair_marginal",
    )
    assert experiment._classify_hardnetpp_verdict(-0.1, viable_summary) == (
        False,
        "hardnetpp_nonlinear_repair_not_viable",
    )


def test_script_main_writes_terminal_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-KONA-029: experiment script writes the terminal JSON artifact."""
    output_path = tmp_path / "experiment_1291_hardnetpp_nonlinear_repair_benchmark.json"
    monkeypatch.setattr(experiment, "RESULT_PATH", output_path)

    artifact = experiment.main()

    assert output_path.exists()
    written = json.loads(output_path.read_text())
    assert written == artifact
    assert written["status"] == "complete"
    assert written["honest_verdict"] == "hardnetpp_nonlinear_repair_viable"
