"""Exp 1291: HardNet++ nonlinear repair benchmark.

Spec refs: REQ-KONA-029, SCENARIO-KONA-029.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(PYTHON_DIR))

import numpy as np

from carnot.phase3.continuous_ebm import (
    AdaptiveRepairLayer,
    ContinuousEBM,
    feasibility_step,
    sample_langevin,
)
from carnot.phase3.nonlinear_repair import (
    NonlinearProjectionResult,
    hardnetpp_damped_projection,
    measure_nonlinear_violation,
    verified_span_reuse,
)

RUN_DATE = "20260504"
RESULT_PATH = Path("results/experiment_1291_hardnetpp_nonlinear_repair_benchmark.json")
VERIFIED_SPAN_INDICES = [2, 3]


def _energy(model: ContinuousEBM, state: np.ndarray) -> float:
    return float(-0.5 * state @ model.coupling @ state - model.bias @ state)


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([row[field] for row in rows]))


def _mean_pairwise_l2(states: list[np.ndarray]) -> float:
    distances: list[float] = []
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            distances.append(float(np.linalg.norm(states[left] - states[right])))
    return float(np.mean(distances)) if distances else 0.0


def _build_reference_model() -> ContinuousEBM:
    coupling = np.diag(
        [-2.4, -2.2, 0.18, 0.16, -0.30, -0.25],
    ).astype(np.float64)
    coupling[2, 3] = coupling[3, 2] = -0.04
    bias = np.array([0.0, 0.0, 0.42, -0.40, 0.03, -0.02], dtype=np.float64)
    return ContinuousEBM(variables=6, coupling=coupling, bias=bias)


def _nonlinear_constraints(state: np.ndarray) -> np.ndarray:
    x, y, copy_left, copy_right, guard_left, guard_right = state
    left_disk = (x + 0.55) ** 2 + y**2 - 0.18**2
    right_disk = (x - 0.55) ** 2 + y**2 - 0.18**2
    basin_membership = left_disk * right_disk
    copy_consistency = 0.05 * (copy_left + copy_right) ** 2 - 0.02
    guard_balance = 0.04 * (guard_left - guard_right) ** 2 - 0.12
    return np.array(
        [basin_membership, copy_consistency, guard_balance],
        dtype=np.float64,
    )


def _nonlinear_jacobian(state: np.ndarray) -> np.ndarray:
    x, y, copy_left, copy_right, guard_left, guard_right = state
    left_disk = (x + 0.55) ** 2 + y**2 - 0.18**2
    right_disk = (x - 0.55) ** 2 + y**2 - 0.18**2
    basin_dx = (2.0 * (x + 0.55) * right_disk) + (
        2.0 * (x - 0.55) * left_disk
    )
    basin_dy = 2.0 * y * (left_disk + right_disk)
    copy_grad = 0.10 * (copy_left + copy_right)
    guard_grad = 0.08 * (guard_left - guard_right)
    return np.array(
        [
            [basin_dx, basin_dy, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, copy_grad, copy_grad, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, guard_grad, -guard_grad],
        ],
        dtype=np.float64,
    )


def _linearize_at(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = _nonlinear_jacobian(state)
    bias = _nonlinear_constraints(state) - matrix @ state
    return matrix, bias


def _true_violation(state: np.ndarray) -> tuple[float, int]:
    return measure_nonlinear_violation(_nonlinear_constraints(state), tolerance=1e-8)


def _arm_summary(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    states: list[np.ndarray],
    raw_diversity: float,
) -> dict[str, float]:
    diversity = _mean_pairwise_l2(states)
    diversity_ratio = float(diversity / raw_diversity) if raw_diversity else 1.0
    return {
        "final_energy_mean": _mean(rows, f"{prefix}_final_energy"),
        "violation_energy_mean": _mean(rows, f"{prefix}_violation_energy"),
        "violation_count_mean": _mean(rows, f"{prefix}_violation_count"),
        "convergence_steps_mean": _mean(rows, f"{prefix}_convergence_steps"),
        "distortion_from_initial_mean": _mean(rows, f"{prefix}_distortion_from_initial"),
        "diversity_mean_pairwise_l2": diversity,
        "diversity_ratio_vs_raw": diversity_ratio,
        "verified_span_reuse_mean": _mean(rows, f"{prefix}_verified_span_reuse"),
    }


def _hardnetpp_result(raw_state: np.ndarray) -> NonlinearProjectionResult:
    return hardnetpp_damped_projection(
        raw_state,
        _nonlinear_constraints,
        _nonlinear_jacobian,
        n_steps=40,
        damping=5e-4,
        step_size=0.85,
        anchor_weight=0.008,
        tolerance=1e-8,
        verified_span_indices=VERIFIED_SPAN_INDICES,
    )


def _classify_hardnetpp_verdict(
    hardnetpp_delta_over_snarenet: float,
    hardnetpp_summary: dict[str, float],
) -> tuple[bool, str]:
    nonlinear_repair_viable = bool(
        hardnetpp_delta_over_snarenet > 0.0
        and hardnetpp_summary["violation_count_mean"] <= 0.25
        and hardnetpp_summary["diversity_ratio_vs_raw"] >= 0.75
        and hardnetpp_summary["verified_span_reuse_mean"] >= 0.85
    )
    if nonlinear_repair_viable:
        return True, "hardnetpp_nonlinear_repair_viable"
    if hardnetpp_delta_over_snarenet > 0.0:
        return False, "hardnetpp_nonlinear_repair_marginal"
    return False, "hardnetpp_nonlinear_repair_not_viable"


def _result_row(
    seed: int,
    model: ContinuousEBM,
    raw_state: np.ndarray,
    fsnet_state: np.ndarray,
    fsnet_steps: int,
    snarenet_state: np.ndarray,
    snarenet_steps: int,
    hardnetpp_result: NonlinearProjectionResult,
) -> dict[str, Any]:
    raw_violation_energy, raw_violation_count = _true_violation(raw_state)
    fsnet_violation_energy, fsnet_violation_count = _true_violation(fsnet_state)
    snarenet_violation_energy, snarenet_violation_count = _true_violation(snarenet_state)
    return {
        "seed": seed,
        "raw_final_energy": _energy(model, raw_state),
        "fsnet_final_energy": _energy(model, fsnet_state),
        "snarenet_final_energy": _energy(model, snarenet_state),
        "hardnetpp_final_energy": _energy(model, hardnetpp_result.state),
        "raw_violation_energy": raw_violation_energy,
        "fsnet_violation_energy": fsnet_violation_energy,
        "snarenet_violation_energy": snarenet_violation_energy,
        "hardnetpp_violation_energy": hardnetpp_result.violation_energy,
        "raw_violation_count": raw_violation_count,
        "fsnet_violation_count": fsnet_violation_count,
        "snarenet_violation_count": snarenet_violation_count,
        "hardnetpp_violation_count": hardnetpp_result.violation_count,
        "raw_convergence_steps": 0,
        "fsnet_convergence_steps": fsnet_steps,
        "snarenet_convergence_steps": snarenet_steps,
        "hardnetpp_convergence_steps": hardnetpp_result.convergence_steps,
        "raw_distortion_from_initial": 0.0,
        "fsnet_distortion_from_initial": float(np.linalg.norm(fsnet_state - raw_state)),
        "snarenet_distortion_from_initial": float(
            np.linalg.norm(snarenet_state - raw_state)
        ),
        "hardnetpp_distortion_from_initial": hardnetpp_result.distortion_l2,
        "raw_verified_span_reuse": 1.0,
        "fsnet_verified_span_reuse": verified_span_reuse(
            raw_state,
            fsnet_state,
            VERIFIED_SPAN_INDICES,
        ),
        "snarenet_verified_span_reuse": verified_span_reuse(
            raw_state,
            snarenet_state,
            VERIFIED_SPAN_INDICES,
        ),
        "hardnetpp_verified_span_reuse": hardnetpp_result.verified_span_reuse,
        "hardnetpp_converged": hardnetpp_result.converged,
        "hardnetpp_final_step_norm": hardnetpp_result.final_step_norm,
    }


def build_artifact() -> dict[str, Any]:
    model = _build_reference_model()
    repair_layer = AdaptiveRepairLayer(
        fsnet_steps=8,
        fsnet_lr=0.42,
        fsnet_anchor_weight=0.01,
        n_steps=8,
        lr=0.16,
        anchor_weight=0.01,
        initial_relaxation=0.15,
        min_relaxation=0.04,
        max_relaxation=0.45,
        tolerance=1e-10,
    )

    raw_states: list[np.ndarray] = []
    fsnet_states: list[np.ndarray] = []
    snarenet_states: list[np.ndarray] = []
    hardnetpp_states: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for seed in range(18):
        raw_state = sample_langevin(
            model,
            n_steps=90,
            lr=0.018,
            noise_scale=0.045,
            temp_schedule="cosine",
            seed=1291 + seed,
        )
        matrix, bias = _linearize_at(raw_state)
        fsnet_result = feasibility_step(
            raw_state,
            matrix,
            bias,
            n_steps=8,
            lr=0.42,
            anchor_weight=0.01,
            tolerance=1e-8,
        )
        snarenet_result = repair_layer.repair(raw_state, matrix, bias)
        hardnetpp_result = _hardnetpp_result(raw_state)

        raw_states.append(raw_state)
        fsnet_states.append(fsnet_result.state)
        snarenet_states.append(snarenet_result.state)
        hardnetpp_states.append(hardnetpp_result.state)
        rows.append(
            _result_row(
                seed,
                model,
                raw_state,
                fsnet_result.state,
                fsnet_result.convergence_steps,
                snarenet_result.state,
                snarenet_result.repair_iterations,
                hardnetpp_result,
            )
        )

    raw_diversity = _mean_pairwise_l2(raw_states)
    raw_summary = _arm_summary(
        rows,
        prefix="raw",
        states=raw_states,
        raw_diversity=raw_diversity,
    )
    fsnet_summary = _arm_summary(
        rows,
        prefix="fsnet",
        states=fsnet_states,
        raw_diversity=raw_diversity,
    )
    snarenet_summary = _arm_summary(
        rows,
        prefix="snarenet",
        states=snarenet_states,
        raw_diversity=raw_diversity,
    )
    hardnetpp_summary = _arm_summary(
        rows,
        prefix="hardnetpp",
        states=hardnetpp_states,
        raw_diversity=raw_diversity,
    )

    hardnetpp_delta_over_snarenet = float(
        (snarenet_summary["violation_count_mean"] - hardnetpp_summary["violation_count_mean"])
        + 0.25
        * (
            hardnetpp_summary["verified_span_reuse_mean"]
            - snarenet_summary["verified_span_reuse_mean"]
        )
        + 0.10
        * (
            hardnetpp_summary["diversity_ratio_vs_raw"]
            - snarenet_summary["diversity_ratio_vs_raw"]
        )
        - 0.10
        * (
            hardnetpp_summary["distortion_from_initial_mean"]
            - snarenet_summary["distortion_from_initial_mean"]
        )
    )
    nonlinear_repair_viable, honest_verdict = _classify_hardnetpp_verdict(
        hardnetpp_delta_over_snarenet,
        hardnetpp_summary,
    )

    return {
        "schema": "carnot.phase3.hardnetpp_nonlinear_repair.v1",
        "experiment": "1291_hardnetpp_nonlinear_repair_benchmark",
        "run_date": RUN_DATE,
        "status": "complete",
        "spec_refs": ["REQ-KONA-029", "SCENARIO-KONA-029"],
        "latent_dim": model.variables,
        "n_states": len(rows),
        "constraint_cases": {
            "description": (
                "Product-of-disks nonlinear inequality with two valid basins "
                "around x=-0.55 and x=0.55 plus an energy-preferred misleading "
                "local basin near x=0."
            ),
            "n_constraints": 3,
            "valid_basin_count": 2,
            "misleading_local_basin_count": 1,
            "valid_basin_centers": [[-0.55, 0.0], [0.55, 0.0]],
            "misleading_local_basin_center": [0.0, 0.0],
            "verified_span_indices": VERIFIED_SPAN_INDICES,
        },
        "arms": {
            "raw_langevin": raw_summary,
            "fsnet_fixed_local_linear": fsnet_summary,
            "snarenet_fixed_local_linear": snarenet_summary,
            "hardnetpp_damped_projection": hardnetpp_summary,
        },
        "hardnetpp_delta_over_snarenet": hardnetpp_delta_over_snarenet,
        "nonlinear_repair_viable": nonlinear_repair_viable,
        "construct_refine_iterations": hardnetpp_summary["convergence_steps_mean"],
        "copy_as_decode_verified_span_reuse": hardnetpp_summary[
            "verified_span_reuse_mean"
        ],
        "honest_verdict": honest_verdict,
        "per_seed": rows,
    }


def main() -> dict[str, Any]:
    artifact = build_artifact()
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
