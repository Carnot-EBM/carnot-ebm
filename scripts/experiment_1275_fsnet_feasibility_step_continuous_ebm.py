"""Exp 1275: FSNet-style feasibility step for ContinuousEBM latents.

Spec refs: REQ-KONA-027, SCENARIO-KONA-027.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot.phase3.continuous_ebm import (
    ContinuousEBM,
    FeasibilityStepResult,
    feasibility_step,
    sample_langevin,
)

RUN_DATE = "20260504"
RESULT_PATH = Path("results/experiment_1275_fsnet_feasibility_step_continuous_ebm.json")


def _energy(model: ContinuousEBM, state: np.ndarray) -> float:
    return float(-0.5 * state @ model.coupling @ state - model.bias @ state)


def _mean_pairwise_l2(states: list[np.ndarray]) -> float:
    distances: list[float] = []
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            distances.append(float(np.linalg.norm(states[left] - states[right])))
    return float(np.mean(distances)) if distances else 0.0


def _build_reference_model() -> ContinuousEBM:
    rng = np.random.default_rng(1275)
    raw_coupling = rng.normal(0.0, 0.08, size=(10, 10))
    coupling = (raw_coupling + raw_coupling.T) / 2.0
    np.fill_diagonal(coupling, 0.0)
    bias = np.array(
        [0.80, 0.75, 0.70, 0.35, 0.25, 0.05, -0.03, 0.02, -0.01, 0.04],
        dtype=np.float64,
    )
    return ContinuousEBM(variables=10, coupling=coupling, bias=bias)


def _fover_like_constraints() -> tuple[np.ndarray, np.ndarray]:
    constraint_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    constraint_bias = np.array([-0.05, -0.05, -0.05, -0.10, -0.12], dtype=np.float64)
    return constraint_matrix, constraint_bias


def _result_row(
    seed: int,
    model: ContinuousEBM,
    raw_state: np.ndarray,
    raw_result: FeasibilityStepResult,
    repaired_result: FeasibilityStepResult,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "raw_final_energy": _energy(model, raw_state),
        "feasibility_final_energy": _energy(model, repaired_result.state),
        "raw_violation_energy": raw_result.violation_energy,
        "feasibility_violation_energy": repaired_result.violation_energy,
        "raw_violation_count": raw_result.violation_count,
        "feasibility_violation_count": repaired_result.violation_count,
        "feasibility_convergence_steps": repaired_result.convergence_steps,
        "distortion_l2": repaired_result.distortion_l2,
        "feasibility_converged": repaired_result.converged,
    }


def build_artifact() -> dict[str, Any]:
    model = _build_reference_model()
    constraint_matrix, constraint_bias = _fover_like_constraints()
    raw_states: list[np.ndarray] = []
    repaired_states: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for seed in range(16):
        raw_state = sample_langevin(
            model,
            n_steps=160,
            lr=0.01,
            noise_scale=0.06,
            temp_schedule="cosine",
            seed=seed,
        )
        raw_result = feasibility_step(raw_state, constraint_matrix, constraint_bias, n_steps=0)
        repaired_result = feasibility_step(
            raw_state,
            constraint_matrix,
            constraint_bias,
            n_steps=48,
            lr=0.55,
            anchor_weight=0.02,
            tolerance=1e-8,
        )
        raw_states.append(raw_state)
        repaired_states.append(repaired_result.state)
        rows.append(_result_row(seed, model, raw_state, raw_result, repaired_result))

    raw_energy_mean = float(np.mean([row["raw_final_energy"] for row in rows]))
    repaired_energy_mean = float(np.mean([row["feasibility_final_energy"] for row in rows]))
    raw_violation_mean = float(np.mean([row["raw_violation_count"] for row in rows]))
    repaired_violation_mean = float(
        np.mean([row["feasibility_violation_count"] for row in rows])
    )
    raw_violation_energy_mean = float(np.mean([row["raw_violation_energy"] for row in rows]))
    repaired_violation_energy_mean = float(
        np.mean([row["feasibility_violation_energy"] for row in rows])
    )
    distortion_mean = float(np.mean([row["distortion_l2"] for row in rows]))
    convergence_steps_mean = float(
        np.mean([row["feasibility_convergence_steps"] for row in rows])
    )
    raw_diversity = _mean_pairwise_l2(raw_states)
    repaired_diversity = _mean_pairwise_l2(repaired_states)
    diversity_ratio = float(repaired_diversity / raw_diversity) if raw_diversity else 1.0

    energy_delta = raw_energy_mean - repaired_energy_mean
    violation_delta = raw_violation_mean - repaired_violation_mean
    violation_energy_delta = raw_violation_energy_mean - repaired_violation_energy_mean
    diversity_penalty = max(0.0, 0.85 - diversity_ratio)
    feasibility_delta_overall = float(
        violation_delta + violation_energy_delta - distortion_mean - diversity_penalty
    )
    feasibility_step_viable = (
        violation_delta > 0.0
        and violation_energy_delta > 0.0
        and diversity_ratio >= 0.85
        and distortion_mean <= 1.0
    )
    honest_verdict = (
        "feasibility_step_viable"
        if feasibility_step_viable
        else "feasibility_step_not_viable"
    )

    source_path = Path("results/experiment_1264_q11_tss_instrumentation_v2.json")
    source_context = json.loads(source_path.read_text()) if source_path.exists() else {}

    return {
        "schema": "carnot.phase3.fsnet_feasibility_step.v1",
        "experiment": "1275_fsnet_feasibility_step_continuous_ebm",
        "run_date": RUN_DATE,
        "status": "complete",
        "spec_refs": ["REQ-KONA-027", "SCENARIO-KONA-027"],
        "source_context": {
            "experiment_1264_honest_verdict": source_context.get("honest_verdict"),
            "tss_instrumented": source_context.get("tss_instrumented"),
        },
        "latent_dim": model.variables,
        "n_states": len(rows),
        "constraints": {
            "description": "FoVer-like verifier logits constrained by A @ z + b <= 0",
            "n_constraints": int(constraint_matrix.shape[0]),
        },
        "raw_langevin": {
            "final_energy_mean": raw_energy_mean,
            "violation_energy_mean": raw_violation_energy_mean,
            "violation_count_mean": raw_violation_mean,
            "convergence_steps_mean": 160.0,
            "distortion_mean": 0.0,
            "diversity_mean_pairwise_l2": raw_diversity,
        },
        "feasibility_step": {
            "final_energy_mean": repaired_energy_mean,
            "violation_energy_mean": repaired_violation_energy_mean,
            "violation_count_mean": repaired_violation_mean,
            "convergence_steps_mean": convergence_steps_mean,
            "distortion_mean": distortion_mean,
            "diversity_mean_pairwise_l2": repaired_diversity,
            "diversity_ratio_vs_raw": diversity_ratio,
        },
        "energy_delta": energy_delta,
        "violation_delta": violation_delta,
        "violation_energy_delta": violation_energy_delta,
        "distortion_mean": distortion_mean,
        "diversity_delta": float(repaired_diversity - raw_diversity),
        "feasibility_delta_overall": feasibility_delta_overall,
        "feasibility_step_viable": feasibility_step_viable,
        "honest_verdict": honest_verdict,
        "per_seed": rows,
    }


def main() -> None:
    artifact = build_artifact()
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
