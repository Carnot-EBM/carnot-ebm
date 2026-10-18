"""Exp 1276: SnareNet-style adaptive repair layer gated on Exp 1275.

Spec refs: REQ-KONA-028, SCENARIO-KONA-028.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import numpy as np

from carnot.phase3.continuous_ebm import (
    AdaptiveRepairLayer,
    AdaptiveRepairResult,
    ContinuousEBM,
    FeasibilityStepResult,
    feasibility_step,
    sample_langevin,
)

RUN_DATE = "20260504"
RESULT_PATH = Path("results/experiment_1276_snarenet_repair_layer_gated.json")
SOURCE_PATH = Path("results/experiment_1275_fsnet_feasibility_step_continuous_ebm.json")


def _energy(model: ContinuousEBM, state: np.ndarray) -> float:
    return float(-0.5 * state @ model.coupling @ state - model.bias @ state)


def _mean_pairwise_l2(states: list[np.ndarray]) -> float:
    distances: list[float] = []
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            distances.append(float(np.linalg.norm(states[left] - states[right])))
    return float(np.mean(distances)) if distances else 0.0


def _hard_constraint_satisfaction(
    state: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_bias: np.ndarray,
) -> float:
    scores = constraint_matrix @ state + constraint_bias
    return float(np.mean(scores <= 0.0))


def _hard_violation_count(
    state: np.ndarray,
    constraint_matrix: np.ndarray,
    constraint_bias: np.ndarray,
) -> int:
    scores = constraint_matrix @ state + constraint_bias
    return int(np.sum(scores > 0.0))


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


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([row[field] for row in rows]))


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
        "constraint_satisfaction_mean": _mean(rows, f"{prefix}_constraint_satisfaction"),
        "hard_constraint_satisfaction_mean": _mean(
            rows, f"{prefix}_hard_constraint_satisfaction"
        ),
        "distortion_from_initial_mean": _mean(rows, f"{prefix}_distortion_from_initial"),
        "repair_iterations_mean": _mean(rows, f"{prefix}_repair_iterations"),
        "diversity_mean_pairwise_l2": diversity,
        "diversity_ratio_vs_raw": diversity_ratio,
    }


def _result_row(
    seed: int,
    model: ContinuousEBM,
    raw_state: np.ndarray,
    raw_result: FeasibilityStepResult,
    adaptive_result: AdaptiveRepairResult,
    constraint_matrix: np.ndarray,
    constraint_bias: np.ndarray,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "raw_final_energy": _energy(model, raw_state),
        "fsnet_final_energy": _energy(model, adaptive_result.fsnet_state),
        "adaptive_final_energy": _energy(model, adaptive_result.state),
        "raw_violation_energy": raw_result.violation_energy,
        "fsnet_violation_energy": adaptive_result.fsnet_violation_energy,
        "adaptive_violation_energy": adaptive_result.violation_energy,
        "raw_violation_count": raw_result.violation_count,
        "fsnet_violation_count": _hard_violation_count(
            adaptive_result.fsnet_state, constraint_matrix, constraint_bias
        ),
        "adaptive_violation_count": adaptive_result.violation_count,
        "raw_constraint_satisfaction": adaptive_result.initial_constraint_satisfaction,
        "fsnet_constraint_satisfaction": adaptive_result.fsnet_constraint_satisfaction,
        "adaptive_constraint_satisfaction": adaptive_result.final_constraint_satisfaction,
        "raw_hard_constraint_satisfaction": _hard_constraint_satisfaction(
            raw_state, constraint_matrix, constraint_bias
        ),
        "fsnet_hard_constraint_satisfaction": _hard_constraint_satisfaction(
            adaptive_result.fsnet_state, constraint_matrix, constraint_bias
        ),
        "adaptive_hard_constraint_satisfaction": _hard_constraint_satisfaction(
            adaptive_result.state, constraint_matrix, constraint_bias
        ),
        "raw_distortion_from_initial": 0.0,
        "fsnet_distortion_from_initial": adaptive_result.fsnet_distortion_from_initial,
        "adaptive_distortion_from_initial": adaptive_result.distortion_from_initial,
        "raw_repair_iterations": 0,
        "fsnet_repair_iterations": 48,
        "adaptive_repair_iterations": adaptive_result.repair_iterations,
        "adaptive_distortion_from_fsnet": adaptive_result.distortion_from_fsnet,
        "adaptive_final_relaxation": adaptive_result.final_relaxation,
        "adaptive_converged": adaptive_result.converged,
    }


def _blocked_artifact(source_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "carnot.phase3.snarenet_repair_layer.v1",
        "experiment": "1276_snarenet_repair_layer_gated",
        "run_date": RUN_DATE,
        "status": "blocked",
        "spec_refs": ["REQ-KONA-028", "SCENARIO-KONA-028"],
        "source_context": source_context,
        "final_constraint_satisfaction": 0.0,
        "repair_iterations": 0.0,
        "distortion_from_initial": 0.0,
        "diversity_preserved": False,
        "repair_delta_over_fsnet": 0.0,
        "honest_verdict": "blocked_exp1275_not_positive",
        "per_seed": [],
    }


def build_artifact() -> dict[str, Any]:
    source = json.loads(SOURCE_PATH.read_text()) if SOURCE_PATH.exists() else {}
    feasibility_delta = float(source.get("feasibility_delta_overall", 0.0) or 0.0)
    source_context = {
        "experiment_1275_honest_verdict": source.get("honest_verdict"),
        "experiment_1275_feasibility_delta_overall": feasibility_delta,
        "experiment_1275_feasibility_step_viable": bool(
            source.get("feasibility_step_viable", False)
        ),
    }
    if feasibility_delta <= 0.0 or not source_context["experiment_1275_feasibility_step_viable"]:
        return _blocked_artifact(source_context)

    model = _build_reference_model()
    constraint_matrix, constraint_bias = _fover_like_constraints()
    repair_layer = AdaptiveRepairLayer(
        fsnet_steps=48,
        fsnet_lr=0.55,
        fsnet_anchor_weight=0.02,
        n_steps=16,
        lr=0.18,
        anchor_weight=0.02,
        initial_relaxation=0.12,
        min_relaxation=0.03,
        max_relaxation=0.50,
        relaxation_growth=1.40,
        relaxation_decay=0.75,
        tolerance=1e-10,
    )

    raw_states: list[np.ndarray] = []
    fsnet_states: list[np.ndarray] = []
    adaptive_states: list[np.ndarray] = []
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
        adaptive_result = repair_layer.repair(raw_state, constraint_matrix, constraint_bias)
        raw_states.append(raw_state)
        fsnet_states.append(adaptive_result.fsnet_state)
        adaptive_states.append(adaptive_result.state)
        rows.append(
            _result_row(
                seed,
                model,
                raw_state,
                raw_result,
                adaptive_result,
                constraint_matrix,
                constraint_bias,
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
    adaptive_summary = _arm_summary(
        rows,
        prefix="adaptive",
        states=adaptive_states,
        raw_diversity=raw_diversity,
    )

    repair_delta_over_fsnet = float(
        adaptive_summary["constraint_satisfaction_mean"]
        - fsnet_summary["constraint_satisfaction_mean"]
    )
    distortion_delta_over_fsnet = float(
        adaptive_summary["distortion_from_initial_mean"]
        - fsnet_summary["distortion_from_initial_mean"]
    )
    diversity_preserved = adaptive_summary["diversity_ratio_vs_raw"] >= 0.85
    distortion_not_excessive = (
        adaptive_summary["distortion_from_initial_mean"]
        <= fsnet_summary["distortion_from_initial_mean"] + 0.20
    )
    adaptive_improves = repair_delta_over_fsnet > 1e-6
    adaptive_matches = repair_delta_over_fsnet >= -1e-8

    if adaptive_improves and diversity_preserved and distortion_not_excessive:
        honest_verdict = "adaptive_repair_improves_fsnet"
    elif adaptive_matches and diversity_preserved and distortion_not_excessive:
        honest_verdict = "adaptive_repair_matches_fsnet"
    else:
        honest_verdict = "adaptive_repair_distorts_or_collapses"

    return {
        "schema": "carnot.phase3.snarenet_repair_layer.v1",
        "experiment": "1276_snarenet_repair_layer_gated",
        "run_date": RUN_DATE,
        "status": "complete",
        "spec_refs": ["REQ-KONA-028", "SCENARIO-KONA-028"],
        "source_context": source_context,
        "latent_dim": model.variables,
        "n_states": len(rows),
        "constraints": {
            "description": "FoVer-like verifier logits constrained by A @ z + b <= 0",
            "n_constraints": int(constraint_matrix.shape[0]),
        },
        "repair_layer": {
            "fsnet_steps": repair_layer.fsnet_steps,
            "adaptive_steps": repair_layer.n_steps,
            "initial_relaxation": repair_layer.initial_relaxation,
            "min_relaxation": repair_layer.min_relaxation,
            "max_relaxation": repair_layer.max_relaxation,
            "relaxation_growth": repair_layer.relaxation_growth,
            "relaxation_decay": repair_layer.relaxation_decay,
        },
        "arms": {
            "raw_langevin": raw_summary,
            "fsnet_feasibility_step": fsnet_summary,
            "adaptive_repair": adaptive_summary,
        },
        "final_constraint_satisfaction": adaptive_summary["constraint_satisfaction_mean"],
        "repair_iterations": adaptive_summary["repair_iterations_mean"],
        "distortion_from_initial": adaptive_summary["distortion_from_initial_mean"],
        "diversity_preserved": diversity_preserved,
        "repair_delta_over_fsnet": repair_delta_over_fsnet,
        "distortion_delta_over_fsnet": distortion_delta_over_fsnet,
        "honest_verdict": honest_verdict,
        "per_seed": rows,
    }


def main() -> dict[str, Any]:
    artifact = build_artifact()
    RESULT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


if __name__ == "__main__":
    main()
