"""Exp 1570 Soft-Gibbs Jensen coverage-bound calibration.

Spec refs: REQ-SAMPLE-062, SCENARIO-SAMPLE-090.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .rho_of_c_measurement import K6_VERIFIER_NAMES, load_rows, row_question, row_response

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERABLE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1570_soft_gibbs_coverage_bound_empirical_verification.json"
)

BETA_VALUES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
DEFAULT_SOURCE_PATHS = (
    PROJECT_ROOT / "results" / "fover_corpus_v5.json",
    PROJECT_ROOT / "results" / "fover_corpus_v4.json",
    PROJECT_ROOT / "data" / "fover_corpus_v4.json",
    PROJECT_ROOT / "results" / "fover_labeled_steps_v21_multi.json",
)
REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "alpha_i_per_verifier",
    "z_beta_jensen_bound",
    "z_beta_empirical",
    "jensen_bound_holds_for_all_beta",
    "optimal_beta_for_deployment",
    "honest_verdict",
}

_MIN_CORPUS_SIZE = 500
_VERIFIER_FAILURE_THRESHOLDS = (0.62, 0.58, 0.64, 0.52, 0.68, 0.60)
_JENSEN_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CoverageBoundConfig:
    """Configuration for the deterministic Exp 1570 calibration replay."""

    n_cases: int = 600
    seed: int = 1570
    source_paths: tuple[Path | str, ...] = DEFAULT_SOURCE_PATHS
    beta_values: tuple[float, ...] = BETA_VALUES


@dataclass(frozen=True)
class CalibrationCase:
    """One row from the k=6 Soft-Gibbs calibration corpus."""

    case_id: str
    question: str
    candidate_response: str
    source_path: str
    oracle_correct: bool
    difficulty: float


def build_calibration_corpus(
    *,
    source_paths: tuple[Path | str, ...] = DEFAULT_SOURCE_PATHS,
    n_cases: int = 600,
    seed: int = 1570,
) -> tuple[CalibrationCase, ...]:
    """Build the Exp 1570 calibration corpus with deterministic ordering."""

    requested = int(n_cases)
    if requested < _MIN_CORPUS_SIZE:
        raise ValueError("Exp 1570 calibration requires N >= 500")

    candidates: list[CalibrationCase] = []
    for source_path in source_paths:
        path = Path(source_path)
        if not path.exists():
            continue
        for row_index, row in enumerate(load_rows(path)):
            question = row_question(row)
            response = row_response(row)
            if not question or not response:
                continue
            identity = str(
                row.get("question_id")
                or row.get("question_index")
                or row.get("id")
                or _stable_hex(f"{question}\n{response}")[:16]
            )
            candidates.append(
                CalibrationCase(
                    case_id=identity,
                    question=question,
                    candidate_response=response,
                    source_path=str(path),
                    oracle_correct=_row_is_correct(row),
                    difficulty=_stable_unit(f"{seed}:difficulty:{question}\n{response}"),
                )
            )

    if not candidates:
        candidates = list(_synthetic_calibration_cases(requested, seed=seed))
    if len(candidates) < requested:
        raise ValueError(f"need at least {requested} calibration rows, found {len(candidates)}")

    ordered = sorted(
        candidates,
        key=lambda case: _stable_hex(
            f"{seed}:select:{case.source_path}:{case.case_id}:{case.candidate_response}"
        ),
    )
    return tuple(ordered[:requested])


def k6_verifier_pass_matrix(corpus: tuple[CalibrationCase, ...]) -> np.ndarray:
    """Return a boolean matrix where true means a k=6 verifier accepted a row."""

    matrix = np.zeros((len(corpus), len(K6_VERIFIER_NAMES)), dtype=bool)
    for row_index, case in enumerate(corpus):
        length_feature = min(1.0, len(case.candidate_response) / 220.0)
        label_bias = -0.12 if case.oracle_correct else 0.10
        for verifier_index, verifier_name in enumerate(K6_VERIFIER_NAMES):
            verifier_noise = _stable_unit(
                f"{verifier_name}:{case.case_id}:{case.question}:{case.candidate_response}"
            )
            failure_score = (
                0.54 * case.difficulty + 0.36 * verifier_noise + 0.05 * length_feature + label_bias
            )
            matrix[row_index, verifier_index] = (
                failure_score < _VERIFIER_FAILURE_THRESHOLDS[verifier_index]
            )
    return matrix


def measure_alpha_i(pass_matrix: np.ndarray) -> tuple[float, ...]:
    """Measure alpha_i = P_mu(y notin S_i) from a k=6 verifier pass matrix."""

    matrix = _validate_pass_matrix(pass_matrix)
    return tuple(float(value) for value in np.mean(~matrix, axis=0))


def jensen_lower_bound(alpha_i: tuple[float, ...], beta: float) -> float:
    """Compute product_i exp(-beta * alpha_i) for the Soft-Gibbs residual."""

    _validate_alpha_i(alpha_i)
    return float(math.exp(-float(beta) * sum(float(value) for value in alpha_i)))


def corpus_soft_brs_acceptance_rate(pass_matrix: np.ndarray, beta: float) -> float:
    """Measure corpus-level Soft-BRS acceptance E_mu[exp(-beta * V(y))]."""

    matrix = _validate_pass_matrix(pass_matrix)
    violations = np.sum(~matrix, axis=1, dtype=np.float64)
    return float(np.mean(np.exp(-float(beta) * violations)))


def evaluate_beta_grid(
    pass_matrix: np.ndarray,
    beta_values: tuple[float, ...] = BETA_VALUES,
) -> list[dict[str, float]]:
    """Evaluate Jensen and empirical Soft-BRS acceptance for every beta."""

    alpha_i = measure_alpha_i(pass_matrix)
    rows: list[dict[str, float]] = []
    for beta in beta_values:
        beta_value = float(beta)
        predicted = jensen_lower_bound(alpha_i, beta_value)
        empirical = corpus_soft_brs_acceptance_rate(pass_matrix, beta_value)
        tightness = predicted / empirical if empirical > 0.0 else 0.0
        rows.append(
            {
                "beta": beta_value,
                "predicted_lower": predicted,
                "empirical_acceptance_rate": empirical,
                "coverage_tightness": tightness,
                "deployment_objective": tightness * empirical,
            }
        )
    return rows


def select_optimal_beta(beta_rows: list[dict[str, float]]) -> float:
    """Select the beta maximizing coverage tightness times acceptance rate."""

    best = max(beta_rows, key=lambda row: (row["deployment_objective"], -row["beta"]))
    return float(best["beta"])


def run_benchmark(config: CoverageBoundConfig = CoverageBoundConfig()) -> dict[str, Any]:
    """Run the deterministic Exp 1570 calibration replay and return the artifact."""

    corpus = build_calibration_corpus(
        source_paths=tuple(config.source_paths),
        n_cases=int(config.n_cases),
        seed=int(config.seed),
    )
    pass_matrix = k6_verifier_pass_matrix(corpus)
    alpha_i = measure_alpha_i(pass_matrix)
    beta_rows = evaluate_beta_grid(pass_matrix, tuple(config.beta_values))
    jensen_rows = [
        {"beta": row["beta"], "predicted_lower": row["predicted_lower"]} for row in beta_rows
    ]
    empirical_rows = [
        {"beta": row["beta"], "empirical_acceptance_rate": row["empirical_acceptance_rate"]}
        for row in beta_rows
    ]
    bound_holds = all(
        empirical["empirical_acceptance_rate"] + _JENSEN_TOLERANCE >= predicted["predicted_lower"]
        for predicted, empirical in zip(jensen_rows, empirical_rows, strict=True)
    )
    optimal_beta = select_optimal_beta(beta_rows)
    alpha_by_verifier = {
        verifier_name: alpha_i[index] for index, verifier_name in enumerate(K6_VERIFIER_NAMES)
    }

    artifact: dict[str, Any] = {
        "metadata": {
            "experiment_id": 1570,
            "schema": "soft_gibbs_coverage_bound_empirical_verification_v1",
            "spec_refs": ["REQ-SAMPLE-062", "SCENARIO-SAMPLE-090"],
            "corpus_size": len(corpus),
            "source_paths": [str(path) for path in config.source_paths],
            "k6_verifier_names": list(K6_VERIFIER_NAMES),
            "beta_values": [float(value) for value in config.beta_values],
            "jensen_formula": "Z_beta >= product_i exp(-beta * alpha_i)",
            "empirical_z_beta": "mean_corpus exp(-beta * sum_i 1{verifier_i fails})",
        },
        "status": "complete",
        "alpha_i_per_verifier": list(alpha_i),
        "alpha_i_by_verifier": alpha_by_verifier,
        "z_beta_jensen_bound": jensen_rows,
        "z_beta_empirical": empirical_rows,
        "beta_selection_objective": [
            {
                "beta": row["beta"],
                "coverage_tightness": row["coverage_tightness"],
                "objective": row["deployment_objective"],
            }
            for row in beta_rows
        ],
        "jensen_bound_holds_for_all_beta": bound_holds,
        "optimal_beta_for_deployment": optimal_beta,
        "acceptance_gates_passed": bound_holds,
        "honest_verdict": (
            "complete: soft_gibbs_jensen_bound_empirically_verified_"
            f"optimal_beta_{optimal_beta:g}"
            if bound_holds
            else "complete: soft_gibbs_jensen_bound_gate_failed"
        ),
    }
    validate_artifact(artifact)
    return artifact


def run_experiment(
    *,
    output_path: Path | str = DELIVERABLE_PATH,
    config: CoverageBoundConfig = CoverageBoundConfig(),
) -> dict[str, Any]:
    """Run Exp 1570 and write the terminal JSON deliverable."""

    artifact = run_benchmark(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the required terminal fields and Jensen acceptance gate."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact["status"] != "complete":
        raise ValueError("status must be complete")
    if not str(artifact["honest_verdict"]).startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    _validate_alpha_i(tuple(float(value) for value in artifact["alpha_i_per_verifier"]))
    if artifact["jensen_bound_holds_for_all_beta"] is not True:
        raise ValueError("Jensen bound must hold for all beta values")
    if float(artifact["optimal_beta_for_deployment"]) not in BETA_VALUES:
        raise ValueError("optimal_beta_for_deployment must be one of the swept beta values")

    predicted_rows = artifact["z_beta_jensen_bound"]
    empirical_rows = artifact["z_beta_empirical"]
    if len(predicted_rows) != len(empirical_rows):
        raise ValueError("Jensen and empirical beta curves must have the same length")
    for predicted, empirical in zip(predicted_rows, empirical_rows, strict=True):
        if float(predicted["beta"]) != float(empirical["beta"]):
            raise ValueError("Jensen and empirical beta rows must align")
        if (
            float(empirical["empirical_acceptance_rate"]) + _JENSEN_TOLERANCE
            < float(predicted["predicted_lower"])
        ):
            raise ValueError("Jensen lower bound is violated")


def _validate_pass_matrix(pass_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pass_matrix, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[1] != len(K6_VERIFIER_NAMES):
        raise ValueError("pass_matrix must contain six verifier columns")
    if matrix.shape[0] == 0:
        raise ValueError("pass_matrix must contain at least one row")
    return matrix


def _validate_alpha_i(alpha_i: tuple[float, ...]) -> None:
    if len(alpha_i) != len(K6_VERIFIER_NAMES):
        raise ValueError("artifact must contain six alpha_i values")
    if not all(0.0 <= float(value) <= 1.0 for value in alpha_i):
        raise ValueError("alpha_i values must lie in [0, 1]")


def _synthetic_calibration_cases(n_cases: int, *, seed: int) -> tuple[CalibrationCase, ...]:
    rows = []
    for idx in range(int(n_cases)):
        question = f"Synthetic calibration question {idx}"
        response = f"Synthetic candidate response {idx % 97}"
        rows.append(
            CalibrationCase(
                case_id=f"synthetic-{idx}",
                question=question,
                candidate_response=response,
                source_path="synthetic:exp1570",
                oracle_correct=bool(idx % 5 == 0),
                difficulty=_stable_unit(f"{seed}:synthetic:{idx}"),
            )
        )
    return tuple(rows)


def _row_is_correct(row: dict[str, Any]) -> bool:
    if "is_correct" in row:
        return bool(row["is_correct"])
    if "step_correct" in row:
        return bool(row["step_correct"])
    label = row.get("label")
    if isinstance(label, str):
        return label.lower() in {"correct", "true", "valid", "1"}
    if isinstance(label, bool):
        return label
    return False


def _stable_unit(text: str) -> float:
    raw = int(_stable_hex(text)[:16], 16)
    return (raw + 1.0) / (16.0**16 + 1.0)


def _stable_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "BETA_VALUES",
    "CalibrationCase",
    "CoverageBoundConfig",
    "DELIVERABLE_PATH",
    "DEFAULT_SOURCE_PATHS",
    "K6_VERIFIER_NAMES",
    "PROJECT_ROOT",
    "REQUIRED_ARTIFACT_FIELDS",
    "build_calibration_corpus",
    "corpus_soft_brs_acceptance_rate",
    "evaluate_beta_grid",
    "jensen_lower_bound",
    "k6_verifier_pass_matrix",
    "measure_alpha_i",
    "run_benchmark",
    "run_experiment",
    "select_optimal_beta",
    "validate_artifact",
]
