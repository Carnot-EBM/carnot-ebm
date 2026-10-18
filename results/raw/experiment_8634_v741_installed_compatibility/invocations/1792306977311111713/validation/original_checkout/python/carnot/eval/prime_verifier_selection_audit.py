"""Exp 1272 PRIME-style verifier selection audit for GRPO.

Spec: REQ-VERIFY-1272, SCENARIO-VERIFY-1272
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FOVER_PATH = REPO_ROOT / "results" / "fover_corpus_v5.json"
DEFAULT_EXP1256_PATH = (
    REPO_ROOT / "results" / "experiment_1256_verifier_orthogonality_audit_v3.json"
)
DEFAULT_EXP1271_PATH = (
    REPO_ROOT / "results" / "experiment_1271_triggered_certificate_extraction_sota_gguf.json"
)
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "results" / "experiment_1272_prime_verifier_selection_audit.json"
)
EXPERIMENT_NAME = "1272_prime_verifier_selection_audit"
SCHEMA = "prime_verifier_selection_audit_v1"
RUN_DATE = "20260504"

REQUIRED_ARTIFACT_FIELDS: set[str] = {
    "experiment",
    "schema",
    "run_date",
    "status",
    "n_fover_pairs",
    "n_process_error_rows",
    "n_outcome_correct",
    "n_outcome_incorrect",
    "source_artifacts",
    "per_verifier_metrics",
    "verifier_weight_vector_written",
    "verifier_weight_vector",
    "missing_fields",
    "missing_optional_fields",
    "honest_verdict",
}

PROCESS_ERROR_LABELS = {
    "incorrect",
    "wrong",
    "violation",
    "violated",
    "unsat",
    "false",
    "causal_break",
    "arithmetic",
}
DETECTION_THRESHOLDS = {
    "Z3MathVerifier": 0.5,
    "SemEnergyProbe": 0.5,
    "SOSKANEnergyV3": 0.5,
    "k5_ensemble_summary": 0.5,
    "CausalReasoningVerifier": 0.0,
    "SymCodeVerifier": 0.0,
}


@dataclass(frozen=True)
class FoVerAuditRow:
    """One FoVer response with outcome and process labels for audit scoring."""

    question: str
    response: str
    is_correct: bool
    fover_labels: tuple[str, ...] = field(default_factory=tuple)
    cot_steps: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def outcome_error(self) -> bool:
        """Return True when the final answer label says the response is wrong."""

        return not self.is_correct

    @property
    def process_error(self) -> bool:
        """Return True when FoVer step labels expose a reasoning-process error."""

        labels = {str(label).strip().lower() for label in self.fover_labels}
        for step in self.cot_steps:
            for key in ("z3_label", "label", "verdict", "violation_type"):
                value = step.get(key)
                if value is not None:
                    labels.add(str(value).strip().lower())
        return bool(labels & PROCESS_ERROR_LABELS)


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path | str | None) -> Any | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return _read_json(candidate)


def rows_from_payload(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[FoVerAuditRow]:
    """Build audit rows from either the FoVer corpus object or a raw pair list."""

    raw_pairs: Sequence[Mapping[str, Any]]
    if isinstance(payload, Mapping):
        raw_pairs = payload.get("pairs", [])  # type: ignore[assignment]
    else:
        raw_pairs = payload

    rows: list[FoVerAuditRow] = []
    for pair in raw_pairs:
        labels = tuple(str(label) for label in pair.get("fover_labels", []) or [])
        steps = tuple(dict(step) for step in pair.get("cot_steps", []) or [])
        rows.append(
            FoVerAuditRow(
                question=str(pair.get("question", "")),
                response=str(pair.get("response", "")),
                is_correct=bool(pair.get("is_correct", True)),
                fover_labels=labels,
                cot_steps=steps,
            )
        )
    return rows


def load_fover_rows(path: Path | str = DEFAULT_FOVER_PATH) -> list[FoVerAuditRow]:
    """Load FoVer corpus rows with labels needed by REQ-VERIFY-1272."""

    return rows_from_payload(_read_json(path))


def _score_to_detection(verifier_name: str, score: float) -> bool:
    threshold = DETECTION_THRESHOLDS.get(verifier_name, 0.5)
    return float(score) > threshold


def _detection_vector(verifier_name: str, scores: Sequence[float]) -> list[bool]:
    return [_score_to_detection(verifier_name, float(score)) for score in scores]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _pearson_binary(left: Sequence[bool], right: Sequence[bool]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    xs = [1.0 if item else 0.0 for item in left]
    ys = [1.0 if item else 0.0 for item in right]
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return float(num / (den_x * den_y))


def _matrix_penalty(name: str, exp1256_payload: Mapping[str, Any]) -> float | None:
    if name == "k5_ensemble_summary":
        value = exp1256_payload.get("max_pairwise_r_k5", exp1256_payload.get("max_pairwise_r"))
        return None if value is None else max(0.0, min(1.0, abs(float(value))))

    matrix = exp1256_payload.get("pairwise_r_matrix")
    if not isinstance(matrix, Mapping):
        return None

    penalties: list[float] = []
    for key, value in matrix.items():
        left, sep, right = str(key).partition("|")
        if sep and name in (left, right) and left != right:
            penalties.append(max(0.0, min(1.0, abs(float(value)))))
    if not penalties:
        return None
    return max(penalties)


def _pairwise_penalty(
    name: str,
    detections_by_name: Mapping[str, Sequence[bool]],
    exp1256_payload: Mapping[str, Any],
) -> float:
    matrix_value = _matrix_penalty(name, exp1256_payload)
    if matrix_value is not None:
        return matrix_value

    current = detections_by_name[name]
    correlations = [
        abs(_pearson_binary(current, other))
        for other_name, other in detections_by_name.items()
        if other_name != name
    ]
    if not correlations:
        return 0.0
    return float(max(0.0, min(1.0, max(correlations))))


def compute_prime_metrics(
    rows: Sequence[FoVerAuditRow],
    verifier_signals: Mapping[str, Sequence[float]],
    *,
    exp1256_payload: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute REQ-VERIFY-1272 PRIME process/outcome metrics per verifier."""

    exp1256_payload = exp1256_payload or {}
    process_rows = [index for index, row in enumerate(rows) if row.process_error]
    outcome_errors = [row.outcome_error for row in rows]
    target_errors = [row.outcome_error or row.process_error for row in rows]

    detections_by_name = {
        name: _detection_vector(name, list(scores)[: len(rows)])
        for name, scores in verifier_signals.items()
        if len(scores) >= len(rows)
    }

    metrics: dict[str, dict[str, float]] = {}
    for name, detections in detections_by_name.items():
        process_rate = _mean([1.0 if detections[index] else 0.0 for index in process_rows])
        final_agreement = _mean(
            [
                1.0 if detected == outcome_error else 0.0
                for detected, outcome_error in zip(detections, outcome_errors, strict=True)
            ]
        )
        consistency = _mean(
            [
                1.0 if detected == target_error else 0.0
                for detected, target_error in zip(detections, target_errors, strict=True)
            ]
        )
        penalty = _pairwise_penalty(name, detections_by_name, exp1256_payload)
        alignment = 0.45 * process_rate + 0.35 * final_agreement + 0.20 * consistency
        raw_weight_score = max(0.0, alignment * (1.0 - penalty))
        metrics[name] = {
            "process_error_detection_rate": round(process_rate, 6),
            "final_answer_agreement": round(final_agreement, 6),
            "answer_reasoning_consistency": round(consistency, 6),
            "pairwise_correlation_penalty": round(penalty, 6),
            "raw_weight_score": round(raw_weight_score, 12),
        }
    return metrics


def normalize_weight_vector(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Normalize non-negative raw PRIME scores into GRPO verifier weights."""

    raw = {name: max(0.0, float(values.get("raw_weight_score", 0.0))) for name, values in metrics.items()}
    total = sum(raw.values())
    if total <= 0.0:
        return {}
    return {name: value / total for name, value in raw.items()}


def _has_exp1271_certificate_outputs(payload: Mapping[str, Any] | None) -> bool:
    if not payload or payload.get("status") == "blocked":
        return False
    for key in (
        "certificates",
        "certificate_outputs",
        "verification_certificates",
        "per_step_certificates",
    ):
        value = payload.get(key)
        if value:
            return True
    return False


def _missing_required_data(
    rows: Sequence[FoVerAuditRow],
    verifier_signals: Mapping[str, Sequence[float]],
    exp1256_payload: Mapping[str, Any],
    metrics: Mapping[str, Mapping[str, float]],
) -> list[str]:
    missing: list[str] = []
    if not rows:
        missing.append("fover_pairs")
    if not any(row.process_error for row in rows):
        missing.append("process_error_labels")
    if len({row.outcome_error for row in rows}) < 2:
        missing.append("outcome_label_classes")
    if len(verifier_signals) < 2:
        missing.append("verifier_signals")
    if not exp1256_payload.get("pairwise_r_matrix"):
        missing.append("exp1256_pairwise_r_matrix")
    if not normalize_weight_vector(metrics):
        missing.append("positive_weight_scores")
    return missing


def build_audit_artifact(
    rows: Sequence[FoVerAuditRow],
    *,
    verifier_signals: Mapping[str, Sequence[float]],
    exp1256_payload: Mapping[str, Any] | None,
    exp1271_payload: Mapping[str, Any] | None,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the Exp 1272 artifact from FoVer labels and verifier signals."""

    exp1256_payload = exp1256_payload or {}
    metrics = compute_prime_metrics(
        rows,
        verifier_signals,
        exp1256_payload=exp1256_payload,
    )
    missing_fields = _missing_required_data(rows, verifier_signals, exp1256_payload, metrics)
    missing_optional_fields: list[str] = []
    if not _has_exp1271_certificate_outputs(exp1271_payload):
        missing_optional_fields.append("exp1271_certificate_outputs")

    weights = {} if missing_fields else normalize_weight_vector(metrics)
    vector_written = bool(weights) and not missing_fields
    top_name = max(weights, key=weights.get) if weights else None

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "complete" if vector_written else "blocked",
        "n_fover_pairs": len(rows),
        "n_process_error_rows": sum(1 for row in rows if row.process_error),
        "n_outcome_correct": sum(1 for row in rows if not row.outcome_error),
        "n_outcome_incorrect": sum(1 for row in rows if row.outcome_error),
        "source_artifacts": {
            "fover_corpus": str(DEFAULT_FOVER_PATH.relative_to(REPO_ROOT)),
            "exp1256": str(DEFAULT_EXP1256_PATH.relative_to(REPO_ROOT)),
            "exp1271": str(DEFAULT_EXP1271_PATH.relative_to(REPO_ROOT)),
        },
        "exp1256_k_eff": exp1256_payload.get("k_eff"),
        "exp1256_max_pairwise_r_k5": exp1256_payload.get("max_pairwise_r_k5"),
        "per_verifier_metrics": metrics,
        "metric_definitions": {
            "process_error_detection_rate": "fraction of FoVer process-error rows detected",
            "final_answer_agreement": "fraction where verifier error signal matches final answer label",
            "answer_reasoning_consistency": "fraction where verifier signal matches process-or-outcome error",
            "pairwise_correlation_penalty": "max absolute verifier correlation penalty in [0,1]",
        },
        "verifier_weight_vector_written": vector_written,
        "verifier_weight_vector": weights,
        "missing_fields": missing_fields,
        "missing_optional_fields": missing_optional_fields,
        "honest_verdict": (
            f"prime_verifier_weights_selected_top_{top_name}"
            if vector_written and top_name is not None
            else "prime_verifier_weights_blocked_missing_data"
        ),
    }
    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:  # pragma: no cover - guards future schema edits.
        raise ValueError(f"missing required artifact fields: {sorted(missing)}")
    return artifact


def evaluate_default_verifier_signals(rows: Sequence[FoVerAuditRow]) -> dict[str, list[float]]:
    """Evaluate lightweight verifier signals over FoVer rows for the live audit."""

    signals: dict[str, list[float]] = {}
    texts = [f"{row.question}\n{row.response}" for row in rows]

    try:
        from carnot.verify.z3_math_verifier import Z3MathVerifier

        verifier = Z3MathVerifier()
        signals["Z3MathVerifier"] = [float(verifier.score(text)) for text in texts]
    except Exception:
        pass

    try:
        from carnot.pipeline.symcode_verifier import SymCodeVerifier

        verifier = SymCodeVerifier()
        signals["SymCodeVerifier"] = [float(verifier.detection_score(row.response)) for row in rows]
    except Exception:
        pass

    try:
        from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier

        verifier = CausalReasoningVerifier()
        signals["CausalReasoningVerifier"] = [
            float(verifier.detection_score(row.response)) for row in rows
        ]
    except Exception:
        pass

    try:
        from carnot.verify.and_composition_verifier import SemEnergyProbeAdapter

        verifier = SemEnergyProbeAdapter()
        signals["SemEnergyProbe"] = [float(verifier.score(text)) for text in texts]
    except Exception:
        pass

    try:
        from carnot.verify.and_composition_verifier import SOSKANEnergyV3Adapter

        verifier = SOSKANEnergyV3Adapter()
        signals["SOSKANEnergyV3"] = [float(verifier.score(text)) for text in texts]
    except Exception:
        pass

    component_names = [
        name for name in ("SOSKANEnergyV3", "SemEnergyProbe", "Z3MathVerifier") if name in signals
    ]
    if component_names:
        ensemble_scores: list[float] = []
        for index in range(len(rows)):
            detected = any(
                _score_to_detection(name, signals[name][index]) for name in component_names
            )
            ensemble_scores.append(1.0 if detected else 0.0)
        signals["k5_ensemble_summary"] = ensemble_scores

    return signals


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_RESULT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Write the required in-progress Exp 1272 artifact."""

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "schema": SCHEMA,
        "run_date": run_date,
        "status": "in_progress",
        "verifier_weight_vector_written": False,
        "verifier_weight_vector": {},
        "honest_verdict": "in_progress",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def run_experiment(
    *,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    exp1256_path: Path | str = DEFAULT_EXP1256_PATH,
    exp1271_path: Path | str | None = DEFAULT_EXP1271_PATH,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    verifier_signals: Mapping[str, Sequence[float]] | None = None,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run Exp 1272 and persist the PRIME verifier selection audit."""

    write_in_progress_artifact(output_path, run_date=run_date)
    rows = load_fover_rows(fover_path)
    exp1256_payload = _read_json_if_exists(exp1256_path) or {}
    exp1271_payload = _read_json_if_exists(exp1271_path)
    signals = dict(verifier_signals) if verifier_signals is not None else evaluate_default_verifier_signals(rows)
    artifact = build_audit_artifact(
        rows,
        verifier_signals=signals,
        exp1256_payload=exp1256_payload,
        exp1271_payload=exp1271_payload,
        run_date=run_date,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
