"""Exp 1157 SECL-style cheap-tier discriminative calibration.

Spec: REQ-VERIFY-1157, SCENARIO-VERIFY-1157.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from carnot.eval.goodfire_cheap_tier_distillation import (
    DEFAULT_EXEMPLAR_PATH,
    DEFAULT_FOVER_PATH,
    DEFAULT_SEMENERGY_THRESHOLD,
    DEFAULT_THINKPRM_THRESHOLD,
    CheapTierScore,
    load_json_or_jsonl,
    score_exemplar_rows,
)
from carnot.eval.goodfire_cheap_tier_distillation import (
    _select_correct_fover_rows as select_correct_fover_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP1145_PATH = (
    REPO_ROOT / "results" / "experiment_1145_goodfire_cheap_tier_distillation.json"
)
DEFAULT_RESULT_PATH = REPO_ROOT / "results" / "experiment_1157_secl_cheap_tier_calibration.json"

FPR_BUDGET = 0.30
TP_TARGET = 0.80
CLASS_WEIGHT_CORRECT = 1.0
CLASS_WEIGHT_FAILURE = FPR_BUDGET / TP_TARGET
DEFAULT_FOVER_CORRECT_N = 200

REQUIRED_ARTIFACT_FIELDS = [
    "n_exemplars",
    "n_correct_examples",
    "thinkprm_fpr_exp1145",
    "thinkprm_tp_exp1145",
    "secl_tp_rate",
    "secl_fpr",
    "precision_recall_improved",
    "discriminative_signal_used",
    "cheap_tier_fpr_below_30pct",
    "cheap_tier_tp_above_80pct",
    "honest_verdict",
]
ALLOWED_HONEST_VERDICTS = {
    "calibrated_tp_above_80_fpr_below_30",
    "tp_improved_fpr_reduced_not_gate",
    "trade_off_tp_dropped",
    "honest_negative_no_improvement",
}


@dataclass
class SECLCalibrationExample:
    """One labeled row for the two-feature SECL calibration probe."""

    id: str
    label: int
    cheap_tier_score: float
    discriminative_signal: float
    source: str
    category: str = "unknown"

    @property
    def features(self) -> np.ndarray:
        return np.array([self.cheap_tier_score, self.discriminative_signal], dtype=np.float64)


@dataclass
class SECLCalibrationProbe:
    """Deterministic logistic probe over cheap-tier score and P(True)."""

    coefficients: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    operating_threshold: float
    class_weights: dict[int, float]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        standardized = (matrix - self.feature_mean) / self.feature_scale
        logits = standardized @ self.coefficients + self.intercept
        return _sigmoid_array(logits)

    def predict_flags(self, features: np.ndarray, threshold: float | None = None) -> np.ndarray:
        cutoff = self.operating_threshold if threshold is None else float(threshold)
        return self.predict_proba(features) >= cutoff

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": [round(float(value), 8) for value in self.coefficients],
            "intercept": round(float(self.intercept), 8),
            "feature_mean": [round(float(value), 8) for value in self.feature_mean],
            "feature_scale": [round(float(value), 8) for value in self.feature_scale],
            "operating_threshold": round(float(self.operating_threshold), 8),
            "class_weights": {str(key): float(value) for key, value in self.class_weights.items()},
        }


def cheap_tier_score(
    thinkprm_score: float,
    semenergy_score: float,
    *,
    thinkprm_threshold: float = DEFAULT_THINKPRM_THRESHOLD,
    semenergy_threshold: float = DEFAULT_SEMENERGY_THRESHOLD,
) -> float:
    """Return a continuous cheap-tier OR-risk score from ThinkPRM and SemEnergy."""

    think_margin = float(thinkprm_score) - float(thinkprm_threshold)
    sem_margin = float(semenergy_score) - float(semenergy_threshold)
    return float(think_margin + max(0.0, sem_margin))


def discriminative_signal(
    thinkprm_score: float,
    *,
    thinkprm_true_logprob: float | None = None,
) -> float:
    """Approximate P(True | "Is this response correct?") without a live LLM."""

    if thinkprm_true_logprob is not None:
        return _clamp_probability(math.exp(float(thinkprm_true_logprob)))
    return _clamp_probability(float(_sigmoid_array(np.array([float(thinkprm_score)]))[0]))


def build_calibration_examples(
    goodfire_scores: Sequence[CheapTierScore],
    correct_scores: Sequence[CheapTierScore],
) -> list[SECLCalibrationExample]:
    """Build labeled SECL rows from Goodfire failures and FoVer correct examples."""

    examples: list[SECLCalibrationExample] = []
    for row in goodfire_scores:
        examples.append(_example_from_score(row, label=1, source="goodfire"))
    for row in correct_scores:
        examples.append(_example_from_score(row, label=0, source="fover_correct"))
    return examples


def train_secl_probe(
    examples: Sequence[SECLCalibrationExample],
    *,
    learning_rate: float = 0.2,
    n_steps: int = 20_000,
    l2: float = 1e-4,
) -> SECLCalibrationProbe:
    """Fit the REQ-VERIFY-1157 weighted logistic calibration probe."""

    features, labels = _feature_label_arrays(examples)
    feature_mean = features.mean(axis=0)
    feature_scale = np.where(features.std(axis=0) == 0.0, 1.0, features.std(axis=0))
    matrix = (features - feature_mean) / feature_scale
    weights = _class_weight_array(labels)
    weights = weights / weights.mean()

    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    intercept = 0.0
    for _ in range(int(n_steps)):
        logits = matrix @ coefficients + intercept
        probabilities = _sigmoid_array(logits)
        residual = weights * (probabilities - labels)
        intercept_gradient = float(residual.mean())
        coefficient_gradient = matrix.T @ residual / len(labels)
        coefficient_gradient += float(l2) * coefficients
        intercept -= float(learning_rate) * intercept_gradient
        coefficients -= float(learning_rate) * coefficient_gradient

    probabilities = _sigmoid_array(matrix @ coefficients + intercept)
    operating_threshold = choose_operating_threshold(
        probabilities,
        labels,
        fpr_budget=FPR_BUDGET,
    )
    return SECLCalibrationProbe(
        coefficients=coefficients,
        intercept=float(intercept),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        operating_threshold=float(operating_threshold),
        class_weights={0: CLASS_WEIGHT_CORRECT, 1: CLASS_WEIGHT_FAILURE},
    )


def choose_operating_threshold(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    fpr_budget: float,
) -> float:
    """Choose the threshold with maximal TP rate while keeping FPR within budget."""

    probs = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    candidates = [
        float(probs.max() + 1e-12),
        *sorted({float(value) for value in probs}, reverse=True),
    ]
    best_threshold = candidates[0]
    best_tp = -1.0
    best_fpr = math.inf
    for threshold in candidates:
        flags = probs >= threshold
        tp_rate = _rate(flags[y == 1])
        fpr = _rate(flags[y == 0])
        if fpr <= fpr_budget and (tp_rate > best_tp or (tp_rate == best_tp and fpr < best_fpr)):
            best_threshold = float(threshold)
            best_tp = tp_rate
            best_fpr = fpr
    return float(best_threshold)


def evaluate_secl_probe(
    examples: Sequence[SECLCalibrationExample],
    probe: SECLCalibrationProbe,
    *,
    operating_threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate the SECL probe on the labeled calibration corpus."""

    features, labels = _feature_label_arrays(examples)
    threshold = (
        probe.operating_threshold if operating_threshold is None else float(operating_threshold)
    )
    probabilities = probe.predict_proba(features)
    flags = probabilities >= threshold
    secl_tp_rate = round(_rate(flags[labels == 1]), 6)
    secl_fpr = round(_rate(flags[labels == 0]), 6)
    return {
        "secl_tp_rate": secl_tp_rate,
        "secl_fpr": secl_fpr,
        "precision_recall_improved": bool(secl_tp_rate >= TP_TARGET and secl_fpr <= FPR_BUDGET),
        "discriminative_signal_used": True,
        "cheap_tier_fpr_below_30pct": bool(secl_fpr <= FPR_BUDGET),
        "cheap_tier_tp_above_80pct": bool(secl_tp_rate >= TP_TARGET),
        "operating_threshold": round(float(threshold), 8),
        "n_flagged_failures": int(flags[labels == 1].sum()),
        "n_flagged_correct": int(flags[labels == 0].sum()),
    }


def build_exp1157_artifact(
    *,
    examples: Sequence[SECLCalibrationExample],
    metrics: dict[str, Any],
    exp1145_artifact: dict[str, Any],
    duration_s: float,
    probe: SECLCalibrationProbe | None = None,
) -> dict[str, Any]:
    """Build the stable Exp 1157 result artifact."""

    n_exemplars = sum(example.label == 1 for example in examples)
    n_correct = sum(example.label == 0 for example in examples)
    exp1145_tp = round(float(exp1145_artifact["combined_cheap_tp_after"]), 3)
    exp1145_fpr = round(float(exp1145_artifact["false_positive_rate_after"]), 3)
    secl_tp = float(metrics["secl_tp_rate"])
    secl_fpr = float(metrics["secl_fpr"])
    artifact = {
        "experiment": 1157,
        "schema": "secl_cheap_tier_calibration_v1",
        "run_date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "started_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 2),
        "n_exemplars": int(n_exemplars),
        "n_correct_examples": int(n_correct),
        "thinkprm_fpr_exp1145": exp1145_fpr,
        "thinkprm_tp_exp1145": exp1145_tp,
        "secl_tp_rate": round(secl_tp, 6),
        "secl_fpr": round(secl_fpr, 6),
        "precision_recall_improved": bool(metrics["precision_recall_improved"]),
        "discriminative_signal_used": bool(metrics["discriminative_signal_used"]),
        "cheap_tier_fpr_below_30pct": bool(metrics["cheap_tier_fpr_below_30pct"]),
        "cheap_tier_tp_above_80pct": bool(metrics["cheap_tier_tp_above_80pct"]),
        "honest_verdict": _honest_verdict(secl_tp, secl_fpr, exp1145_tp, exp1145_fpr),
        "fpr_budget": FPR_BUDGET,
        "tp_target": TP_TARGET,
        "class_weight_correct": CLASS_WEIGHT_CORRECT,
        "class_weight_failure": CLASS_WEIGHT_FAILURE,
        "operating_threshold": float(metrics["operating_threshold"]),
        "exp1145_thresholds": _exp1145_threshold_summary(exp1145_artifact),
        "thinkprm_default_threshold_exp1145": float(exp1145_artifact["thinkprm_default_threshold"]),
        "thinkprm_threshold_after_exp1145": float(exp1145_artifact["thinkprm_threshold_after"]),
        "semenergy_default_threshold_exp1145": float(
            exp1145_artifact["semenergy_default_threshold"]
        ),
        "semenergy_threshold_after_exp1145": float(exp1145_artifact["semenergy_threshold_after"]),
        "n_flagged_failures": int(metrics.get("n_flagged_failures", 0)),
        "n_flagged_correct": int(metrics.get("n_flagged_correct", 0)),
        "feature_names": ["cheap_tier_score", "discriminative_signal"],
        "note": (
            "Exp1157 adds a SECL-style discriminative calibration probe to avoid "
            "Exp1145's recall-only threshold that produced a 0.96 false-positive rate."
        ),
    }
    if probe is not None:
        artifact["probe"] = probe.to_dict()
    return artifact


def run_experiment(
    *,
    exemplar_path: Path = DEFAULT_EXEMPLAR_PATH,
    fover_path: Path = DEFAULT_FOVER_PATH,
    exp1145_path: Path = DEFAULT_EXP1145_PATH,
    result_path: Path = DEFAULT_RESULT_PATH,
    think_probe: Any | None = None,
    semenergy_probe: Any | None = None,
    fover_correct_n: int = DEFAULT_FOVER_CORRECT_N,
) -> dict[str, Any]:
    """Run Exp 1157 and write the SECL calibration artifact."""

    t0 = time.time()
    if think_probe is None:  # pragma: no cover - exercised by the CLI experiment path
        from carnot.verify.spilled_energy import SpilledEnergyDetector

        think_probe = SpilledEnergyDetector()
    if semenergy_probe is None:  # pragma: no cover - exercised by the CLI experiment path
        from carnot.verify.semenergy_probe import SemEnergyProbe

        semenergy_probe = SemEnergyProbe()

    exp1145_artifact = json.loads(exp1145_path.read_text(encoding="utf-8"))
    goodfire_rows = load_json_or_jsonl(exemplar_path)
    correct_rows = select_correct_fover_rows(load_json_or_jsonl(fover_path), fover_correct_n)

    goodfire_scores = score_exemplar_rows(
        goodfire_rows,
        think_probe=think_probe,
        semenergy_probe=semenergy_probe,
        text_keys=("buggy_response", "response", "completion", "answer"),
    )
    correct_scores = score_exemplar_rows(
        correct_rows,
        think_probe=think_probe,
        semenergy_probe=semenergy_probe,
        text_keys=("step_text", "correct_response", "response", "completion"),
    )
    examples = build_calibration_examples(goodfire_scores, correct_scores)
    probe = train_secl_probe(examples)
    metrics = evaluate_secl_probe(examples, probe)
    artifact = build_exp1157_artifact(
        examples=examples,
        metrics=metrics,
        exp1145_artifact=exp1145_artifact,
        duration_s=time.time() - t0,
        probe=probe,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _example_from_score(row: CheapTierScore, *, label: int, source: str) -> SECLCalibrationExample:
    return SECLCalibrationExample(
        id=row.id,
        label=int(label),
        cheap_tier_score=cheap_tier_score(row.thinkprm_score, row.semenergy_score),
        discriminative_signal=discriminative_signal(
            row.thinkprm_score,
            thinkprm_true_logprob=getattr(row, "thinkprm_true_logprob", None),
        ),
        source=source,
        category=row.category,
    )


def _feature_label_arrays(
    examples: Sequence[SECLCalibrationExample],
) -> tuple[np.ndarray, np.ndarray]:
    features = np.vstack([example.features for example in examples]).astype(np.float64)
    labels = np.array([example.label for example in examples], dtype=np.float64)
    return features, labels


def _class_weight_array(labels: np.ndarray) -> np.ndarray:
    return np.where(labels == 1, CLASS_WEIGHT_FAILURE, CLASS_WEIGHT_CORRECT).astype(np.float64)


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _clamp_probability(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _rate(flags: Sequence[bool] | np.ndarray) -> float:
    values = np.asarray(flags, dtype=bool)
    return float(values.mean()) if len(values) else 0.0


def _exp1145_threshold_summary(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "thinkprm": {
            "default": float(artifact["thinkprm_default_threshold"]),
            "applied": float(artifact["thinkprm_threshold_after"]),
            "adjusted": bool(artifact["thinkprm_threshold_adjusted"]),
        },
        "semenergy": {
            "default": float(artifact["semenergy_default_threshold"]),
            "applied": float(artifact["semenergy_threshold_after"]),
            "adjusted": bool(artifact["semenergy_threshold_adjusted"]),
        },
    }


def _honest_verdict(secl_tp: float, secl_fpr: float, exp1145_tp: float, exp1145_fpr: float) -> str:
    if secl_tp >= TP_TARGET and secl_fpr <= FPR_BUDGET:
        return "calibrated_tp_above_80_fpr_below_30"
    if secl_tp >= TP_TARGET and secl_fpr < exp1145_fpr:
        return "tp_improved_fpr_reduced_not_gate"
    if secl_fpr < exp1145_fpr and secl_tp < exp1145_tp:
        return "trade_off_tp_dropped"
    return "honest_negative_no_improvement"
