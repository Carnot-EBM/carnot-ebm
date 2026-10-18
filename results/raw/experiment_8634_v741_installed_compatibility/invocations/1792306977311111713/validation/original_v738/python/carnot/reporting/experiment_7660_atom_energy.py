"""Small conditional energy head for REQ-REPORT-7660.

The inherited probability is the error-state baseline. A bounded fitted
residual changes its log odds; the two normalized states are correct and error.
No source is executed and no generator weight is changed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize


FEATURES = ("bias", "checked_fraction", "contradiction_fraction", "ambiguity_fraction")
SCHEMA = "carnot.exp7660.head.v1"
CLIP = 1e-6
RIDGES = (0.1, 1.0, 10.0)


def join_role(features: list[dict], labels: list[dict], role: str) -> list[dict]:
    """Join an original-source feature row to its isolated dataset label."""

    label_by_id = {row["component_hash"]: row for row in labels}
    if len(label_by_id) != len(labels) or len(features) != len(labels):
        raise ValueError("role_roster_mismatch")
    joined = []
    seen: set[str] = set()
    for feature in features:
        unit = feature["unit_id"]
        if unit in seen or feature["arm"] != "original_source" or feature["role"] != role:
            raise ValueError("feature_role_or_arm_invalid")
        seen.add(unit)
        label = label_by_id.get(unit)
        if label is None:
            raise ValueError("label_missing")
        if feature["source_sha256"] != feature["original_source_sha256"]:
            raise ValueError("source_hash_mismatch")
        allowed = role == "fit" and feature["partition"] == "fit_optimization"
        if (
            label["role"] != role
            or label["learning_partition"] != feature["partition"]
            or label["label"] not in (0, 1)
            or label["training_allowed"] is not allowed
            or label["evaluator_only"] is not False
        ):
            raise ValueError("independent_label_custody_invalid")
        probability = float(label["raw_probability"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("baseline_probability_invalid")
        joined.append({"feature": feature, "label": label["label"], "probability": probability})
    return joined


def _vector(row: dict) -> tuple[float, ...]:
    denominator = max(1, int(row["denominator"]))
    return (
        1.0,
        min(1.0, max(0.0, row["checked_structural_propositions"] / denominator)),
        min(1.0, max(0.0, row["scoped_contradictions"] / denominator)),
        min(1.0, max(0.0, row["ambiguity"] / denominator)),
    )


def _head(weights: list[float]) -> dict:
    return {
        "schema": SCHEMA,
        "feature_order": list(FEATURES),
        "weights": weights,
        "clip": CLIP,
        "normalization": "log1p_counts_and_sentence_fraction",
        "residual_clip": 2.0,
        "state_order": ["correct", "error"],
    }


def energy_probability(row: dict, inherited: float, head: dict) -> tuple[float, float]:
    """Return normalized (correct, error) state probabilities."""

    if (
        head.get("schema") != SCHEMA
        or head.get("feature_order") != list(FEATURES)
        or head.get("clip") != CLIP
        or head.get("normalization") != "log1p_counts_and_sentence_fraction"
        or len(head.get("weights", [])) != len(FEATURES)
    ):
        raise ValueError("head_schema_incompatible")
    p = float(inherited)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("baseline_probability_invalid")
    p = min(1 - CLIP, max(CLIP, p))
    residual = sum(float(w) * x for w, x in zip(head["weights"], _vector(row), strict=True))
    if not math.isfinite(residual):
        raise ValueError("head_weight_not_finite")
    residual = min(2.0, max(-2.0, residual))
    logit = math.log(p / (1 - p)) + residual
    error = 1 / (1 + math.exp(-logit))
    return 1 - error, error


def score(row: dict, inherited: float, head: dict) -> float:
    """Return the error-state probability for the typed policy."""

    return energy_probability(row, inherited, head)[1]


def action(error_probability: float, thresholds: tuple[float, float]) -> str:
    """Choose reject, escalate, or accept from frozen error-risk cutoffs."""

    lower, upper = thresholds
    if not 0 <= lower <= upper <= 1:
        raise ValueError("thresholds_invalid")
    if error_probability < lower:
        return "accept"
    if error_probability > upper:
        return "reject"
    return "escalate"


def _fit(rows: list[dict], ridge: float, active: tuple[int, ...], erased: bool = False) -> dict:
    x = np.array(
        [
            _vector(row.get("erased_feature", row["feature"]) if erased else row["feature"])
            for row in rows
        ],
        dtype=float,
    )
    p = np.clip(np.array([row["probability"] for row in rows]), CLIP, 1 - CLIP)
    y = np.array([row["label"] for row in rows], dtype=float)
    baseline = np.log(p / (1 - p))

    def objective(theta: np.ndarray) -> float:
        logits = baseline + np.clip(x[:, active] @ theta, -2, 2)
        return float(np.mean(np.logaddexp(0, logits) - y * logits) + ridge * theta @ theta)

    result = minimize(
        objective, np.zeros(len(active)), method="L-BFGS-B", bounds=[(-2, 2)] * len(active)
    )
    if not result.success or not math.isfinite(result.fun):
        return _head([0.0] * len(FEATURES))
    weights = [0.0] * len(FEATURES)
    for index, value in zip(active, result.x, strict=True):
        weights[index] = float(value)
    return _head(weights)


def _brier(rows: list[dict], head: dict, erased: bool = False) -> float:
    return sum(
        (
            score(
                row.get("erased_feature", row["feature"]) if erased else row["feature"],
                row["probability"],
                head,
            )
            - row["label"]
        )
        ** 2
        for row in rows
    ) / len(rows)


def fit_heads(fit: list[dict], tune: list[dict]) -> dict:
    """Freeze six preregistered candidates and honest cheap controls."""

    if not fit or not tune:
        raise ValueError("fit_or_tune_empty")
    identity = _head([0.0] * len(FEATURES))
    settings = []
    candidates = []
    for capacity in (2, 4):
        for ridge in RIDGES:
            head = _fit(fit, ridge, tuple(range(capacity)))
            loss = _brier(tune, head)
            settings.append({"capacity": capacity, "ridge": ridge, "tune_brier": loss})
            candidates.append((loss, head))
    best_loss, best_head = min(candidates, key=lambda item: item[0])
    identity_loss = _brier(tune, identity)
    selected = "atom" if best_loss < identity_loss else "identity"
    heads = {
        "identity": identity,
        "scalar": _fit(fit, 1.0, (0,)),
        "cheap_atom": _fit(fit, 1.0, (0, 2)),
        "atom": best_head,
        "source_erased": _fit(fit, 1.0, tuple(range(4)), erased=True),
    }
    return {
        "heads": heads,
        "settings": settings,
        "selected": selected,
        "identity_tune_brier": identity_loss,
        "selected_tune_brier": min(best_loss, identity_loss),
    }
