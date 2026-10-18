"""Small normalized decision energies for REQ-ENERGY-7703.

This module reads source and answer text to count narrow certificates. It
does not read a dataset label or turn an unchecked sentence into support.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Mapping

import numpy as np

from carnot.verify.record_addresses import analyze_answer


FEATURE_ORDER = (
    "tuple_supported",
    "tuple_contradicted",
    "path_line_supported",
    "path_line_contradicted",
    "unknown_propositions",
    "residual_unknown_bytes",
    "checked_fraction",
    "source_records",
)
ATOM_INDICES = (2, 3, 4, 6, 7)
HEAD_SCHEMA = "carnot.exp7703.binary_energy.v1"


def feature_view(row: Mapping[str, Any], source_override: str | None = None) -> dict[str, Any]:
    """Count typed evidence from public text only, preserving empty evidence."""

    if any("label" in key.lower() and key != "labels_accessible" for key in row):
        raise ValueError("label_or_field_access_rejected")
    source = row["complete_source"] if source_override is None else source_override
    analysis = analyze_answer(source, row["complete_answer"])
    propositions = analysis["propositions"]
    tuple_rows = [p for p in propositions if p["certificate_type"] == "bound_tuple"]
    path_rows = [p for p in propositions if p["certificate_type"] != "bound_tuple"]
    counts = {
        "tuple_supported": sum(p["narrow_status"] == "supported" for p in tuple_rows),
        "tuple_contradicted": sum(p["narrow_status"] == "contradicted" for p in tuple_rows),
        "path_line_supported": sum(p["narrow_status"] == "supported" for p in path_rows),
        "path_line_contradicted": sum(p["narrow_status"] == "contradicted" for p in path_rows),
        "unknown_propositions": sum(p["narrow_status"] == "unknown" for p in propositions),
        "residual_unknown_bytes": len(analysis["residual_unknown_text"].encode()),
        "checked_fraction": sum(p["narrow_status"] != "unknown" for p in propositions)
        / max(1, len(propositions)),
        "source_records": len(analysis["records"]),
    }
    vector = [
        float(counts[name]) if name == "checked_fraction" else math.log1p(counts[name])
        for name in FEATURE_ORDER
    ]
    return {
        "vector": vector,
        "counts": counts,
        "proposition_count": len(propositions),
        "checked_count": sum(p["narrow_status"] != "unknown" for p in propositions),
        "whole_answer_status": analysis["whole_answer_status"],
    }


def feature_information(x: np.ndarray, checked: np.ndarray) -> dict[str, Any]:
    """Show measured variation without discarding zero-coverage groups."""

    return {
        "groups": int(len(x)),
        "checked_coverage": int(np.count_nonzero(checked)),
        "zero_coverage_groups": int(np.count_nonzero(checked == 0)),
        "feature_variance": np.var(x, axis=0).tolist(),
        "source_information_available": bool(np.any(np.var(x, axis=0) > 0) and np.any(checked)),
    }


def fit_head(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    *,
    seed: int,
    steps: int,
    ridge: float,
) -> dict[str, Any]:
    """Fit one bounded two-state head by full-batch gradient descent."""

    if kind not in {"prior", "logistic", "mlp", "gibbs"}:
        raise ValueError("head_kind_invalid")
    if not 0 <= steps <= 500 or not 0 <= ridge <= 10:
        raise ValueError("training_budget_invalid")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or not np.isin(y, [0, 1]).all():
        raise ValueError("training_rows_invalid")
    if not np.isfinite(x).all():
        raise ValueError("features_not_finite")
    d = x.shape[1]
    prevalence = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
    bias = math.log(prevalence / (1 - prevalence))
    no_information = not bool(np.any(np.var(x, axis=0) > 0))
    if kind == "prior" or no_information:
        return {
            "schema": HEAD_SCHEMA,
            "kind": "prior",
            "requested_kind": kind,
            "input_width": d,
            "bias": bias,
            "parameter_count": 1,
            "steps": 0,
            "seed": seed,
            "ridge": ridge,
            "source_information_learned": False,
        }
    rng = np.random.default_rng(seed)
    head: dict[str, Any] = {
        "schema": HEAD_SCHEMA,
        "kind": kind,
        "requested_kind": kind,
        "input_width": d,
        "steps": steps,
        "seed": seed,
        "ridge": ridge,
        "source_information_learned": True,
    }
    if kind == "logistic":
        w = np.zeros(d)
        b = bias
        for _ in range(steps):
            z = np.clip(x @ w + b, -30, 30)
            residual = 1 / (1 + np.exp(-z)) - y
            w -= 0.05 * (x.T @ residual / len(x) + ridge * w)
            b -= 0.05 * float(np.mean(residual))
        head.update(weights=w.tolist(), bias=b, parameter_count=d + 1)
        return head
    width = 8
    w1 = rng.normal(0, 0.05, (d, width))
    b1 = np.zeros(width)
    outputs = 2 if kind == "gibbs" else 1
    w2 = rng.normal(0, 0.05, (width, outputs))
    b2 = np.array([0.0, bias]) if kind == "gibbs" else np.array([bias])
    parameter_count = d * width + width + width * outputs + outputs
    if parameter_count > 4096:
        raise ValueError("parameter_cap_exceeded")
    for _ in range(steps):
        hidden = np.tanh(x @ w1 + b1)
        logits = hidden @ w2 + b2
        z = logits[:, 1] - logits[:, 0] if kind == "gibbs" else logits[:, 0]
        residual = 1 / (1 + np.exp(-np.clip(z, -30, 30))) - y
        derivative = (
            np.column_stack((-residual, residual)) if kind == "gibbs" else residual[:, None]
        )
        gw2 = hidden.T @ derivative / len(x) + ridge * w2
        gb2 = np.mean(derivative, axis=0)
        upstream = (derivative @ w2.T) * (1 - hidden**2)
        gw1 = x.T @ upstream / len(x) + ridge * w1
        gb1 = np.mean(upstream, axis=0)
        w2 -= 0.05 * gw2
        b2 -= 0.05 * gb2
        w1 -= 0.05 * gw1
        b1 -= 0.05 * gb1
    head.update(
        w1=w1.tolist(),
        b1=b1.tolist(),
        w2=w2.tolist(),
        b2=b2.tolist(),
        parameter_count=parameter_count,
    )
    return head


def probabilities(x: np.ndarray, head: Mapping[str, Any]) -> tuple[float, float]:
    """Evaluate exact normalized p(y|x) from the two saved energies."""

    if head.get("schema") != HEAD_SCHEMA:
        raise ValueError("head_schema_invalid")
    vector = np.asarray(x, dtype=float)
    if vector.shape != (head["input_width"],) or not np.isfinite(vector).all():
        raise ValueError("feature_shape_invalid")
    kind = head["kind"]
    if kind == "prior":
        logits = np.array([0.0, float(head["bias"])])
    elif kind == "logistic":
        logits = np.array([0.0, float(vector @ np.asarray(head["weights"]) + head["bias"])])
    elif kind in {"mlp", "gibbs"}:
        hidden = np.tanh(vector @ np.asarray(head["w1"]) + np.asarray(head["b1"]))
        output = hidden @ np.asarray(head["w2"]) + np.asarray(head["b2"])
        logits = output if kind == "gibbs" else np.array([0.0, output[0]])
    else:
        raise ValueError("head_kind_invalid")
    if not np.isfinite(logits).all():
        raise ValueError("energy_not_finite")
    energies = -logits
    weights = np.exp(-(energies - np.min(energies)))
    return tuple(float(v) for v in weights / np.sum(weights))  # type: ignore[return-value]


def action(error_probability: float, thresholds: tuple[float, float]) -> str:
    """Map error risk to accept, escalate or reject using frozen cutoffs."""

    low, high = thresholds
    if not 0 <= low <= high <= 1:
        raise ValueError("thresholds_invalid")
    if error_probability < low:
        return "accept"
    if error_probability >= high:
        return "reject"
    return "escalate"


def decision_cost(chosen: str, label: int) -> float:
    """Charge one for a wrong unassisted action and 0.2 for escalation."""

    if chosen == "escalate":
        return 0.2
    if chosen not in {"accept", "reject"} or label not in (0, 1):
        raise ValueError("decision_or_label_invalid")
    return float((chosen == "accept") == bool(label))


def freeze_policy(
    probabilities_error: list[float], labels: list[int]
) -> tuple[tuple[float, float], list[dict]]:
    """Choose cutoffs by fixed policy-role cost, with stable tie breaking."""

    if len(probabilities_error) != len(labels) or not labels:
        raise ValueError("policy_rows_invalid")
    grid = [i / 20 for i in range(21)]
    options = [
        {
            "thresholds": [low, high],
            "mean_cost": sum(
                decision_cost(action(p, (low, high)), y)
                for p, y in zip(probabilities_error, labels, strict=True)
            )
            / len(labels),
        }
        for low in grid
        for high in grid
        if low <= high
    ]
    best = min(options, key=lambda item: (item["mean_cost"], item["thresholds"]))
    return tuple(best["thresholds"]), options  # type: ignore[return-value]


def freeze_online_protocol(replay_losses: list[float]) -> dict[str, Any]:
    """Freeze candidate grammar and update schedule from fit/tune replay."""

    if not replay_losses or not np.isfinite(replay_losses).all():
        raise ValueError("replay_losses_invalid")
    return {
        "primitives": list(FEATURE_ORDER),
        "pairs": [list(pair) for pair in combinations(FEATURE_ORDER, 2)],
        "max_proposals_per_arm": 6,
        "admission_checks_per_proposal": 1,
        "max_gradient_steps_per_proposal": 50,
        "rejected_proposals_count_toward_budget": True,
        "scheduler": {
            "rule": "trigger_when_delayed_loss_at_or_above_fit_tune_p75",
            "percentile": 75,
            "threshold": float(np.percentile(replay_losses, 75)),
            "calibration_rows": len(replay_losses),
        },
        "fixed_period_comparator": {"period_labeled_feedback": 5},
        "labels_sealed": ["evaluation", "online_update", "online_admission", "retention"],
    }
