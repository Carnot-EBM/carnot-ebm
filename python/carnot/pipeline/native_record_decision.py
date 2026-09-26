"""Reference typed record score for REQ-REPORT-7710.

The caller supplies counts and fixture parameters. This module reads no labels,
model weights or source text; it is a parity oracle for the native boundary.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


FEATURES = (
    "tuple_supported",
    "tuple_contradicted",
    "path_line_supported",
    "path_line_contradicted",
    "unknown_propositions",
    "residual_unknown_bytes",
    "checked_fraction",
    "source_records",
)
RECORD_SCHEMA = "carnot.exp7700.record_features.v1"
PARAMETER_SCHEMA = "carnot.exp7710.record_binary_energy.v1"


def predict_reference(
    payload: Mapping[str, Any], parameters: Mapping[str, Any]
) -> tuple[float, str]:
    """Return normalized error probability and typed action for one record."""

    if payload.get("schema") != RECORD_SCHEMA:
        raise ValueError("record_schema_invalid")
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(FEATURES):
        raise ValueError("record_fields_invalid")
    vector = []
    for name in FEATURES:
        value = counts[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("record_field_invalid")
        if name == "checked_fraction":
            if not 0 <= value <= 1:
                raise ValueError("record_field_invalid")
            vector.append(float(value))
        else:
            if value < 0 or int(value) != value:
                raise ValueError("record_field_invalid")
            vector.append(math.log1p(value))
    if parameters.get("schema") != PARAMETER_SCHEMA:
        raise ValueError("parameter_schema_invalid")
    weights = parameters.get("weights")
    bias = parameters.get("bias")
    thresholds = parameters.get("thresholds")
    if (
        not isinstance(weights, list)
        or len(weights) != 8
        or not isinstance(thresholds, list)
        or len(thresholds) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in [*weights, bias, *thresholds]
        )
        or not 0 <= thresholds[0] <= thresholds[1] <= 1
    ):
        raise ValueError("parameter_fields_invalid")
    logit = float(bias) + sum(float(x) * float(w) for x, w in zip(vector, weights, strict=True))
    if not math.isfinite(logit):
        raise ValueError("energy_not_finite")
    probability = (
        1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
    )
    action = (
        "accept"
        if probability < thresholds[0]
        else "reject"
        if probability >= thresholds[1]
        else "escalate"
    )
    return probability, action
