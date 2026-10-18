"""REQ-VERIFY-8608: complete inference without the research import graph.

The private worker imports this file directly because the public package eagerly
loads training. Arithmetic below retains the shipped SciPy binary64 operations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any, TypeAlias
import unicodedata
import re

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

Json = dict[str, Any]
Array: TypeAlias = NDArray[np.float64]
KNOTS = [0.0] * 4 + [i / 9 for i in range(1, 9)] + [1.0] * 4
PUBLIC_KEYS = frozenset({"family_id", "source_bytes", "answer_bytes"})


def load_module(path: Path, name: str) -> Any:
    """File imports avoid running a package initializer that loads training."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("extension_spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


alignment = load_module(
    Path(__file__).parents[1] / "verify/source_alignment.py", "_resident_alignment"
)


def canonical_hash(value: Any) -> str:
    """Stable JSON spelling keeps public feature identifiers unchanged."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def normalized(source: bytes) -> str:
    """Merge formatting variants so one source cannot occupy two roles."""
    text = unicodedata.normalize("NFKC", source.decode()).casefold()
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def extract(row: Json) -> Json:
    """Measure complete original text and abstain instead of truncating it."""
    if set(row) != PUBLIC_KEYS or not isinstance(row["family_id"], str) or not row["family_id"]:
        raise ValueError("public_fields")
    try:
        source, answer = (bytes.fromhex(row[k]) for k in ("source_bytes", "answer_bytes"))
        view = alignment.prepare(source, answer)
    except (ValueError, TypeError, UnicodeError) as error:
        raise ValueError("public_bytes") from error
    reason = view["abstention"] or ("empty_source" if not source.strip() else None)
    windows = dict.fromkeys(w.strip() for w in view["windows"])
    pairs = (
        [alignment.pair_features(w, view["answer_units"])[-4:] for w in windows]
        if not reason
        else []
    )
    values = (
        [
            v
            for i in range(4)
            for v in (sum(p[i] for p in pairs) / len(pairs), max(p[i] for p in pairs))
        ]
        if pairs
        else None
    )
    value = dict(
        family_id=row["family_id"],
        values=values,
        abstention=reason,
        window_count=len(windows),
        source_normalized_hash=normalized(source),
    )
    return dict(value, feature_hash=canonical_hash(value))


def scale(x: Array, scaler: Json) -> tuple[Array, Json]:
    """Clip extrapolation explicitly and count both directions per feature."""
    low, high = np.asarray(scaler["minimum"]), np.asarray(scaler["maximum"])
    span = np.where(high > low, high - low, 1.0)
    counters = dict(below=(x < low).sum(axis=0).tolist(), above=(x > high).sum(axis=0).tolist())
    return np.asarray(np.clip((x - low) / span, 0, 1), dtype=float), counters


def basis(x: Array) -> Array:
    """Clamped cubic bases sum to one and expose at most four local weights."""
    return np.column_stack(
        [BSpline.design_matrix(x[:, j], KNOTS, 3).toarray() for j in range(9)] + [np.ones(len(x))]
    )


def parameters(head: Json) -> Array:
    """The multiplier is part of the state and defines effective coefficients."""
    return np.asarray(head["parameters"], dtype=float) * float(head["decay_scale"])


def normalize(head: Json, raw: Array, residual: Array) -> Array:
    """Use the shipped raw-p2 offset and stable two-energy normalization stages."""
    q = np.clip(raw[:, 0], 1e-4, 1 - 1e-4)
    z = (np.log(q / (1 - q)) + residual) / head["temperature"]
    energies = np.column_stack((np.zeros(len(raw)), -z))
    weights = np.exp(-energies - np.max(-energies, axis=1, keepdims=True))
    return np.asarray(weights[:, 1] / weights.sum(axis=1), dtype=float)


def action(p: float) -> str:
    """Fixed thresholds retain the original accept, reject and review policy."""
    return "accept" if p <= 0.2 else "reject" if p >= 0.8 else "escalate"


def predict(head: Json, raw: Array) -> Array:
    """The boundary control uses the exact shipped spline logit operation order."""
    x, _ = scale(raw, head["scaler"])
    offset = np.log(np.clip(raw[:, 0], 1e-4, 1 - 1e-4) / (1 - np.clip(raw[:, 0], 1e-4, 1 - 1e-4)))
    z = np.asarray(offset + basis(x) @ parameters(head)) / head["temperature"]
    energies = np.column_stack((np.zeros(len(raw)), -z))
    weights = np.exp(-energies - np.max(-energies, axis=1, keepdims=True))
    return np.asarray(weights[:, 1] / weights.sum(axis=1), dtype=float)


def batch(head: Json, rows: list[Json], native: Any = None) -> tuple[list[Json], list[Json]]:
    """Recompute every text feature before numerical work; cached vectors are unused."""
    began = time.monotonic_ns()
    spans: list[Json] = []

    def mark(name: str) -> None:
        """Adjacent boundaries make all operation spans disjoint by construction."""
        end = time.monotonic_ns()
        spans.append(
            dict(operation=name, start_ns=spans[-1]["end_ns"] if spans else began, end_ns=end)
        )

    features = [extract(r["public"])["values"] for r in rows]
    if any(x is None or r["q"] is None for r, x in zip(rows, features, strict=True)):
        raise ValueError("request_features")
    raw = np.asarray([[r["q"], *x] for r, x in zip(rows, features, strict=True)], dtype=float)
    mark("public_text_features")
    scaled, _ = scale(raw, head["scaler"])
    mark("scaling")
    q = np.clip(raw[:, 0], 1e-4, 1 - 1e-4)
    offset = np.log(q / (1 - q))
    mark("offset")
    theta = parameters(head)
    residual: Array = np.asarray(
        basis(scaled) @ theta
        if native is None
        else native.nine_input_residual_8533(theta.tolist(), scaled.tolist()),
        dtype=float,
    )
    mark("binding_residual")
    z = (offset + residual) / head["temperature"]
    energies = np.column_stack((np.zeros(len(raw)), -z))
    weights = np.exp(-energies - np.max(-energies, axis=1, keepdims=True))
    probabilities = weights[:, 1] / weights.sum(axis=1)
    fallback: Any = np.zeros(len(rows), dtype=bool)
    if native is not None:
        fallback = (np.abs(probabilities - 0.2) <= 1e-10) | (np.abs(probabilities - 0.8) <= 1e-10)
        if np.any(fallback):
            probabilities[fallback] = predict(head, raw[fallback])
    mark("normalization_and_python_fallback")
    decisions = [action(float(p)) for p in probabilities]
    mark("typed_decision")
    results = []
    for row, p, decision, used in zip(rows, probabilities, decisions, fallback, strict=True):
        response = json.dumps(
            dict(
                unit_id=row["unit_id"],
                probability=float(p),
                decision=decision,
                boundary_fallback=bool(used),
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        results.append(
            dict(
                unit_id=row["unit_id"],
                response=response,
                start_ns=began,
                end_ns=time.monotonic_ns(),
            )
        )
    mark("serialization")
    for result in results:
        result["latency_ns"] = result["end_ns"] - began
    return results, spans


def evaluate(head: Json, row: Json, native: Any) -> tuple[Json, list[Json]]:
    """Text requests use the complete adapter; finite controls keep original scaled operands."""
    if row.get("kind") != "numerical":
        if (
            type(row["q"]) not in {int, float}
            or not math.isfinite(row["q"])
            or not 0 <= row["q"] <= 1
        ):
            raise ValueError("probability_input")
        responses, spans = batch(head, [row], native)
        return json.loads(responses[0]["response"]), spans
    start = time.monotonic_ns()
    scaled: Any = np.asarray([row["scaled_features"]], dtype=float)
    theta = parameters(head) * row["decay_multiplier"]
    residual: Array = np.asarray(
        basis(scaled) @ theta
        if native is None
        else native.nine_input_residual_8533(theta.tolist(), scaled.tolist()),
        dtype=float,
    )
    end = time.monotonic_ns()
    low, high = np.asarray(head["scaler"]["minimum"]), np.asarray(head["scaler"]["maximum"])
    raw = low + scaled * np.where(high > low, high - low, 1.0)
    probability = float(normalize(head, raw, residual)[0])
    fallback = native is not None and min(abs(probability - 0.2), abs(probability - 0.8)) <= 1e-10
    if fallback:
        reference = basis(scaled) @ theta
        probability = float(normalize(head, raw, reference)[0])
    response = dict(
        unit_id=row["unit_id"],
        probability=probability,
        residual=float(residual[0]),
        decision=action(probability),
        boundary_fallback=fallback,
    )
    return response, [
        dict(operation="binding_residual", start_ns=start, end_ns=end),
        dict(
            operation="normalization_and_python_fallback", start_ns=end, end_ns=time.monotonic_ns()
        ),
    ]
