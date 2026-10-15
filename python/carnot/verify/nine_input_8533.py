"""REQ-VERIFY-8533: finite parity keeps the established Python policy intact."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify.evidence_features_7980 import extract
from carnot.verify.v726_decision_metrics import action

Json = dict[str, Any]


def normalize(head: Json, raw: sparse.Array, residual: sparse.Array) -> sparse.Array:
    """Use the shipped raw-p2 offset and stable two-energy normalization stages."""
    q = np.clip(raw[:, 0], 1e-4, 1 - 1e-4)
    z = (np.log(q / (1 - q)) + residual) / head["temperature"]
    energies = np.column_stack((np.zeros(len(raw)), -z))
    weights = np.exp(-energies - np.max(-energies, axis=1, keepdims=True))
    return np.asarray(weights[:, 1] / weights.sum(axis=1), dtype=float)


def probabilities(head: Json, raw: sparse.Array, native: Any = None) -> tuple[sparse.Array, Any]:
    """Change only the residual provider and guard both final typed thresholds."""
    scaled, _ = sparse.scale(raw, head["scaler"])
    theta = sparse.parameters(head)
    residual = np.asarray(
        sparse.basis(scaled) @ theta
        if native is None
        else native.nine_input_residual_8533(theta.tolist(), scaled.tolist()),
        dtype=float,
    )
    p = normalize(head, raw, residual)
    fallback: Any = np.zeros(len(raw), dtype=bool)
    if native is not None:
        fallback = (np.abs(p - 0.2) <= 1e-10) | (np.abs(p - 0.8) <= 1e-10)
        if np.any(fallback):
            p[fallback] = sparse.predict(head, raw[fallback])
    return p, fallback


def request(head: Json, row: Json, native: Any = None) -> str:
    """Recompute public text features through serialization for one cached request."""
    features = extract(row["public"])
    if features["values"] is None or row["q"] is None:
        raise ValueError("request_features")
    raw = np.asarray([[row["q"], *features["values"]]], dtype=float)
    p, fallback = probabilities(head, raw, native)
    return json.dumps(
        dict(
            unit_id=row["unit_id"],
            probability=float(p[0]),
            decision=action(float(p[0])),
            boundary_fallback=bool(fallback[0]),
        ),
        sort_keys=True,
        separators=(",", ":"),
    )


def qualify(head: Json, raw: sparse.Array, native: Any) -> list[Json]:
    """Compare every finite operand with independently evaluated dense SciPy bases."""
    scaled, _ = sparse.scale(raw, head["scaler"])
    cases = [("answer", i, x.tolist(), 1.0) for i, x in enumerate(scaled)]
    for dimension in range(9):
        for knot in sorted(set(sparse.KNOTS)):
            for value in [np.nextafter(knot, -np.inf), knot, np.nextafter(knot, np.inf)]:
                x = [0.5] * 9
                x[dimension] = float(np.clip(value, 0.0, 1.0))
                cases.append(("knot", len(cases), x, 1.0))
    for scale in [0.0, 0.125, 1.0, 2.0]:
        cases.append(("zero" if scale == 0 else "decay", len(cases), [0.37] * 9, scale))
    rows = []
    for kind, index, values, decay in cases:
        x = np.asarray([values])
        theta = sparse.parameters(head) * decay
        reference = float((sparse.basis(x) @ theta)[0])
        actual = float(native.nine_input_residual_8533(theta.tolist(), [values])[0])
        low = np.asarray(head["scaler"]["minimum"])
        high = np.asarray(head["scaler"]["maximum"])
        operand = (
            raw[index : index + 1]
            if kind == "answer"
            else low + x * np.where(high > low, high - low, 1.0)
        )
        pref = float(normalize(head, operand, np.asarray([reference]))[0])
        pact = float(normalize(head, operand, np.asarray([actual]))[0])
        fallback = min(abs(pact - 0.2), abs(pact - 0.8)) <= 1e-10
        guarded = pref if fallback else pact
        re, pe = abs(actual - reference), abs(pact - pref)
        mismatch = action(pref) != action(guarded)
        rows.append(
            dict(
                unit_id=f"{kind}-{index}",
                kind=kind,
                scaled_features=values,
                decay_multiplier=decay,
                reference_residual=reference,
                native_residual=actual,
                reference_probability=pref,
                native_probability=pact,
                residual_error=re,
                probability_error=pe,
                boundary_fallback=fallback,
                decision_mismatch=mismatch,
                native_execution_count=1,
                passed=re <= 1e-12 * (1 + abs(reference))
                and pe <= 1e-12 * (1 + abs(pref))
                and not mismatch,
            )
        )
        if len(rows) % 100 == 0:
            print(
                f"[exp8533] phase=finite_panel completed={len(rows)} pending={len(cases) - len(rows)}",
                flush=True,
            )
    invalid = [
        ("coefficient_dimension", [0.0] * 108, [[0.5] * 9]),
        ("coefficient_nonfinite", [float("nan")] * 109, [[0.5] * 9]),
        ("empty_batch", [0.0] * 109, []),
    ]
    invalid += [
        (name, [0.0] * 109, [x])
        for name, x in [
            ("dimension", [0.5] * 8),
            ("extra_dimension", [0.5] * 10),
            ("nan", [float("nan")] * 9),
            ("infinity", [float("inf")] * 9),
            ("below_clip", [-0.01] * 9),
            ("above_clip", [1.01] * 9),
        ]
    ]
    invalid.append(("overflow", [float(np.finfo(float).max)] * 109, [[0.5] * 9]))
    for name, coefficients, features in invalid:
        passed = False
        try:
            native.nine_input_residual_8533(coefficients, features)
        except ValueError:
            passed = True
        rows.append(
            dict(
                unit_id="malformed-" + name,
                kind="malformed",
                passed=passed,
                missing_reason=None if passed else "native_accepted_malformed_input",
                native_execution_count=1,
            )
        )
    zero = dict(head, parameters=[0.0] * 109, decay_scale=1.0, temperature=1.0)
    for threshold in [0.2, 0.8]:
        for q in [np.nextafter(threshold, 0.0), threshold, np.nextafter(threshold, 1.0)]:
            operand = np.asarray([[q] + [0.5] * 8])
            p, fallback = probabilities(zero, operand, native)
            pref = float(sparse.predict(zero, operand)[0])
            mismatch = action(pref) != action(float(p[0]))
            rows.append(
                dict(
                    unit_id=f"threshold-{q!r}",
                    kind="threshold",
                    reference_residual=0.0,
                    native_residual=0.0,
                    residual_error=0.0,
                    reference_probability=pref,
                    native_probability=float(p[0]),
                    probability_error=abs(float(p[0]) - pref),
                    boundary_fallback=bool(fallback[0]),
                    decision_mismatch=mismatch,
                    native_execution_count=1,
                    passed=bool(fallback[0]) and not mismatch,
                )
            )
    return rows
