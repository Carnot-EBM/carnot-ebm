"""REQ-VERIFY-8461: independent dense arithmetic checks the sparse Brier update."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.special import expit

from carnot.verify import objective_checkpoint_8461 as adapter

Json = dict[str, Any]
Array: TypeAlias = NDArray[np.float64]


def operands(head: Json, x: Array) -> tuple[Array, Array, float]:
    """Individual cubic splines avoid the production sparse design-matrix implementation."""
    low, high = np.asarray(head["scaler"]["minimum"]), np.asarray(head["scaler"]["maximum"])
    scaled = np.clip((x - low) / np.where(high > low, high - low, 1.0), 0, 1)
    knots = [0.0] * 4 + [i / 9 for i in range(1, 9)] + [1.0] * 4
    matrix = np.asarray(
        [BSpline(knots, np.eye(12)[k], 3)(v) for v in scaled for k in range(12)] + [1.0]
    )
    theta = np.asarray(head["parameters"], dtype=float) * head["decay_scale"]
    q = min(max(float(x[0]), 1e-4), 1 - 1e-4)
    return matrix, theta, float(np.log(q / (1 - q)))


def dense(head: Json, x: Array, y: int) -> Array:
    """The bias receives the same data derivative and L2 convention as other coefficients."""
    matrix, theta, offset = operands(head, x)
    t = head["temperature"]
    p = float(expit((offset + matrix @ theta) / t))
    return np.asarray(matrix * (2 * (p - y) * p * (1 - p) / t) + 0.002 * theta)


def checks(original: Json) -> list[Json]:
    """Finite differences operate on effective coefficients and include the stated regularizer."""
    rows: list[Json] = []
    low, high = np.asarray(original["scaler"]["minimum"]), np.asarray(original["scaler"]["maximum"])
    for support, v in [("lower", 0.0), ("interior", 4 / 9), ("upper", 1.0)]:
        x = np.asarray(low + v * (high - low), dtype=float)
        for t in (0.5, 1.0, 2.0):
            for z in (-20.0, -4.0, 0.0, 4.0, 20.0):
                h = deepcopy(original)
                h["temperature"] = t
                matrix, theta, offset = operands(h, x)
                theta[-1] += z - offset - matrix @ theta
                h.update(parameters=theta.tolist(), decay_scale=1.0)
                for y in (0, 1):

                    def objective(coefficients: Array) -> float:
                        """A declared Brier objective makes numerical differentiation independent."""
                        p = float(expit((offset + matrix @ coefficients) / t))
                        return float((p - y) ** 2 + 0.001 * coefficients @ coefficients)

                    reference = dense(h, x, y)
                    changed, receipt = adapter.update(h, x, y)
                    gradient = 0.002 * theta
                    gradient[np.asarray(receipt["gradient_ids"])] += np.asarray(
                        receipt["data_gradient"]
                    )
                    numeric = []
                    for i in range(109):
                        delta = np.zeros(109)
                        delta[i] = 1e-4
                        numeric.append((objective(theta + delta) - objective(theta - delta)) / 2e-4)
                    numeric_array = np.asarray(numeric)
                    error = np.maximum(
                        np.abs(gradient - reference), np.abs(numeric_array - reference)
                    )
                    mask = np.abs(reference) > 1e-6
                    relative = float(np.max(error[mask] / np.abs(reference[mask]), initial=0))
                    absolute = float(np.max(error))
                    p = float(expit(z / t))
                    data = reference - 0.002 * theta
                    omission = float(np.max(np.abs(data * t - data)))
                    reversal = float(np.max(np.abs(2 * data)))
                    stationary = float(reference @ reference) < 1e-24
                    decrease = objective(theta - 1e-4 * reference) < objective(theta)
                    update_error = float(
                        np.max(
                            np.abs(adapter.sparse.parameters(changed) - (theta - 0.01 * reference))
                        )
                    )
                    row = dict(
                        unit_id=f"{support}:T{t}:z{z}:y{y}",
                        family="constructed_derivatives",
                        arm="brier",
                        temperature=t,
                        logit=z,
                        target=y,
                        support=support,
                        absolute_error=absolute,
                        relative_error=relative,
                        update_error=update_error,
                        stationary=stationary,
                        saturated=p * (1 - p) < 1e-7 or float(np.max(np.abs(data))) < 1e-7,
                        directional_decrease=decrease,
                        basis_support=int(np.count_nonzero(matrix)),
                        negative_controls=dict(
                            omit_temperature_rejected=omission > 1e-7,
                            reverse_gradient_rejected=reversal > 1e-7,
                        ),
                        passed=absolute <= 1e-7
                        and relative <= 1e-5
                        and update_error <= 1e-12
                        and (stationary or decrease),
                    )
                    rows.append(row)
        print(
            f"[exp8461] derivative_completed={len(rows)} derivative_pending={90 - len(rows)}",
            flush=True,
        )
    return rows


def transition_controls(head: Json) -> list[Json]:
    """Constructed negative controls prove that future labels cannot enter the update adapter."""
    cfg = adapter.config(head)
    initial = adapter.bind(adapter.legacy.initial(cfg), cfg)
    sealed = adapter.legacy.seal(initial, cfg, 0, adapter.legacy.questions(0))
    labels = adapter.legacy.feedback(sealed)
    rows = []
    for name, state, block, targets, error in [
        ("future_label", sealed, 0, dict(labels, future_label=1), "release_roster"),
        ("early_release", initial, 0, labels, "unsealed_release"),
        ("future_block", sealed, 1, labels, "unsealed_release"),
    ]:
        try:
            adapter.release(state, cfg, block, targets)
            observed = None
        except ValueError as failure:
            observed = str(failure)
        rows.append(
            dict(
                name=name,
                expected_error=error,
                observed_error=observed,
                passed=observed == error,
                before_state_hash=adapter.legacy.canonical_hash(state),
            )
        )
    return rows
