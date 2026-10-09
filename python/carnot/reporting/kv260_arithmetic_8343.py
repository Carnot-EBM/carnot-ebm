"""REQ-REPORT-8343: constructed arithmetic does not measure a durable service.

SciPy supplies an independent basis; centered loss differences check gradients.
Only the established pure cubic kernel is reused, with no prior receipt credit.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.interpolate import BSpline

from carnot.experiment_7425_v651_spline_prototype import cubic_basis

Json = dict[str, Any]
KNOTS = [0.0] * 4 + [0.2, 0.4, 0.6, 0.8] + [1.0] * 4
ARMS = ["dense", "active"]
SCIENCE: Json = dict(
    warmups=1,
    repetitions=5,
    vectors=128,
    degree=3,
    knots=KNOTS,
    learning_rate=0.01,
    gradient_norm_cap=1,
    coordinate_bounds=[-4, 4],
    frozen_slope=1,
    frozen_intercept=0,
    temperature=1,
    holistic_input=0,
    decay=0,
    k_max=5,
    model_load=False,
    targets="(i-1)%2",
)
FEATURES = [[((17 * i + 13 * j) % 101) / 100 for j in range(4)] for i in range(1, 129)]
TARGETS = [i % 2 for i in range(128)]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counts so short and long benchmarks remain observable."""
    print(f"[exp8343] phase={phase} completed={completed} pending={pending}", flush=True)


def basis(x: list[float]) -> list[float]:
    """Evaluate the uncached kernel so basis cost cannot disappear into a warm cache."""
    return [float(v) for feature in x for v in cubic_basis(feature, KNOTS).values]


def reference_basis(x: list[float]) -> list[float]:
    """SciPy's implementation checks knot order and endpoints independently."""
    return list(np.asarray(BSpline(KNOTS, np.eye(8), 3)(x)).reshape(-1).astype(float))


def action(probability: float) -> str:
    """Threshold ties require escalation under the frozen V717 decision rule."""
    return "accept" if probability < 0.25 else "reject" if probability > 0.75 else "escalate"


def audit() -> Json:
    """Validate before timing; finite differences do not call the measured update."""
    basis_delta = gradient_delta = 0.0
    coefficients = np.linspace(-0.3, 0.3, 32)
    for x, y in zip(FEATURES + [[0.0, 0.2, 0.8, 1.0]], TARGETS + [1], strict=True):
        d = np.asarray(reference_basis(x))
        basis_delta = max(basis_delta, float(np.max(np.abs(d - basis(x)))))
        eta = float(coefficients @ d)
        gradient = (1 / (1 + math.exp(-eta)) - y) * d
        for coordinate in range(32):
            shift = 1e-6 * d[coordinate]
            difference = (
                (np.logaddexp(0, eta + shift) - y * (eta + shift))
                - (np.logaddexp(0, eta - shift) - y * (eta - shift))
            ) / 2e-6
            gradient_delta = max(gradient_delta, abs(float(difference - gradient[coordinate])))
    return dict(
        passed=basis_delta <= 1e-10 and gradient_delta < 1e-7,
        basis_max_abs_delta=basis_delta,
        finite_difference_max_abs_delta=gradient_delta,
    )


def reference(arm: str) -> Json:
    """Use vectorized SciPy/NumPy updates rather than the timed coordinate loop.

    This catches mistakes in normalization, clipping and gradient application,
    even when the measured dense and sparse arms share the same mistake.
    """
    coefficients = np.zeros(32)
    probabilities, actions = [], []
    touches = 0
    matrix = BSpline(KNOTS, np.eye(8), 3)(np.asarray(FEATURES)).reshape(128, 32)
    for d, target in zip(matrix, TARGETS, strict=True):
        probability = float(1 / (1 + np.exp(-(coefficients @ d))))
        gradient = (probability - target) * d
        norm = float(np.linalg.norm(gradient))
        coefficients = np.clip(coefficients - 0.01 * gradient / max(1.0, norm), -4, 4)
        probabilities.append(probability)
        actions.append(
            "accept" if probability < 0.25 else "reject" if probability > 0.75 else "escalate"
        )
        touches += 32 if arm == "dense" else int(np.count_nonzero(gradient))
    return dict(
        probabilities=probabilities,
        actions=actions,
        coefficients=coefficients.tolist(),
        coefficient_touches=touches,
    )


def run(arm: str, repetition: int) -> Json:
    """Panels are disjoint; clock and loop overhead remain in the CPU wall cost."""
    coefficients = [0.0] * 32
    probabilities: list[float] = []
    actions: list[str] = []
    spans = dict(basis_evaluation=0, gradient=0, coefficient_update=0)
    touches = 0
    start = time.monotonic_ns()
    for index, (x, y) in enumerate(zip(FEATURES, TARGETS, strict=True)):
        before = time.monotonic_ns()
        d = basis(x)
        spans["basis_evaluation"] += time.monotonic_ns() - before
        before = time.monotonic_ns()
        probability = 1 / (1 + math.exp(-sum(c * v for c, v in zip(coefficients, d, strict=True))))
        gradient = [(probability - y) * v for v in d]
        norm = math.sqrt(sum(g * g for g in gradient))
        scale = min(1.0, 1.0 / norm) if norm else 1.0
        probabilities.append(probability)
        actions.append(action(probability))
        spans["gradient"] += time.monotonic_ns() - before
        before = time.monotonic_ns()
        for coordinate, g in enumerate(gradient):
            if arm == "dense" or g != 0:
                coefficients[coordinate] = min(
                    4.0, max(-4.0, coefficients[coordinate] - 0.01 * g * scale)
                )
                touches += 1
        spans["coefficient_update"] += time.monotonic_ns() - before
        if (index + 1) % 32 == 0:
            progress(f"benchmark_{arm}_{repetition}", index + 1, 127 - index)
    end = time.monotonic_ns()
    return dict(
        arm=arm,
        repetition=repetition,
        branch="arithmetic",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        wall_ns=end - start,
        operation_ns=spans,
        coefficient_touches=touches,
        probabilities=probabilities,
        actions=actions,
        coefficients=coefficients,
    )


def verify(row: Json) -> None:
    """Recompute numerical meaning so a rehashed primitive cannot change decisions."""
    expected = reference(row["arm"])
    if (
        row["arm"] not in ARMS
        or not all(math.isfinite(v) for v in row["probabilities"] + row["coefficients"])
        or row["actions"] != expected["actions"]
        or len(row["probabilities"]) != 128
        or max(
            abs(a - b) for a, b in zip(row["probabilities"], expected["probabilities"], strict=True)
        )
        > 1e-10
        or len(row["coefficients"]) != 32
        or max(
            abs(a - b) for a, b in zip(row["coefficients"], expected["coefficients"], strict=True)
        )
        > 1e-10
        or row["coefficient_touches"] != expected["coefficient_touches"]
    ):
        raise ValueError("arithmetic_semantics")
    spans = row["operation_ns"]
    if (
        set(spans) != {"basis_evaluation", "gradient", "coefficient_update"}
        or any(type(v) is not int or v < 0 for v in spans.values())
        or row["wall_ns"] != row["ended_monotonic_ns"] - row["started_monotonic_ns"]
        or row["wall_ns"] <= 0
        or sum(spans.values()) > row["wall_ns"]
    ):
        raise ValueError("arithmetic_clock")


def measure() -> list[Json]:
    """Alternate paired order; repeated clocks never increase independent sources."""
    rows = []
    for repetition in range(-1, 5):
        for arm in ARMS if repetition % 2 == 0 else list(reversed(ARMS)):
            progress(f"before_benchmark_{arm}_{repetition}")
            row = run(arm, repetition)
            verify(row)
            if repetition >= 0:
                rows.append(row)
            progress(f"after_benchmark_{arm}_{repetition}")
    return rows
