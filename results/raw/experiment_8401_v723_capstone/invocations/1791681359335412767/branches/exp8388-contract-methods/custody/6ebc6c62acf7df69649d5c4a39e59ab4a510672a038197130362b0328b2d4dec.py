"""REQ-KAN-8334: small logistic heads share public inputs and a fixed budget.

The spline score is also a sigmoid classifier. Energy notation introduces no
extra information or independent comparison arm.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.verify import local_update_isolation_8306 as kernel

Json = dict[str, Any]
Array = NDArray[np.float64]
ARMS = ["spline34", "RBF34", "linear6", "scalar2"]
COUNTS = dict(spline34=34, RBF34=34, linear6=6, scalar2=2)
CONFIG = dict(seed=7178308, steps=400, learning_rate=0.01, l2=0.001)


def geometry(local: Array, ids: list[str]) -> Json:
    """Fit-only distances keep labels and later rows out of radial geometry."""
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    if len(np.unique(local, axis=0)) < 32:
        raise ValueError("insufficient_geometry")
    chosen = [order[0]]
    for _ in range(31):
        distance = np.min(np.sum((local[:, None] - local[chosen]) ** 2, axis=2), axis=1)
        chosen.append(min(order, key=lambda i: (-float(distance[i]), ids[i])))
    centers = local[chosen]
    distances = np.sqrt(np.sum((centers[:, None] - centers) ** 2, axis=2))
    return dict(
        centers=centers.tolist(),
        center_ids=[ids[i] for i in chosen],
        width=float(np.median(distances[distances > 0])),
    )


def matrix(arm: str, x: Array, g: Json) -> Array:
    """Only the same holistic logit and four local features enter every head."""
    shared = np.column_stack((x[:, 0], np.ones(len(x))))
    if arm == "spline34":
        return np.asarray([kernel.design(row.tolist()) for row in x], dtype=float)
    if arm == "RBF34":
        squared = np.sum((x[:, None, 1:] - np.asarray(g["centers"])) ** 2, axis=2)
        return np.column_stack((shared, np.exp(-squared / (2 * g["width"] ** 2))))
    if arm == "linear6":
        return np.column_stack((shared, x[:, 1:]))
    if arm == "scalar2":
        return shared
    raise ValueError("arm")


def project(c: Array) -> Array:
    """Bounds limit coefficient growth without changing the frozen objective."""
    result = np.clip(c, -4.0, 4.0)
    result[0] = np.clip(result[0], 0.0, 2.0)
    return np.asarray(result, dtype=float)


def objective(c: Array, phi: Array, y: Array) -> tuple[float, Array]:
    """Penalty excludes intercept and shrinks the slope toward its initial one."""
    z = phi @ c
    penalized = c.copy()
    penalized[0] -= 1
    penalized[1] = 0
    loss = np.mean(np.logaddexp(0, z) - y * z) + 0.001 * (penalized @ penalized)
    gradient = phi.T @ (expit(z) - y) / len(y) + 0.002 * penalized
    return float(loss), np.asarray(gradient, dtype=float)


def diagnostic(c: Array, phi: Array, y: Array) -> Json:
    """Residuals reveal when a fixed-budget comparison has not converged."""
    loss, gradient = objective(c, phi, y)
    return dict(
        loss=loss,
        gradient=gradient.tolist(),
        projected_gradient_residual=float(np.max(np.abs(c - project(c - gradient)))),
        saturated_logit_count=int(np.sum(np.abs(phi @ c) >= 30)),
    )


def fit(phi: Array, y: Array) -> Json:
    """Full-batch projected descent runs all 400 steps without arm-specific tuning."""
    c = np.zeros(phi.shape[1])
    c[0] = 1.0
    start = diagnostic(c, phi, y)
    losses = [start["loss"]]
    for step in range(400):
        _, grad = objective(c, phi, y)
        c = project(c - 0.01 * grad)
        losses.append(objective(c, phi, y)[0])
        if (step + 1) % 100 == 0:
            kernel.progress("static_fit", step + 1, 400 - step - 1)
    return dict(
        coefficients=c.tolist(),
        initial=start,
        final=diagnostic(c, phi, y),
        loss_rows=losses,
        steps=400,
    )


def action(p: float | None) -> str:
    """Fixed conservative thresholds escalate ties; they are not cost-optimal."""
    return "escalate" if p is None else kernel.action(p)


def cost(selected: str, y: int | None) -> float:
    """Every unavailable intended slot keeps its half-unit escalation cost."""
    return 0.5 if selected == "escalate" or y is None else float((selected == "reject") != bool(y))


def predict(head: Json, public: Json) -> float | None:
    """A feature-only allowlist excludes source identity and target metadata."""
    if set(public) != {"features"}:
        raise ValueError("prediction_input")
    x = public["features"]
    if x is None:
        return None
    kernel.design(x)
    phi = matrix(head["arm"], np.asarray([x], dtype=float), head["geometry"])
    return float(expit((phi @ np.asarray(head["coefficients"]))[0] / head["temperature"]))


def numeric_audit() -> Json:
    """Independent spline values and centered differences qualify pure arithmetic."""
    values = np.unique(np.r_[np.linspace(0, 1, 41), 0.2 - 1e-9, 0.2 + 1e-9])
    expected = BSpline.design_matrix(values, kernel.KNOTS, 3).toarray()
    actual = np.asarray([kernel.local_basis(float(v)) for v in values])
    basis_error = float(np.max(np.abs(expected - actual)))
    rng = np.random.default_rng(7178308)
    x = rng.random((16, 5))
    x[:, 0] -= 0.5
    phi = matrix("spline34", x, {})
    c = rng.normal(0, 0.2, 34)
    y = np.asarray([i % 2 for i in range(16)], dtype=float)
    _, analytic = objective(c, phi, y)
    finite = []
    for i in range(34):
        offset = np.zeros(34)
        offset[i] = 1e-6
        finite.append((objective(c + offset, phi, y)[0] - objective(c - offset, phi, y)[0]) / 2e-6)
    gradient_error = float(np.max(np.abs(analytic - finite)))
    return dict(
        basis_error=basis_error,
        gradient_error=gradient_error,
        partition_error=float(np.max(np.abs(actual.sum(axis=1) - 1))),
        passed=basis_error < 1e-12 and gradient_error < 1e-7,
    )


def optimizer_control() -> Json:
    """Constructed fit-only examples prove the fixed budget can change an action."""
    x = np.asarray([[-1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    phi = matrix("spline34", x, {})
    fitted = fit(phi, np.asarray([0.0, 1.0]))
    initial_action = action(float(expit(-1.0)))
    final_action = action(float(expit((phi @ np.asarray(fitted["coefficients"]))[0])))
    return dict(
        fitted,
        initial_action=initial_action,
        final_action=final_action,
        scope="constructed_fit_only_optimizer_control",
        verdict_class="circular_positive",
        passed=fitted["final"]["loss"] < fitted["initial"]["loss"]
        and initial_action != final_action,
    )


def train(bundle: Json) -> Json:
    """Disjoint source roles own training, temperatures and comparator selection."""
    usable = [
        (p, t)
        for p, t in zip(bundle["predictors"]["fit"], bundle["evaluators"]["fit"], strict=True)
        if p["x"] is not None and t["y"] in (0, 1)
    ]
    x = np.asarray([[p["x"][0], *p["x"][12:16]] for p, _ in usable], dtype=float)
    y = np.asarray([t["y"] for _, t in usable], dtype=float)
    g = geometry(x[:, 1:], [p["source_cluster_id"] for p, _ in usable])
    heads, calibration, rows = [], [], []
    for arm in ARMS:
        kernel.progress("before_benchmark_" + arm)
        fitted = fit(matrix(arm, x, g), y)
        head = dict(fitted, arm=arm, geometry=g, temperature=1.0)
        pairs = list(
            zip(
                bundle["predictors"]["calibration"],
                bundle["evaluators"]["calibration"],
                strict=True,
            )
        )
        choices = []
        for temperature in [0.5, 1.0, 2.0]:
            head["temperature"] = temperature
            losses = []
            for p, target in pairs:
                prob = predict(
                    head, {"features": None if p["x"] is None else [p["x"][0], *p["x"][12:16]]}
                )
                if prob is not None and target["y"] in (0, 1):
                    losses.append(
                        float(
                            -target["y"] * np.log(max(prob, 1e-300))
                            - (1 - target["y"]) * np.log(max(1 - prob, 1e-300))
                        )
                    )
            score = float(np.mean(losses))
            choices.append((score, abs(temperature - 1), temperature))
            calibration.append(
                dict(arm=arm, temperature=temperature, mean_log_loss=score, usable=len(losses))
            )
        head["temperature"] = min(choices)[2]
        heads.append(head)
        for role in ["calibration", "comparator_selection"]:
            for p, target in zip(
                bundle["predictors"][role], bundle["evaluators"][role], strict=True
            ):
                prob = predict(
                    head, {"features": None if p["x"] is None else [p["x"][0], *p["x"][12:16]]}
                )
                selected = action(prob) if target["y"] is not None else "escalate"
                rows.append(
                    dict(
                        unit_id=p["unit_id"],
                        source_cluster_id=p["source_cluster_id"],
                        role=role,
                        slot=p["slot"],
                        arm=arm,
                        status="completed"
                        if prob is not None and target["y"] is not None
                        else "excluded",
                        p=prob,
                        y=target["y"],
                        action=selected,
                        numerator=cost(selected, target["y"]),
                        denominator=1,
                        brier=None
                        if prob is None or target["y"] is None
                        else (prob - target["y"]) ** 2,
                    )
                )
        kernel.progress("after_benchmark_" + arm, len(heads), len(ARMS) - len(heads))
    scores = {}
    for arm in ARMS[1:]:
        subset = [r for r in rows if r["arm"] == arm and r["role"] == "comparator_selection"]
        scores[arm] = (
            sum(r["numerator"] for r in subset) / 32,
            float(np.mean([r["brier"] for r in subset if r["brier"] is not None])),
            ARMS.index(arm),
        )
    holistic = []
    for role in ["calibration", "comparator_selection"]:
        for p, target in zip(bundle["predictors"][role], bundle["evaluators"][role], strict=True):
            prob = None if p["x"] is None else float(expit(p["x"][0]))
            selected = action(prob) if target["y"] is not None else "escalate"
            holistic.append(
                dict(
                    unit_id=p["unit_id"],
                    role=role,
                    arm="frozen_holistic_probability",
                    p=prob,
                    y=target["y"],
                    action=selected,
                    numerator=cost(selected, target["y"]),
                    denominator=1,
                    status="completed"
                    if prob is not None and target["y"] is not None
                    else "excluded",
                )
            )
    z = matrix("spline34", x, g) @ np.asarray(heads[0]["coefficients"]) / heads[0]["temperature"]
    energy = np.exp(-np.logaddexp(0, -z))
    return dict(
        heads=heads,
        selected_comparator=min(scores, key=lambda arm: scores[arm]),
        comparator_scores=scores,
        calibration_rows=calibration,
        rows=rows + holistic,
        sigmoid_equivalence_error=float(np.max(np.abs(energy - expit(z)))),
    )
