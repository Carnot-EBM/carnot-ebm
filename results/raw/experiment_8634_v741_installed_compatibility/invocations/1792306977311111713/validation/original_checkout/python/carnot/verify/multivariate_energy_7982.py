"""REQ-VERIFY-7982: small source-aware heads use exact binary probabilities.

The energy network takes nine public numbers and one candidate label. Both
labels are evaluated, so no sampler or pretrained model is needed. Comparing
these heads measures architecture and training, rather than normalization.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.verify import qwen_energy_calibration_7972 as scalar

Json = dict[str, Any]
Array = NDArray[np.float64]
ARMS = ("gibbs", "logistic", "quadratic", "mlp", "spline")
COUNTS = dict(gibbs=97, logistic=10, quadratic=55, mlp=177, spline=63)
SEEDS = (69201, 69202, 69203)
CONFIG = dict(
    primary="gibbs",
    optimizer="Adam",
    steps=200,
    learning_rate=0.01,
    l2=0.001,
    seeds=list(SEEDS),
    temperatures=list(scalar.TEMPERATURES),
    temperature_objective="tune_brier",
    input_dim=9,
    spline_quantiles=[0.25, 0.5, 0.75],
    seed_reduction="probability_mean",
)


def validate_data(data: Json) -> None:
    """Only the explicit numeric view can reach fitting; unknowns stay unknown."""
    if set(data) != {"fit", "tune", "policy_design"}:
        raise ValueError("role_roster")
    seen, clusters = set(), {}
    for role, rows in data.items():
        for r in rows:
            if set(r) != {"family_id", "source_cluster_id", "q", "features", "y", "status"}:
                raise ValueError("predictor_fields")
            if r["family_id"] in seen or clusters.get(r["source_cluster_id"], role) != role:
                raise ValueError("cross_role")
            seen.add(r["family_id"])
            clusters[r["source_cluster_id"]] = role
            if r["q"] is not None and (
                type(r["q"]) not in (int, float) or not np.isfinite(r["q"]) or not 0 <= r["q"] <= 1
            ):
                raise ValueError("probability")
            if r["y"] is not None and (type(r["y"]) is not int or r["y"] not in (0, 1)):
                raise ValueError("label")
            if r["features"] is not None and (
                len(r["features"]) != 8
                or not all(type(v) in (int, float) and np.isfinite(v) for v in r["features"])
            ):
                raise ValueError("features")


def valid(rows: list[Json]) -> list[Json]:
    """Missing public evidence cannot become a fitted example by imputation."""
    return [
        r for r in rows if r["features"] is not None and r["q"] is not None and r["y"] is not None
    ]


def inputs(rows: list[Json]) -> Array:
    """IDs and evaluator metadata are omitted from the nine predictor numbers."""
    q = np.clip(np.asarray([r["q"] for r in rows]), 1e-4, 1 - 1e-4)
    return np.column_stack(
        (np.log(q / (1 - q)), np.asarray([r["features"] for r in rows]).reshape(-1, 8))
    )


def knots(x: Array) -> list[list[float]]:
    """Fit quantiles fix cubic bases before any tune or decision target access."""
    return [
        (
            [float(x[:, i].min() - 1e-8)] * 4
            + np.quantile(x[:, i], [0.25, 0.5, 0.75]).tolist()
            + [float(x[:, i].max() + 1e-8)] * 4
        )
        for i in range(9)
    ]


def design(arm: str, x: Array, fixed: list[list[float]]) -> Array:
    """Quadratic terms and additive cubic bases use the same frozen inputs."""
    if arm == "spline":
        return np.column_stack(
            [
                BSpline.design_matrix(np.clip(x[:, i], k[0], k[-1]), k, 3).toarray()
                for i, k in enumerate(fixed)
            ]
        )
    linear = np.column_stack((x, np.ones(len(x))))
    if arm == "quadratic":
        return np.column_stack(
            (linear, *[x[:, i] * x[:, j] for i in range(9) for j in range(i, 9)])
        )
    return linear


def energies(theta: Array, x: Array) -> Array:
    """The shared scalar energy explicitly scores label zero and label one."""
    w, b, v = theta[:80].reshape(10, 8), theta[80:88], theta[88:96]
    return np.column_stack(
        [np.tanh(np.column_stack((x, np.full(len(x), y))) @ w + b) @ v + theta[96] for y in (0, 1)]
    )


def logits_jacobian(
    arm: str, theta: Array, x: Array, fixed: list[list[float]]
) -> tuple[Array, Array]:
    """Analytical derivatives reuse Exp7972's two-label gradient pattern."""
    if arm in ("logistic", "quadratic", "spline"):
        matrix = design(arm, x, fixed)
        offset = x[:, 0] if arm == "spline" else 0
        return matrix @ theta + offset, matrix
    if arm == "mlp":
        w, b, v = theta[:144].reshape(9, 16), theta[144:160], theta[160:176]
        h = expit(x @ w + b)
        d = h * (1 - h) * v
        jac = np.column_stack(
            ((x[:, :, None] * d[:, None, :]).reshape(len(x), 144), d, h, np.ones(len(x)))
        )
        return h @ v + theta[176], jac
    if arm != "gibbs":
        raise ValueError("arm")
    w, b, v = theta[:80].reshape(10, 8), theta[80:88], theta[88:96]
    x0, x1 = (np.column_stack((x, np.full(len(x), y))) for y in (0, 1))
    h0, h1 = np.tanh(x0 @ w + b), np.tanh(x1 @ w + b)
    d0, d1 = (1 - h0 * h0) * v, (1 - h1 * h1) * v
    dw = x0[:, :, None] * d0[:, None, :] - x1[:, :, None] * d1[:, None, :]
    return (h0 - h1) @ v, np.column_stack(
        (dw.reshape(len(x), 80), d0 - d1, h0 - h1, np.zeros(len(x)))
    )


def predict(head: Json, x: Array) -> Array:
    """Stable exact two-label normalization avoids numerical overflow."""
    if not len(x):
        return np.empty(0)
    theta = np.asarray(head["parameters"], dtype=float)
    if head["arm"] == "gibbs":
        z = -energies(theta, x) / head["temperature"]
        weights = np.exp(z - z.max(axis=1, keepdims=True))
        return np.asarray(weights[:, 1] / weights.sum(axis=1), dtype=np.float64)
    z, _ = logits_jacobian(head["arm"], theta, x, head["knots"])
    return np.asarray(expit(z / head["temperature"]), dtype=np.float64)


def sigmoid_conversion(head: Json, x: Array) -> Array:
    """This no-fit classical classifier is algebraically identical to Gibbs."""
    e = energies(np.asarray(head["parameters"]), x)
    return np.asarray(expit((e[:, 0] - e[:, 1]) / head["temperature"]), dtype=np.float64)


def gradient_check(arm: str, theta: Array, x: Array, fixed: list[list[float]]) -> Json:
    """Check all coefficients at a fixed fit-only probe, including zero derivatives."""
    _, jac = logits_jacobian(arm, theta, x, fixed)
    errors = []
    for i in range(len(theta)):
        delta = np.zeros(len(theta))
        delta[i] = 1e-6
        numerical = (
            logits_jacobian(arm, theta + delta, x, fixed)[0]
            - logits_jacobian(arm, theta - delta, x, fixed)[0]
        ) / 2e-6
        errors.append(float(np.max(np.abs(numerical - jac[:, i]))))
    return dict(
        arm=arm,
        coefficients_checked=len(theta),
        epsilon=1e-6,
        max_absolute_error=max(errors),
        tolerance=1e-7,
        passed=max(errors) <= 1e-7,
    )


def fit(data: Json) -> Json:
    """Fit labels update coefficients; tune labels only select a frozen temperature."""
    validate_data(data)
    fitting, tuning = valid(data["fit"]), valid(data["tune"])
    fs, ts = scalar.support(fitting, 128, 16), scalar.support(tuning, 32, 4)
    if not fs["passed"] or not ts["passed"]:
        raise ValueError("support_floor")
    x, tx = inputs(fitting), inputs(tuning)
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    x, tx = (x - means) / scales, (tx - means) / scales
    y, ty = (np.asarray([r["y"] for r in rows], dtype=float) for rows in (fitting, tuning))
    fixed, heads, checks, summary = knots(x), {}, [], {}
    for arm in ARMS:
        heads[arm] = []
        for seed in SEEDS:
            began = time.monotonic()
            count = COUNTS[arm]
            theta = np.random.default_rng(seed).normal(0, 0.2, count)
            initial = theta.copy()
            checks.append(dict(gradient_check(arm, theta, x[:3], fixed), seed=seed))
            first = scalar.bce(np.asarray(expit(logits_jacobian(arm, theta, x, fixed)[0])), y)
            moment, variance = np.zeros(count), np.zeros(count)
            for step in range(1, 201):
                z, jac = logits_jacobian(arm, theta, x, fixed)
                gradient = jac.T @ (expit(z) - y) / len(x) + 0.002 * theta
                moment, variance = (
                    0.9 * moment + 0.1 * gradient,
                    0.999 * variance + 0.001 * gradient * gradient,
                )
                theta -= (
                    0.01
                    * (moment / (1 - 0.9**step))
                    / (np.sqrt(variance / (1 - 0.999**step)) + 1e-8)
                )
                if step % 50 == 0:
                    print(f"[exp7982] fit arm={arm} seed={seed} steps={step}/200", flush=True)
            zt, _ = logits_jacobian(arm, theta, tx, fixed)
            losses = [float(np.mean((expit(zt / t) - ty) ** 2)) for t in scalar.TEMPERATURES]
            head = dict(
                arm=arm,
                seed=seed,
                parameters=theta.tolist(),
                initial_parameters=initial.tolist(),
                knots=fixed,
                temperature=scalar.TEMPERATURES[int(np.argmin(losses))],
                tune_brier_choices=losses,
                parameter_count=count,
                optimizer_steps=200,
                changed_coefficients=int(np.count_nonzero(theta != initial)),
                coefficient_touches=count * 200,
                local_basis_touches_per_step=int(np.count_nonzero(design(arm, x, fixed)))
                if arm == "spline"
                else None,
                initial_bce=first,
                final_bce=scalar.bce(
                    np.asarray(expit(logits_jacobian(arm, theta, x, fixed)[0])), y
                ),
                duration_s=time.monotonic() - began,
            )
            heads[arm].append(head)
        fp, tp = (np.mean([predict(h, a) for h in heads[arm]], axis=0) for a in (x, tx))
        summary[arm] = dict(
            fit_brier=float(np.mean((fp - y) ** 2)),
            tune_brier=float(np.mean((tp - ty) ** 2)),
            parameter_count_per_seed=COUNTS[arm],
            descriptive=True,
        )
    parity = max(
        float(np.max(np.abs(predict(h, a) - sigmoid_conversion(h, a))))
        for h in heads["gibbs"]
        for a in (x, tx)
    )
    return dict(
        heads=heads,
        fit_support=fs,
        tune_support=ts,
        gradient_checks=checks,
        feature_normalization=dict(
            means=means.tolist(), scales=scales.tolist(), frozen_from="fit", clip_q=[1e-4, 1 - 1e-4]
        ),
        optimizer_work=dict(
            total_steps=3000,
            per_head_steps=200,
            coefficient_touches=sum(COUNTS.values()) * 600,
            tune_temperature_scores=15 * 17,
            pretrained_model_calls=0,
        ),
        summary=summary,
        parity_max_error=parity,
    )


def predictions(measured: Json, data: Json) -> list[Json]:
    """Freeze every probability without consulting policy-design targets."""
    normalization = measured["feature_normalization"]
    result = []
    for role, rows in data.items():
        for r in rows:
            usable = r["q"] is not None and r["features"] is not None
            x = (
                ((inputs([r]) - normalization["means"]) / normalization["scales"])
                if usable
                else np.empty((0, 9))
            )
            for arm in ARMS:
                ps = []
                for h in measured["heads"][arm]:
                    p = float(predict(h, x)[0]) if usable else None
                    ps.append(p)
                    result.append(
                        dict(
                            family_id=r["family_id"],
                            source_cluster_id=r["source_cluster_id"],
                            role=role,
                            arm=arm,
                            seed=h["seed"],
                            p=p,
                            status="completed" if usable else "excluded",
                        )
                    )
                result.append(
                    dict(
                        family_id=r["family_id"],
                        source_cluster_id=r["source_cluster_id"],
                        role=role,
                        arm=arm,
                        seed="mean",
                        p=float(np.mean(ps)) if usable else None,
                        status="completed" if usable else "excluded",
                    )
                )
            for h in measured["heads"]["gibbs"]:
                result.append(
                    dict(
                        family_id=r["family_id"],
                        source_cluster_id=r["source_cluster_id"],
                        role=role,
                        arm="exact_sigmoid_conversion",
                        seed=h["seed"],
                        p=float(sigmoid_conversion(h, x)[0]) if usable else None,
                        status="completed" if usable else "excluded",
                    )
                )
            result.append(
                dict(
                    family_id=r["family_id"],
                    source_cluster_id=r["source_cluster_id"],
                    role=role,
                    arm="exact_sigmoid_conversion",
                    seed="mean",
                    p=float(
                        np.mean([sigmoid_conversion(h, x)[0] for h in measured["heads"]["gibbs"]])
                    )
                    if usable
                    else None,
                    status="completed" if usable else "excluded",
                )
            )
    return result
