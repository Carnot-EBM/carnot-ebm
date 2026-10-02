"""REQ-VERIFY-7996: local spline updates with exact two-label probabilities.

Only small coefficients are trained. A shared decay multiplier applies L2 to
all logical coefficients while physical data-gradient writes stay local.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.verify import multivariate_energy_7982 as historical
from carnot.verify import qwen_energy_calibration_7972 as scalar

Json = dict[str, Any]
Array = NDArray[np.float64]
SEEDS = (17, 29, 43, 71, 101)
KNOTS = [0.0] * 4 + [i / 9 for i in range(1, 9)] + [1.0] * 4
COUNTS = dict(spline=109, logistic=10, mlp=89)
CONFIG = dict(
    seeds=list(SEEDS),
    primary_seed=17,
    steps=200,
    learning_rate=0.01,
    l2=0.001,
    optimizer="batch_gradient_descent",
    temperatures=[0.5, 1.0, 2.0],
    temperature_objective="historical_tune_log_loss",
    inputs=9,
    boundary_multiplicity=4,
    internal_knots=8,
    coefficients_per_input=12,
)


def validate_data(data: Json) -> None:
    """Reject extra roles so reserved labels cannot reach any optimizer."""
    if set(data) != {"fit", "tune"}:
        raise ValueError("role_roster")
    historical.validate_data(dict(data, policy_design=[]))
    if (
        not scalar.support(usable(data["fit"]), 128, 16)["passed"]
        or not scalar.support(usable(data["tune"]), 32, 4)["passed"]
    ):
        raise ValueError("support_floor")


def usable(rows: list[Json]) -> list[Json]:
    """Unknown judgments and targets stay excluded rather than imputed."""
    return historical.valid(rows)


def inputs(rows: list[Json]) -> Array:
    """Only q and the eight public lexical numbers enter every arm."""
    return np.asarray([[r["q"], *r["features"]] for r in rows], dtype=float)


def fit_scaler(x: Array) -> Json:
    """Fit-only bounds prevent tune values from changing the input geometry."""
    return dict(minimum=x.min(axis=0).tolist(), maximum=x.max(axis=0).tolist())


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


def logits_jacobian(head: Json, raw: Array) -> tuple[Array, Array]:
    """Analytical derivatives make the same objective testable for all controls."""
    x, _ = scale(raw, head["scaler"])
    theta = parameters(head)
    offset = np.log(np.clip(raw[:, 0], 1e-4, 1 - 1e-4) / (1 - np.clip(raw[:, 0], 1e-4, 1 - 1e-4)))
    arm = head["arm"]
    if arm in ("spline", "sigmoid_equivalent", "logistic"):
        matrix = basis(x) if arm != "logistic" else np.column_stack((x, np.ones(len(x))))
        return np.asarray(offset + matrix @ theta), matrix
    if arm != "mlp":
        raise ValueError("arm")
    w, b, v = theta[:72].reshape(9, 8), theta[72:80], theta[80:88]
    hidden = np.tanh(x @ w + b)
    d = (1 - hidden**2) * v
    jac = np.column_stack(
        ((x[:, :, None] * d[:, None, :]).reshape(len(x), 72), d, hidden, np.ones(len(x)))
    )
    return np.asarray(offset + hidden @ v + theta[88]), jac


def sigmoid_predict(head: Json, x: Array) -> Array:
    """Identical coefficients require no extra fit when written as a sigmoid."""
    return np.asarray(expit(logits_jacobian(head, x)[0] / head["temperature"]), dtype=float)


def predict(head: Json, x: Array) -> Array:
    """E0=0 and E1=-z allow exact stable normalization without sampling."""
    if len(x) == 0:
        return np.empty(0)
    z = logits_jacobian(head, x)[0] / head["temperature"]
    energies = np.column_stack((np.zeros(len(x)), -z))
    weights = np.exp(-energies - np.max(-energies, axis=1, keepdims=True))
    return np.asarray(weights[:, 1] / weights.sum(axis=1), dtype=float)


def objective(head: Json, x: Array, y: Array) -> float:
    """The fit objective includes every coefficient's stated L2 penalty."""
    z, _ = logits_jacobian(head, x)
    return float(np.mean(np.logaddexp(0, z) - y * z) + 0.001 * np.sum(parameters(head) ** 2))


def dense_gradient(head: Json, x: Array, y: int) -> Array:
    """A reference gradient includes temperature and full logical L2 decay."""
    z, jac = logits_jacobian(head, x[None, :])
    t = head["temperature"]
    return np.asarray(jac[0] * (expit(z[0] / t) - y) / t + 0.002 * parameters(head))


def sparse_gradient(head: Json, x: Array, y: int) -> tuple[Array, Array]:
    """Return only active data derivatives; global L2 is a separate scalar decay."""
    scaled, _ = scale(x[None, :], head["scaler"])
    indices, values = [108], [1.0]
    for j in range(9):
        row = BSpline.design_matrix(scaled[:, j], KNOTS, 3)
        for k, v in zip(row.indices, row.data, strict=True):
            if v != 0:
                indices.append(j * 12 + int(k))
                values.append(float(v))
    ids, weights = np.asarray(indices, dtype=np.int64), np.asarray(values)
    q = np.clip(x[0], 1e-4, 1 - 1e-4)
    local = np.asarray([head["parameters"][int(i)] for i in ids]) * head["decay_scale"]
    z = np.log(q / (1 - q)) + weights @ local
    return ids, np.asarray(weights * (expit(z / head["temperature"]) - y) / head["temperature"])


def update(head: Json, x: Array, y: int, learning_rate: float) -> tuple[Json, Json]:
    """One CPU update equals dense descent but writes at most 37 coefficients."""
    if head["arm"] != "spline" or y not in (0, 1) or not 0 <= learning_rate <= 1:
        raise ValueError("update_contract")
    began = time.monotonic()
    result = copy.deepcopy(head)
    ids, gradient = sparse_gradient(head, x, y)
    if learning_rate:
        new_scale = head["decay_scale"] * (1 - 0.002 * learning_rate)
        for i, g in zip(ids, gradient, strict=True):
            result["parameters"][int(i)] -= learning_rate * float(g) / new_scale
        result["decay_scale"] = new_scale
    return result, dict(
        coefficient_touches=len(ids) if learning_rate else 0,
        global_decay_writes=int(learning_rate > 0),
        logical_decay_coefficients=109,
        snapshot_coefficient_copies=109,
        clip_counts=scale(x[None, :], head["scaler"])[1],
        duration_s=time.monotonic() - began,
        device="cpu",
    )


def finite_difference(head: Json, x: Array) -> float:
    """Check every arm's derivative without relying on the optimizer's code."""
    theta = parameters(head)
    _, jac = logits_jacobian(head, x[None, :])
    errors = []
    for i in range(len(theta)):
        up, down = copy.deepcopy(head), copy.deepcopy(head)
        delta = np.zeros(len(theta))
        delta[i] = 1e-6
        up.update(parameters=(theta + delta).tolist(), decay_scale=1.0)
        down.update(parameters=(theta - delta).tolist(), decay_scale=1.0)
        numeric = (
            logits_jacobian(up, x[None, :])[0][0] - logits_jacobian(down, x[None, :])[0][0]
        ) / 2e-6
        errors.append(abs(float(numeric - jac[0, i])))
    return max(errors)


def audit(head: Json, x: Array) -> Json:
    """Check local gradients, exact regularization, sign and zero-step identity."""
    rows = []
    for y in (0, 1):
        dense = dense_gradient(head, x, y)
        ids, gradient = sparse_gradient(head, x, y)
        reconstructed = 0.002 * parameters(head)
        reconstructed[ids] += gradient
        changed, work = update(head, x, y, 0.01)
        rows.append(
            dict(
                y=y,
                numerator=float(np.max(np.abs(dense - reconstructed))),
                denominator=len(dense),
                eligibility=True,
                failure=False,
                censored=False,
                update_error=float(
                    np.max(np.abs(parameters(changed) - (parameters(head) - 0.01 * dense)))
                ),
                **work,
            )
        )
    zero, _ = update(head, x, 1, 0.0)
    fd = finite_difference(head, x)
    return dict(
        rows=rows,
        finite_difference_error=fd,
        zero_step_identity=zero == head,
        passed=zero == head
        and fd < 1e-7
        and all(
            r["numerator"] < 1e-10 and r["update_error"] < 1e-10 and r["coefficient_touches"] <= 37
            for r in rows
        ),
    )


def fit_one(arm: str, seed: int, scaler: Json, x: Array, y: Array, tx: Array, ty: Array) -> Json:
    """Each seed gets the fixed batch budget; tune labels select temperature only."""
    start = time.monotonic()
    theta = np.zeros(COUNTS[arm])
    if arm == "mlp":
        theta = np.random.default_rng(seed).normal(0, 0.1, COUNTS[arm])
        theta[-1] = 0.0
    head = dict(
        arm=arm,
        seed=seed,
        parameters=theta.tolist(),
        decay_scale=1.0,
        scaler=scaler,
        temperature=1.0,
        parameter_count=COUNTS[arm],
    )
    initial = objective(head, x, y)
    for step in range(200):
        z, jac = logits_jacobian(head, x)
        theta -= 0.01 * (jac.T @ (expit(z) - y) / len(x) + 0.002 * theta)
        head["parameters"] = theta.tolist()
        if step % 50 == 0:
            print(f"[exp7996] arm={arm} seed={seed} steps={step + 1}/200", flush=True)
    final = objective(head, x, y)
    losses = {
        str(t): scalar.bce(np.asarray(expit(logits_jacobian(head, tx)[0] / t)), ty)
        for t in CONFIG["temperatures"]
    }
    head.update(
        temperature=min(CONFIG["temperatures"], key=lambda t: losses[str(t)]),
        initial_loss=initial,
        final_loss=final,
        loss_increased=final > initial,
        tune_temperature_losses=losses,
        optimizer_steps=200,
        duration_s=time.monotonic() - start,
    )
    return head


def train(data: Json, checkpoint_dir: Any = None) -> Json:
    """Save all seeds and report mechanism readiness without a natural-data claim."""
    from carnot.reporting.current_work_receipt import atomic_json

    validate_data(data)
    f, t = usable(data["fit"]), usable(data["tune"])
    x, tx = inputs(f), inputs(t)
    y, ty = (np.asarray([r["y"] for r in rows], dtype=float) for rows in (f, t))
    scaler = fit_scaler(x)
    _, fc = scale(x, scaler)
    _, tc = scale(tx, scaler)
    heads: Json = {}
    for arm in COUNTS:
        heads[arm] = []
        for seed in SEEDS:
            print(f"[exp7996] phase=train_begin arm={arm} seed={seed}", flush=True)
            head = fit_one(arm, seed, scaler, x, y, tx, ty)
            heads[arm].append(head)
            if checkpoint_dir is not None:
                atomic_json(checkpoint_dir / f"{arm}-{seed}.json", head)
            print(f"[exp7996] phase=train_end arm={arm} seed={seed}", flush=True)
    sx = np.asarray([[0.5] + [float(i % 2)] * 8 for i in range(128)])
    sy = np.asarray([i % 2 for i in range(128)], dtype=float)
    hx = np.asarray([[0.5] + [float(i % 2)] * 8 for i in range(32)])
    hy = np.asarray([i % 2 for i in range(32)], dtype=float)
    control = fit_one("spline", 17, fit_scaler(sx), sx, sy, hx, hy)
    probabilities = predict(control, hx)
    holdout = scalar.bce(probabilities, hy)
    return dict(
        heads=heads,
        feature_scaler=dict(scaler, fit_clip_counts=fc, tune_clip_counts=tc),
        optimizer_work=dict(
            total_steps=3000,
            steps_per_arm_seed=200,
            fitted_heads=15,
            fit_examples_per_step=len(f),
            device="cpu",
        ),
        positive_control_results=dict(
            passed=bool(holdout < np.log(2) - 0.01),
            initial_holdout_loss=float(np.log(2)),
            final_holdout_loss=holdout,
            headroom=float(np.log(2)),
            scope="circular_separable_unseen_identities",
            optimizer_steps=200,
            independent_holdout=32,
            fit_ids=[f"synthetic-fit-{i}" for i in range(128)],
            holdout_ids=[f"synthetic-holdout-{i}" for i in range(32)],
            rows=[
                dict(
                    source_id=f"synthetic-holdout-{i}",
                    numerator=float(p),
                    denominator=1,
                    y=int(y),
                    eligibility=True,
                    failure=False,
                    censored=False,
                )
                for i, (p, y) in enumerate(zip(probabilities, hy, strict=True))
            ],
        ),
    )
