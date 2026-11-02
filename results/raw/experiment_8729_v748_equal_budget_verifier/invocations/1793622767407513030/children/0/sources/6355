"""REQ-REPORT-8528: small residual heads preserve the option to return raw probabilities.

Only fit labels update coefficients. Tune labels choose a mixture weight after
the fixed final update, so an identity fallback makes no promise about test loss.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify.v730_source_control_fit import scalar_matrix

Json = dict[str, Any]
Array: TypeAlias = NDArray[np.float64]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAINABLE = ("brier_scalar", "brier_full", "brier_source_permuted")


def matrix(arm: str, x: Array, scaler: Json) -> Array:
    """Use the original fit-only geometry and bias without padding scalar capacity."""
    return np.asarray(
        scalar_matrix(x[:, 0], scaler)
        if arm == TRAINABLE[0]
        else sparse.basis(sparse.scale(x, scaler)[0])
    )


def offset(x: Array) -> Array:
    """Only the inherited logit offset needs an endpoint safeguard."""
    q = np.clip(x[:, 0], 1e-4, 1 - 1e-4)
    return np.asarray(np.log(q / (1 - q)), dtype=float)


def objective_gradient(design: Array, base: Array, y: Array, theta: Array) -> tuple[float, Array]:
    """The same Brier objective regularizes every coefficient, including the bias."""
    p = expit(base + design @ theta)
    return float(np.mean((p - y) ** 2) + 0.001 * (theta @ theta)), np.asarray(
        design.T @ (2 * (p - y) * p * (1 - p)) / len(y) + 0.002 * theta
    )


def mix(raw: Array, trained: Array, alpha: float) -> Array:
    """A zero mixture returns the original array so exact endpoint bytes survive."""
    return raw.copy() if alpha == 0 else np.asarray((1 - alpha) * raw + alpha * trained)


def choose_alpha(losses: Json) -> float:
    """Ascending order makes exact tune ties prefer the smaller mixture weight."""
    return min(ALPHAS, key=lambda a: losses[str(a)])


def actions(p: Array) -> Any:
    """Typed decisions keep uncertain predictions visible as escalations."""
    return np.where(p <= 0.2, "accept", np.where(p >= 0.8, "reject", "escalate"))


def energy(p: float) -> Json:
    """Endpoint limits express impossible outcomes without writing nonfinite JSON."""
    return dict(
        E0=None if p == 1 else float(-np.log1p(-p)),
        E1=None if p == 0 else float(-np.log(p)),
        limiting_energy="E1=+infinity" if p == 0 else "E0=+infinity" if p == 1 else None,
    )


def metrics(p: Array, rows: list[Json]) -> Json:
    """Average sibling answers within each question before reporting descriptive losses."""
    y = np.asarray([r["y"] for r in rows])
    losses = (p - y) ** 2
    costs = np.where((p > 0.2) & (p < 0.8), 0.2, np.where(p <= 0.2, y, 1 - y))
    questions = sorted({r["question_hash"] for r in rows})
    return dict(
        brier=float(np.mean(losses)),
        typed_decision_loss=float(np.mean(costs)),
        per_question_brier={
            q: float(np.mean([losses[i] for i, r in enumerate(rows) if r["question_hash"] == q]))
            for q in questions
        },
    )


def fit(arm: str, seed: int, scaler: Json, f: list[Json], t: list[Json]) -> Json:
    """Exactly two hundred full batches diagnose optimization without checkpoint selection."""
    x, tx = (np.asarray([r["x"] for r in rows], dtype=float) for rows in (f, t))
    y = np.asarray([r["y"] for r in f], dtype=float)
    design, tune = matrix(arm, x, scaler), matrix(arm, tx, scaler)
    base, tbase = offset(x), offset(tx)
    theta = np.zeros(design.shape[1])
    prefixes = []
    for step in range(201):
        objective, gradient = objective_gradient(design, base, y, theta)
        if step in (0, 25, 50, 100, 200):
            prefixes.append(
                dict(
                    step=step,
                    fit=metrics(expit(base + design @ theta), f),
                    tune=metrics(expit(tbase + tune @ theta), t),
                    coefficient_norm=float(np.linalg.norm(theta)),
                    gradient_norm=float(np.linalg.norm(gradient)),
                    fit_objective=objective,
                )
            )
            print(
                f"[exp8528] arm={arm} seed={seed} completed={step} pending={200 - step}", flush=True
            )
        if step < 200:
            theta -= 0.01 * gradient
    probability = expit(tbase + tune @ theta)
    losses = {str(a): metrics(mix(tx[:, 0], probability, a), t)["brier"] for a in ALPHAS}
    return dict(
        arm=arm,
        seed=seed,
        scaler=scaler,
        parameters=theta.tolist(),
        parameter_count=len(theta),
        temperature=1.0,
        optimizer_steps=200,
        decay_scale=1.0,
        initial_loss=prefixes[0]["fit_objective"],
        final_loss=prefixes[-1]["fit_objective"],
        alpha=choose_alpha(losses),
        tune_alpha_losses=losses,
        prefixes=prefixes,
    )


def gradient_checks(head: Json) -> Json:
    """Central differences check Brier derivatives at endpoints and interior spline knots."""
    low, high = np.asarray(head["scaler"]["minimum"]), np.asarray(head["scaler"]["maximum"])
    x = np.asarray([low, low + (high - low) * 4 / 9, high])
    x[0, 0], x[-1, 0] = 0.0, 1.0
    design, base = matrix(head["arm"], x, head["scaler"]), offset(x)
    theta: Array = np.asarray(head["parameters"], dtype=float)
    checks = []
    for i in range(3):
        for label in (0.0, 1.0):
            d, b, y = design[i : i + 1], base[i : i + 1], np.array([label])
            _, analytical = objective_gradient(d, b, y, theta)
            numeric = []
            for j in range(len(theta)):
                delta = np.eye(len(theta))[j] * 1e-4
                numeric.append(
                    (
                        objective_gradient(d, b, y, theta + delta)[0]
                        - objective_gradient(d, b, y, theta - delta)[0]
                    )
                    / 2e-4
                )
            error = np.abs(np.asarray(numeric) - analytical)
            mask = np.abs(analytical) > 1e-6
            absolute = float(error.max())
            relative = float(np.max(error[mask] / np.abs(analytical[mask]), initial=0))
            checks.append(
                dict(
                    support=["lower_endpoint", "interior", "upper_endpoint"][i],
                    label=label,
                    absolute_error=absolute,
                    relative_error=relative,
                    passed=absolute <= 1e-7 and relative <= 1e-5,
                )
            )
    return dict(passed=all(r["passed"] for r in checks), rows=checks)
