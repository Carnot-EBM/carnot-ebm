"""REQ-KAN-8986: fixed features make coefficient fitting convex and comparable."""

from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import expit

from carnot.autoresearch import encoder_head_8972 as prior

Json = dict[str, Any]
Array: TypeAlias = NDArray[np.float64]
KNOTS = [-1.0, -1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
OPTIONS = dict(maxiter=500, gtol=1e-6, ftol=1e-12)


def checked(x: Any) -> Array:
    """Reject malformed inputs before scaling can hide missing or nonfinite features."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 27 or not len(a) or not np.isfinite(a).all():
        raise ValueError("features")
    return cast(Array, a)


def fit_scaling(x: Any) -> Json:
    """Only the original fit graphs determine coordinate ranges."""
    a = checked(x)
    return dict(minimum=a.min(axis=0).tolist(), maximum=a.max(axis=0).tolist())


def scale(x: Any, scaling: Json) -> tuple[Array, Json]:
    """Constants stay zero while every later out-of-range observation is counted."""
    a = checked(x)
    low, high = np.asarray(scaling["minimum"]), np.asarray(scaling["maximum"])
    span = high - low
    scaled = np.where(span > 0, 2 * (a - low) / np.where(span > 0, span, 1) - 1, 0)
    return np.clip(scaled, -1, 1), dict(
        below=(a < low).sum(axis=0).tolist(),
        above=(a > high).sum(axis=0).tolist(),
        constant_coordinates=np.flatnonzero(span == 0).tolist(),
    )


def design(x: Any, spline: bool) -> Array:
    """Keep redundant basis columns so neither labels nor rank select dimensions."""
    a = checked(x)
    if np.any(np.abs(a) > 1):
        raise ValueError("scaled_domain")
    linear = np.column_stack((np.ones(len(a)), a))
    return cast(
        Array,
        (
            np.column_stack((linear, BSpline(KNOTS, np.eye(7), 3)(a).reshape(len(a), 189)))
            if spline
            else linear
        ),
    )


def objective(c: Array, phi: Array, y: Array) -> tuple[float, Array]:
    """A proper error-label loss and ridge give both heads the same convex objective."""
    z = phi @ c
    penalty = c.copy()
    penalty[0] = 0
    return float(np.mean(np.logaddexp(0, z) - y * z) + 0.005 * (penalty @ penalty)), np.asarray(
        phi.T @ (expit(z) - y) / len(y) + 0.01 * penalty
    )


def fit(phi: Array, y: Array) -> Json:
    """Actual stationarity decides convergence even if the optimizer reports success."""
    if (
        phi.ndim != 2
        or y.shape != (len(phi),)
        or not len(y)
        or not np.isfinite(phi).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("fit_operands")
    result = minimize(
        objective,
        np.zeros(phi.shape[1]),
        args=(phi, y),
        method="L-BFGS-B",
        jac=True,
        bounds=None,
        options=OPTIONS,
    )
    loss, gradient = objective(result.x, phi, y)
    residual = float(np.max(np.abs(gradient)))
    finite = (
        np.isfinite(result.x).all()
        and np.isfinite(gradient).all()
        and np.isfinite(loss)
        and np.isfinite(expit(phi @ result.x)).all()
    )
    if not finite or residual > 1e-6:
        raise ValueError(f"nonconvergence:gradient_inf={residual};message={result.message}")
    return dict(
        coefficients=result.x.tolist(),
        loss=loss,
        gradient=gradient.tolist(),
        gradient_inf=residual,
        converged=True,
        finite=True,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        iterations=int(result.nit),
        parameter_count=phi.shape[1],
    )


def features(state: Json, rows: list[Json]) -> Array:
    """Sorted summands remove floating-point order drift without changing encoder weights."""
    data = {k: np.asarray(v) for k, v in prior.g.pack(rows).items()}
    e = {k: np.asarray(v) for k, v in state["params"]["encoder"].items()}
    h = np.tanh(data["x"] @ e["project"] + e["bias"])
    for _ in range(3):
        neighbors = np.zeros_like(h)
        for i in range(len(rows)):
            for j in range(len(rows[i]["nodes"])):
                values = h[i, data["a"][i, j] > 0]
                neighbors[i, j] = np.sort(values, axis=0).sum(axis=0) / max(1, len(values))
        h = np.tanh(h @ e["self"] + neighbors @ e["neighbor"] + e["round_bias"])
        h *= data["mask"][:, :, None]
    pooled = np.sort(h, axis=1).sum(axis=1) / data["mask"].sum(axis=1)[:, None]
    return cast(
        Array,
        np.column_stack(
            (
                (h * data["s"][:, :, None]).sum(axis=1),
                (h * data["q"][:, :, None]).sum(axis=1),
                pooled,
                prior.b._standardize(data["c"], state["scaling"]),
                data["missing"],
            )
        ),
    )


def numeric_audit() -> Json:
    """Constructed arithmetic qualifies mechanics and supplies no scientific evidence."""
    x = np.tile(np.linspace(-1, 1, 11)[:, None], (1, 27))
    phi = design(x, True)
    basis = phi[:, 28:].reshape(11, 27, 7)
    c = np.random.default_rng(71).normal(0, 0.1, 217)
    y: Array = np.asarray([i % 2 for i in range(11)], dtype=float)
    _, grad = objective(c, phi, y)
    finite = np.asarray(
        [
            (objective(c + d, phi, y)[0] - objective(c - d, phi, y)[0]) / 2e-6
            for d in np.eye(217) * 1e-6
        ]
    )
    p = expit(phi @ c)
    hessian = phi.T @ ((p * (1 - p))[:, None] * phi) / len(y) + np.diag([0.0] + [0.01] * 216)
    delta, partition, eigenvalue = (
        float(np.max(np.abs(grad - finite))),
        float(np.max(np.abs(basis.sum(axis=2) - 1))),
        float(np.linalg.eigvalsh(hessian).min()),
    )
    return dict(
        gradient_error=delta,
        partition_error=partition,
        # Fixed report precision removes BLAS rounding drift; the gate uses the raw value.
        hessian_min_eigenvalue=round(eigenvalue, 14),
        passed=delta < 1e-8 and partition < 1e-14 and eigenvalue >= -1e-12,
        scope="constructed_mechanics_only",
        independent_count=0,
    )
