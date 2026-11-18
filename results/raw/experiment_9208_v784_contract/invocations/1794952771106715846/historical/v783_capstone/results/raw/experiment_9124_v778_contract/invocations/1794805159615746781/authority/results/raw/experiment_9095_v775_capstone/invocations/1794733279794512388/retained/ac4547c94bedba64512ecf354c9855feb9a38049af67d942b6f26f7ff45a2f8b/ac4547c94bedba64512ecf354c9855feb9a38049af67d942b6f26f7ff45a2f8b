"""REQ-REPORT-9094-V775: equal binary energies need not transmit equal forces.

This CPU diagnostic implements the paper's phase energy rather than changing
Carnot's discrete Ising model. No training, sampling or device benefit follows.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

Json = dict[str, Any]
Array: TypeAlias = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9094_v775_susceptibility", "exp9094-susceptibility"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    f"python/carnot/reporting/susceptibility{suffix}_9094.py"
    for suffix in ("", "_evidence", "_runner")
] + [CLI]
TESTS = ["tests/python/test_susceptibility_9094.py"]
SEEDS = list(range(775094, 775126))
ALPHAS, STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0], [0.1, 1.0, 10.0]
OFFSETS, BETAS = [0.0, 0.001, 0.01, 0.1], [1e-4, 1e-3, 1e-2]
STEP, DERIVATIVE_ATOL, ENERGY_ATOL, STABILITY_MIN = 1e-4, 1e-6, 1e-10, 1e-6
GRID: Json = dict(
    seeds=SEEDS,
    hidden=4,
    outputs=2,
    K=1,
    coupling="Normal(0,1)/4",
    fields="1+incident absolute coupling sum",
    alpha=ALPHAS,
    equal_second_harmonic_strengths=STRENGTHS,
    phase_offsets=OFFSETS,
    beta=BETAS,
    direction="next six RNG normals after W, normalized",
    output_probe="next two RNG normals after direction, normalized; force=-q",
    central_step=STEP,
    derivative_atol=DERIVATIVE_ATOL,
    binary_energy_atol=ENERGY_ATOL,
    minimum_eigenvalue=STABILITY_MIN,
    cancellation_atol=1e-10,
    newton_max_steps=20,
    newton_residual_target=1e-14,
    equilibrium_atol=1e-12,
)


def progress(phase: str) -> None:
    """Flush each boundary so a supervisor can distinguish work from a stall."""
    print(f"[exp9094] phase={phase}", flush=True)


def instance(seed: int) -> tuple[Array, Array, Array, Array]:
    """Paired arms reuse couplings, phase direction and independently applied force."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(4, 2)) / 4
    fields = 1 + np.r_[abs(w).sum(1), abs(w).sum(0)]
    direction, output = rng.normal(size=6), rng.normal(size=2)
    return (
        w,
        fields,
        direction / np.linalg.norm(direction),
        np.r_[np.zeros(4), output / np.linalg.norm(output)],
    )


def configurations() -> list[tuple[int, ...]]:
    """Keep configuration identity; only the all-zero state is used for inverse claims."""
    return list(product((0, 1), repeat=6))


def energy(z: Array, w: Array, fields: Array, alpha: float, strength: float) -> float:
    """Equations 6–8 include the half factor that makes SHI force strength*sin(2z)."""
    u, v = z[:4, None], z[None, 4:]
    coupling = -np.sum(w * ((1 - alpha) * np.cos(u - v) + alpha * np.cos(u + v)))
    return float(coupling - fields @ np.cos(z) - strength / 2 * np.cos(2 * z).sum())


def derivatives(
    z: Array, w: Array, fields: Array, alpha: float, strength: float
) -> tuple[Array, Array]:
    """Differentiate the energy directly, keeping the output phase-difference sign."""
    u, v = z[:4, None], z[None, 4:]
    difference, total = w * (1 - alpha) * np.sin(u - v), w * alpha * np.sin(u + v)
    gradient = fields * np.sin(z) + strength * np.sin(2 * z)
    gradient[:4] += (difference + total).sum(1)
    gradient[4:] += (-difference + total).sum(0)
    diagonal = w * ((1 - alpha) * np.cos(u - v) + alpha * np.cos(u + v))
    hessian = np.diag(
        fields * np.cos(z) + 2 * strength * np.cos(2 * z) + np.r_[diagonal.sum(1), diagonal.sum(0)]
    )
    mixed = -w * (np.sin(u) * np.sin(v) - (2 * alpha - 1) * np.cos(u) * np.cos(v))
    hessian[:4, 4:], hessian[4:, :4] = mixed, mixed.T
    return gradient, hessian


def finite_differences(
    z: Array, w: Array, fields: Array, alpha: float, strength: float
) -> tuple[Array, Array, Array]:
    """Use energy differences for mixed curvature so analytic Hessian errors cannot hide."""
    eye = np.eye(6) * STEP
    g = np.array(
        [
            (energy(z + e, w, fields, alpha, strength) - energy(z - e, w, fields, alpha, strength))
            / (2 * STEP)
            for e in eye
        ]
    )
    h = np.column_stack(
        [
            (
                derivatives(z + e, w, fields, alpha, strength)[0]
                - derivatives(z - e, w, fields, alpha, strength)[0]
            )
            / (2 * STEP)
            for e in eye
        ]
    )
    mixed = np.empty((4, 2))
    for i in range(4):
        for j in range(2):
            x, y = eye[i], eye[j + 4]
            mixed[i, j] = (
                energy(z + x + y, w, fields, alpha, strength)
                - energy(z + x - y, w, fields, alpha, strength)
                - energy(z - x + y, w, fields, alpha, strength)
                + energy(z - x - y, w, fields, alpha, strength)
            ) / (4 * STEP**2)
    return g, h, mixed


def loss_gradient(output: Array, target: Array) -> Array:
    """The cosine squared-loss gradient vanishes at binary phases even for wrong targets."""
    return cast(
        Array, np.asarray(-2 * (np.cos(output) - target) * np.sin(output), dtype=np.float64)
    )


def response(hessian: Array, q: Array) -> Json:
    """Never regularize an unstable Hessian into an apparent physical susceptibility."""
    minimum = float(np.linalg.eigvalsh(hessian)[0])
    if minimum <= STABILITY_MIN:
        return dict(
            status="ill_conditioned",
            minimum_eigenvalue=minimum,
            displacement=None,
            censor_reason="minimum Hessian eigenvalue <=1e-6; no inverse claim",
        )
    dz = np.linalg.solve(hessian, -q)
    hh, hy, yy = hessian[:4, :4], hessian[:4, 4:], hessian[4:, 4:]
    schur = yy - hy.T @ np.linalg.solve(hh, hy)
    hidden = np.linalg.solve(hh, hy @ np.linalg.solve(schur, q[4:]))
    return dict(
        status="completed",
        minimum_eigenvalue=minimum,
        displacement=dz.tolist(),
        schur_hidden_response=hidden.tolist(),
        condition_number=float(np.linalg.cond(hessian)),
        solve_residual_norm=float(np.linalg.norm(hessian @ dz + q)),
        censor_reason=None,
    )


def equilibrium(
    w: Array,
    fields: Array,
    alpha: float,
    strength: float,
    q: Array,
    beta: float,
    max_steps: int = 20,
) -> Json:
    """Solve the independently forced stationarity equation, without a ridge or loss force."""
    z = np.zeros(6)
    for _ in range(max_steps):
        g, h = derivatives(z, w, fields, alpha, strength)
        if np.linalg.norm(g + beta * q) <= 1e-14:
            break
        if np.linalg.eigvalsh(h)[0] <= STABILITY_MIN:
            break
        z -= np.linalg.solve(h, g + beta * q)
    g, h = derivatives(z, w, fields, alpha, strength)
    residual, minimum = float(np.linalg.norm(g + beta * q)), float(np.linalg.eigvalsh(h)[0])
    success = residual <= 1e-12 and minimum > STABILITY_MIN
    return dict(
        phases=z.tolist(),
        residual_norm=residual,
        minimum_eigenvalue=minimum,
        status="completed" if success else "failed",
        censor_reason=None if success else "stationarity_or_stability",
    )


def row(
    seed: int,
    arm: Any,
    condition: Any,
    metric: float | None,
    status: str = "completed",
    **extra: Any,
) -> Json:
    """Every conditional observation stays attached to its one independent coupling case."""
    return dict(
        unit_id=f"{seed}/{arm}/{condition}",
        arm=arm,
        condition=condition,
        family="random_bipartite_4x2",
        group=f"coupling-{seed}",
        seed=seed,
        absolute_metric=metric,
        numerator=metric,
        denominator=1,
        status=status,
        censor_reason=None if status == "completed" else "numerical_gate",
        **extra,
    )


def measure_case(seed: int) -> Json:
    """Retain each configuration, derivative comparison and forced response before reduction."""
    progress(f"case_{seed}_before")
    w, fields, direction, q = instance(seed)
    output: Json = dict(
        seed=seed,
        weights=w.tolist(),
        fields=fields.tolist(),
        phase_direction=direction.tolist(),
        output_probe=q.tolist(),
        **{
            k: []
            for k in (
                "binary_energy_rows",
                "stability_rows",
                "derivative_rows",
                "susceptibility_rows",
                "finite_perturbation_rows",
            )
        },
    )
    for strength in STRENGTHS:
        for bits in configurations():
            z = np.array(bits) * np.pi
            energies = [energy(z, w, fields, a, strength) for a in ALPHAS]
            eigenvalues = [
                float(np.linalg.eigvalsh(derivatives(z, w, fields, a, strength)[1])[0])
                for a in ALPHAS
            ]
            error = max(abs(e - energies[0]) for e in energies)
            identity = dict(second_harmonic=strength, binary_configuration=list(bits))
            output["binary_energy_rows"].append(
                row(
                    seed,
                    "paired_alphas",
                    identity,
                    error,
                    "completed" if error <= ENERGY_ATOL else "failed",
                    alphas=ALPHAS,
                    energies=energies,
                )
            )
            output["stability_rows"].append(
                row(
                    seed,
                    "paired_alphas",
                    identity,
                    min(eigenvalues),
                    alphas=ALPHAS,
                    minimum_eigenvalues=eigenvalues,
                    mixed_curvature_norms=[
                        float(np.linalg.norm(derivatives(z, w, fields, a, strength)[1][:4, 4:]))
                        for a in ALPHAS
                    ],
                    cosine_loss_gradient_norm=float(
                        np.linalg.norm(loss_gradient(z[4:], -np.ones(2)))
                    ),
                    stable=[e > STABILITY_MIN for e in eigenvalues],
                    stationary_gradient_norms=[
                        float(np.linalg.norm(derivatives(z, w, fields, a, strength)[0]))
                        for a in ALPHAS
                    ],
                )
            )
        for alpha in ALPHAS:
            condition = dict(alpha=alpha, second_harmonic=strength)
            for offset in OFFSETS:
                z = offset * direction
                g, h = derivatives(z, w, fields, alpha, strength)
                fdg, fdh, fdm = finite_differences(z, w, fields, alpha, strength)
                errors = [
                    float(np.max(abs(g - fdg))),
                    float(np.max(abs(h - fdh))),
                    float(np.max(abs(h[:4, 4:] - fdm))),
                ]
                output["derivative_rows"].append(
                    row(
                        seed,
                        alpha,
                        dict(condition, offset=offset),
                        max(errors),
                        "completed" if max(errors) <= DERIVATIVE_ATOL else "failed",
                        phases=z.tolist(),
                        analytic_gradient=g.tolist(),
                        finite_difference_gradient=fdg.tolist(),
                        analytic_hessian=h.tolist(),
                        finite_difference_hessian=fdh.tolist(),
                        finite_difference_mixed=fdm.tolist(),
                        gradient_error=errors[0],
                        hessian_error=errors[1],
                        mixed_curvature_error=errors[2],
                        mixed_curvature_norm=float(np.linalg.norm(h[:4, 4:])),
                    )
                )
            _, h = derivatives(np.zeros(6), w, fields, alpha, strength)
            result = response(h, q)
            dz = None if result["displacement"] is None else np.array(result["displacement"])
            output["susceptibility_rows"].append(
                row(
                    seed,
                    alpha,
                    condition,
                    None if dz is None else float(np.linalg.norm(dz[:4])),
                    result["status"],
                    response=result,
                    hessian=h.tolist(),
                    output_force=(-q[4:]).tolist(),
                    output_response_norm=None if dz is None else float(np.linalg.norm(dz[4:])),
                    mixed_curvature_norm=float(np.linalg.norm(h[:4, 4:])),
                    cosine_loss_gradient=loss_gradient(np.zeros(2), -np.ones(2)).tolist(),
                    cosine_loss_gradient_norm=float(
                        np.linalg.norm(loss_gradient(np.zeros(2), -np.ones(2)))
                    ),
                )
            )
            previous: tuple[float, float] | None = None
            for beta in BETAS:
                eq = equilibrium(w, fields, alpha, strength, q, beta)
                perturbation_error = (
                    None
                    if dz is None
                    else float(np.linalg.norm(np.array(eq["phases"]) - beta * dz))
                )
                slope = None
                if (
                    previous
                    and perturbation_error is not None
                    and min(perturbation_error, previous[1]) > 0
                ):
                    slope = float(
                        np.log(perturbation_error / previous[1]) / np.log(beta / previous[0])
                    )
                output["finite_perturbation_rows"].append(
                    row(
                        seed,
                        alpha,
                        dict(condition, beta=beta),
                        perturbation_error,
                        eq["status"] if dz is not None else "censored",
                        equilibrium=eq,
                        linear_displacement=None if dz is None else (beta * dz).tolist(),
                        error_per_beta=None
                        if perturbation_error is None
                        else perturbation_error / beta,
                        adjacent_error_log_slope=slope,
                        floating_point_floor=perturbation_error is not None
                        and perturbation_error < 1e-14,
                    )
                )
                if perturbation_error is not None:
                    previous = beta, perturbation_error
    progress(f"case_{seed}_after")
    return output


def summarize(cases: list[Json]) -> Json:
    """Recompute paired descriptive findings; repeated conditions never add independent cases."""
    units, paired = [], []
    for case in cases:
        rows = [
            r
            for key in (
                "binary_energy_rows",
                "derivative_rows",
                "susceptibility_rows",
                "finite_perturbation_rows",
            )
            for r in case[key]
        ]
        success = all(r["status"] == "completed" for r in rows)
        for strength in STRENGTHS:
            arms = {
                r["arm"]: r
                for r in case["susceptibility_rows"]
                if r["condition"]["second_harmonic"] == strength
            }
            values = [arms[a]["absolute_metric"] for a in (0.0, 0.5, 1.0)]
            cancellation = arms[0.5]["mixed_curvature_norm"] <= 1e-10
            suppression = all(v is not None for v in values) and values[1] <= 1e-10 < min(
                values[0], values[2]
            )
            paired.append(
                row(
                    case["seed"],
                    "alpha_0_0.5_1",
                    dict(second_harmonic=strength),
                    values[1],
                    "completed" if cancellation and suppression else "failed",
                    hidden_response_norms=values,
                    mixed_curvature_cancelled=cancellation,
                    forced_response_suppressed=suppression,
                )
            )
        units.append(
            row(
                case["seed"],
                "all_registered_conditions",
                "case",
                float(success),
                "completed" if success else "failed",
            )
        )
    present = {c["seed"] for c in cases}
    units.extend(
        row(seed, "all_registered_conditions", "case", None, "censored")
        for seed in GRID["seeds"]
        if seed not in present
    )
    paired_ok = (
        bool(paired)
        and all(r["status"] == "completed" for r in paired)
        and all(r["mixed_curvature_norms"][2] <= 1e-10 for c in cases for r in c["stability_rows"])
    )
    completed = sum(r["status"] == "completed" for r in units)
    energy_ok = bool(cases) and all(
        r["absolute_metric"] <= ENERGY_ATOL for c in cases for r in c["binary_energy_rows"]
    )
    derivative_ok = bool(cases) and all(
        r["absolute_metric"] <= DERIVATIVE_ATOL for c in cases for r in c["derivative_rows"]
    )
    return dict(
        rows=units,
        paired_rows=paired,
        counts=dict(
            intended=32,
            attempted=len(cases),
            completed=completed,
            failed=len(cases) - completed,
            censored=32 - len(cases),
        ),
        effective_independent_groups=len(cases),
        equal_energy_agreement=energy_ok,
        derivatives_agree=derivative_ok,
        mixed_curvature_and_response_reproduced=paired_ok,
        mechanism_reproduced=bool(cases)
        and energy_ok
        and derivative_ok
        and paired_ok
        and completed == len(cases),
        statistical_scope="32 coupling cases; seed-paired descriptive rows only; no significance or generalization claim",
    )
