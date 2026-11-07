"""REQ-REPORT-8855: expose discrimination and certificate support without refitting.

Power conditions on independent selection of a fixed rule. Census feasibility
does not establish that 240 selection labels will find that rule.
"""

from functools import cache
import math
from typing import Any

import numpy as np
from scipy.stats import beta, binom

from carnot.reporting import selective_availability_8844 as old

Json = dict[str, Any]
__all__ = ["old"]
ROOT = old.ROOT
NAME, TASK = "experiment_8855_v758_selective_ceiling", "exp8855-selective-ceiling"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/selective_ceiling{s}_8855.py" for s in ["", "_runner"]] + [CLI]
TESTS = ["tests/python/test_selective_ceiling_8855.py"]
PROTOCOLS = [
    ("historical", 360, old.p.ALPHA, 0),
    ("simultaneous", 720, 0.01 / (2 * 7 * 6), 0),
    ("split", 480, 0.01 / (2 * 7), 240),
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose progress while fixed deadlines continue to run."""
    print(f"[exp8855] phase={phase} completed={completed} pending={pending}", flush=True)


def minimum_accepted(errors: int, alpha: float) -> int:
    """Invert the exact binomial test at integer boundaries, including nonzero errors."""
    if errors < 0 or not 0 < alpha < 1:
        raise ValueError("binomial_counts_or_alpha")
    lo, hi = errors, max(1, errors + 1)
    while binom.cdf(errors, hi, 0.05) > alpha:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if binom.cdf(errors, mid, 0.05) <= alpha:
            hi = mid
        else:
            lo = mid
    return hi


@cache
def error_limits(budget: int, alpha: float) -> tuple[int, ...]:
    """A certificate fires exactly when its accepted-error count passes this tail test."""
    limits = [-1]
    for n in range(1, budget + 1):
        allowed = np.flatnonzero(binom.cdf(np.arange(n + 1), n, 0.05) <= alpha)
        limits.append(int(allowed[-1]) if len(allowed) else -1)
    return tuple(limits)


def firing_probability(budget: int, q: float, risk: float | None, alpha: float) -> float:
    """Sum over random accepted counts; conditioning on the mean count is incorrect."""
    if not q or risk is None:
        return 0.0
    n = np.arange(budget + 1)
    limits = np.asarray(error_limits(budget, alpha))
    return min(1.0, max(0.0, float(np.dot(binom.pmf(n, budget, q), binom.cdf(limits, n, risk)))))


def diagnose(scores: list[Json]) -> Json:
    """Enumerate every registered rule; exposed truth never chooses a deployment rule."""
    size = len({r["unit_id"] for r in scores})
    units, lookup = old.arrays(scores, size)
    if not size:
        raise ValueError("population_policy_roster")
    census: list[Json] = []
    power: list[Json] = []
    requirements: list[Json] = []
    decomposition: list[Json] = []
    for name, budget, alpha, selection in PROTOCOLS:
        for errors in range(11):
            n = minimum_accepted(errors, alpha)
            upper = float(beta.isf(alpha, errors + 1, n - errors))
            prior = float(beta.isf(alpha, errors + 1, n - 1 - errors))
            requirements.append(
                dict(
                    protocol=name,
                    alpha=alpha,
                    errors=errors,
                    accepted_required=n,
                    risk_upper=upper,
                    previous_risk_upper=prior,
                    zero_error_log_floor=math.ceil(math.log(alpha) / math.log(0.95))
                    if not errors
                    else None,
                    numerical_check_passed=upper <= 0.05 + 1e-14 and prior > 0.05,
                )
            )
    for arm, seed in old.POLICIES:
        local = [lookup[arm, seed, u] for u in units]
        probabilities = np.asarray([r["probabilities"] for r in local], dtype=float)
        if (
            probabilities.shape != (size, 3)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1)
            or any(r["status"] != "completed" for r in local)
        ):
            raise ValueError("invalid_or_failed_score")
        choices = probabilities.argmax(axis=1)
        expected = np.asarray([r["expected_action"] for r in local])
        actions = np.asarray(old.p.legacy.ACTIONS)[choices]
        for cutoff in old.p.THRESHOLDS:
            accepted = (choices != 2) & (probabilities.max(axis=1) >= cutoff)
            n, e = int(accepted.sum()), int((accepted & (actions != expected)).sum())
            q, risk = n / size, e / n if n else None
            row = dict(
                unit_id=f"{arm}/{seed}/{cutoff}",
                cluster_id="frozen_population",
                arm=arm,
                seed=seed,
                threshold=cutoff,
                accepted=n,
                errors=e,
                denominator=size,
                risk=risk,
                coverage=q,
                metric=q,
                status="completed",
                missing_reason=None,
                population_feasible=risk is not None and risk <= 0.05 and q >= 0.30,
                oracle_policy=False,
            )
            census.append(row)
            for name, budget, alpha, selection in PROTOCOLS:
                probability = firing_probability(budget, q, risk, alpha)
                # A separate elementary sum checks SciPy's conditional binomial CDF.
                check = (
                    math.fsum(
                        float(binom.pmf(k, budget, q))
                        * math.fsum(
                            math.comb(k, j) * risk**j * (1 - risk) ** (k - j)
                            for j in range(limit + 1)
                        )
                        for k, limit in enumerate(error_limits(budget, alpha))
                        if limit >= 0
                    )
                    if risk is not None
                    else 0.0
                )
                power.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        threshold=cutoff,
                        protocol=name,
                        certification_labels=budget,
                        selection_labels=selection,
                        total_labels=budget + selection,
                        alpha=alpha,
                        population_acceptance_probability=q,
                        population_error_probability=risk,
                        certificate_availability=probability,
                        numerical_check_probability=check,
                        numerical_check_passed=abs(probability - check) <= 1e-12,
                        selection_success_probability=None,
                        scope="conditional fixed-rule projection; selection success unmeasured",
                    )
                )
        progress("population_and_exact_power", len(census), 42 - len(census))
        feasible = [
            r for r in census if (r["arm"], r["seed"]) == (arm, seed) and r["population_feasible"]
        ]
        best = {
            name: max(
                (
                    r["certificate_availability"]
                    for r in power
                    if r["protocol"] == name
                    and (r["arm"], r["seed"]) == (arm, seed)
                    and any(c["threshold"] == r["threshold"] for c in feasible)
                ),
                default=0.0,
            )
            for name, *_ in PROTOCOLS
        }
        limitation = (
            "discrimination"
            if not feasible
            else "calibration_support"
            if best["split"] < 0.8 or (best["historical"] < 0.8 and best["simultaneous"] >= 0.8)
            else "multiplicity"
            if best["simultaneous"] < 0.8
            else "uncertainty"
        )
        decomposition.append(
            dict(
                arm=arm,
                seed=seed,
                population_feasible_cutoffs=len(feasible),
                binding_limitation=limitation,
                best_conditional_availability=best,
                selection_uncertainty="240-label independent selection success is not established",
                historical_support_floor=minimum_accepted(0, old.p.ALPHA),
            )
        )
    checks = all(r["numerical_check_passed"] for r in power + requirements)
    if not checks:
        raise ValueError("numerical_control_failed")
    feasible_keys = {
        (r["arm"], r["seed"], r["threshold"]) for r in census if r["population_feasible"]
    }
    return dict(
        rows=census,
        population_cutoff_rows=census,
        exact_sample_requirements=requirements,
        availability_power_rows=power,
        failure_decomposition=decomposition,
        split_study_feasible_score=int(
            any(
                r["protocol"] == "split"
                and (r["arm"], r["seed"], r["threshold"]) in feasible_keys
                and r["certificate_availability"] >= 0.8
                for r in power
            )
        ),
    )
