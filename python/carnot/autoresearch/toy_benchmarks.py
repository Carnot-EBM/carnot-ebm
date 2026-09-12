"""Real, independently-computable potential functions for the autoresearch
demo benchmarks (REQ-AUTO-021).

**Why this exists.** `scripts/demo_autoresearch.py`'s DoubleWell/Rosenbrock
benchmarks let a hypothesis's `run(benchmark_data)` return `final_energy`
directly -- a bare number the sandbox takes on faith. An adversarial review
(2026-09-12) reproduced that a hypothesis can claim ANY energy and be
believed; `orchestrator.py`'s own docstring says "the energy function is the
objective judge (can't be gamed by an LLM)", but for these two benchmarks no
such judge existed. This module IS that judge: two real potential functions,
called by trusted harness code AFTER the sandbox returns, on a `final_state`
the hypothesis reports having found -- never by the sandboxed code itself, so
a hypothesis cannot influence its own score by any means other than actually
finding a lower-energy state.

Spec: REQ-AUTO-021
"""

from __future__ import annotations

from typing import Any

MAX_DIM = 64  # a hypothesis-controlled state length must not blow up the recompute


def double_well_energy(state: list[float]) -> float:
    """Symmetric double-well potential, global minima at every coordinate = +/-1.

    E(x) = sum_i (x_i^2 - 1)^2

    Spec: REQ-AUTO-021
    """
    return sum((float(x) ** 2 - 1.0) ** 2 for x in state)


def rosenbrock_energy(state: list[float]) -> float:
    """The classic Rosenbrock "banana" function, global minimum 0 at all-ones.

    E(x) = sum_i [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

    Needs at least 2 dimensions; a shorter state recomputes to None (caller
    treats that as "no trustworthy energy", same as a missing state).

    Spec: REQ-AUTO-021
    """
    if len(state) < 2:
        raise ValueError("rosenbrock_energy needs at least 2 dimensions")
    xs = [float(x) for x in state]
    return sum(
        100.0 * (xs[i + 1] - xs[i] ** 2) ** 2 + (1.0 - xs[i]) ** 2 for i in range(len(xs) - 1)
    )


BENCHMARK_ENERGY_FUNCTIONS: dict[str, Any] = {
    "double_well": double_well_energy,
    "rosenbrock": rosenbrock_energy,
}


def recompute_final_energy(benchmark_name: str, final_state: Any) -> float | None:
    """The one function trusted harness code calls to turn a hypothesis's
    claimed `final_state` into a real, independently-verified energy.

    Returns None (never raises) on anything that isn't a clean list of
    finite floats for a benchmark this module knows how to score -- an
    unscoreable claim is treated as "not measured", never as a score of 0
    or an error that could itself carry information back to the hypothesis.

    Spec: REQ-AUTO-021
    """
    energy_fn = BENCHMARK_ENERGY_FUNCTIONS.get(benchmark_name)
    if energy_fn is None:
        return None
    if not isinstance(final_state, list) or not final_state:
        return None
    if len(final_state) > MAX_DIM:
        return None
    try:
        values = [float(x) for x in final_state]
    except (TypeError, ValueError):
        return None
    if any(v != v or v in (float("inf"), float("-inf")) for v in values):  # NaN/inf guard
        return None
    try:
        energy = energy_fn(values)
    except (ValueError, ArithmeticError):
        return None
    if energy != energy or energy in (float("inf"), float("-inf")):
        return None
    return energy
