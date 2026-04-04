"""Property-based testing as energy terms for code verification.

**Researcher summary:**
    Instead of checking specific input/output pairs (which LLMs have memorized),
    generates thousands of random inputs and checks invariant properties:
    commutativity, idempotency, type preservation, no-exception, bounds, etc.
    Energy = fraction of random inputs that violate the property.

**Detailed explanation for engineers:**
    Hand-written test cases like ``add(1,2)==3`` are in every LLM training
    dataset. Property-based testing catches the bugs LLMs actually make:

    - Off-by-one on large inputs (``fibonacci(100)`` overflows?)
    - Edge cases on empty/None/negative inputs
    - Floating point precision issues
    - Unicode handling failures
    - Timeout on pathological inputs

    This module provides:

    1. **Property predicates**: reusable checks like ``commutative(f, a, b)``
    2. **Input generators**: random ints, strings, lists, nested structures
    3. **PropertyTestConstraint**: a ``BaseConstraint`` that runs N random tests
    4. **property_test()**: standalone function returning failure details

    The property test results plug directly into ``iterative_refine_code()``
    for the EBM-guided feedback loop: the LLM sees exactly which random
    inputs violated which properties.

Spec: REQ-CODE-006
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint
from carnot.verify.python_types import safe_exec_function


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------


def gen_int(rng: random.Random, lo: int = -1000, hi: int = 1000) -> int:
    """Generate a random integer in [lo, hi].

    Spec: REQ-CODE-006
    """
    return rng.randint(lo, hi)


def gen_pos_int(rng: random.Random, lo: int = 0, hi: int = 1000) -> int:
    """Generate a non-negative integer.

    Spec: REQ-CODE-006
    """
    return rng.randint(lo, hi)


def gen_string(rng: random.Random, max_len: int = 20) -> str:
    """Generate a random ASCII string.

    Spec: REQ-CODE-006
    """
    length = rng.randint(0, max_len)
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    return "".join(rng.choice(chars) for _ in range(length))


def gen_list_int(rng: random.Random, max_len: int = 20, lo: int = -100, hi: int = 100) -> list[int]:
    """Generate a random list of integers.

    Spec: REQ-CODE-006
    """
    length = rng.randint(0, max_len)
    return [rng.randint(lo, hi) for _ in range(length)]


def gen_pair_int(rng: random.Random, lo: int = -1000, hi: int = 1000) -> tuple[int, int]:
    """Generate a pair of random integers.

    Spec: REQ-CODE-006
    """
    return (rng.randint(lo, hi), rng.randint(lo, hi))


# ---------------------------------------------------------------------------
# Property predicates
# ---------------------------------------------------------------------------


@dataclass
class PropertyViolation:
    """A single property violation found during testing.

    Spec: REQ-CODE-006
    """

    property_name: str
    input_args: tuple
    expected: str = ""
    actual: str = ""
    error: str | None = None


@dataclass
class PropertyTestResult:
    """Result of property-based testing.

    Spec: REQ-CODE-006
    """

    n_tests: int = 0
    n_passed: int = 0
    n_failed: int = 0
    violations: list[PropertyViolation] = field(default_factory=list)
    energy: float = 0.0  # n_failed / n_tests
    wall_clock_seconds: float = 0.0


def property_test(
    code: str,
    func_name: str,
    properties: list[dict[str, Any]],
    n_samples: int = 100,
    timeout_per_call: float = 1.0,
    seed: int = 42,
) -> PropertyTestResult:
    """Run property-based testing on a Python function.

    **Researcher summary:**
        Generates random inputs, checks properties, returns violations.
        Energy = fraction of tests that fail. Plugs into the iterative
        refinement loop for EBM-guided feedback.

    **Detailed explanation for engineers:**
        Each property is a dict with:
        - ``name``: human-readable property name
        - ``gen_args``: callable(rng) -> tuple of args for the function
        - ``check``: callable(result, *args) -> bool (True = property holds)
        - ``description``: optional human description for feedback

        The function is executed on each generated input, and the check
        predicate verifies the property. Failures are collected with
        full context for LLM feedback.

    Args:
        code: Python source code containing the function.
        func_name: Name of the function to test.
        properties: List of property dicts (see above).
        n_samples: Number of random tests per property.
        timeout_per_call: Max seconds per function call.
        seed: Random seed for reproducibility.

    Returns:
        PropertyTestResult with violations and energy.

    Spec: REQ-CODE-006
    """
    rng = random.Random(seed)
    start = time.time()

    total_tests = 0
    total_failures = 0
    violations: list[PropertyViolation] = []

    for prop in properties:
        prop_name = prop["name"]
        gen_args = prop["gen_args"]
        check = prop["check"]

        for _ in range(n_samples):
            total_tests += 1
            args = gen_args(rng)
            if not isinstance(args, tuple):
                args = (args,)

            result, error = safe_exec_function(code, func_name, args, timeout=timeout_per_call)

            if error is not None:
                total_failures += 1
                if len(violations) < 20:  # Cap stored violations
                    violations.append(
                        PropertyViolation(
                            property_name=prop_name,
                            input_args=args,
                            error=str(error),
                        )
                    )
                continue

            try:
                holds = check(result, *args)
            except Exception as e:
                holds = False
                if len(violations) < 20:
                    violations.append(
                        PropertyViolation(
                            property_name=prop_name,
                            input_args=args,
                            error=f"Check raised: {e}",
                        )
                    )
                total_failures += 1
                continue

            if not holds:
                total_failures += 1
                if len(violations) < 20:
                    violations.append(
                        PropertyViolation(
                            property_name=prop_name,
                            input_args=args,
                            actual=str(result),
                        )
                    )

    elapsed = time.time() - start
    energy = total_failures / max(total_tests, 1)

    return PropertyTestResult(
        n_tests=total_tests,
        n_passed=total_tests - total_failures,
        n_failed=total_failures,
        violations=violations,
        energy=energy,
        wall_clock_seconds=elapsed,
    )


class PropertyTestConstraint(BaseConstraint):
    """Property-based testing as a BaseConstraint for ComposedEnergy.

    **Researcher summary:**
        Wraps property_test() as an energy term. Energy = fraction of
        random tests that fail. Plugs into the verify-and-repair pipeline.

    Spec: REQ-CODE-006
    """

    def __init__(
        self,
        name: str,
        code: str,
        func_name: str,
        properties: list[dict[str, Any]],
        n_samples: int = 100,
        seed: int = 42,
    ) -> None:
        self._name = name
        self._code = code
        self._func_name = func_name
        self._properties = properties
        self._n_samples = n_samples
        self._seed = seed

    @property
    def name(self) -> str:
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        return 0.01

    def energy(self, x: jax.Array) -> jax.Array:
        """Run property tests. Energy = fraction of failures.

        Note: x (the embedding) is ignored — we execute the stored code.
        This matches ReturnTypeConstraint's pattern.

        Spec: REQ-CODE-006
        """
        result = property_test(
            self._code,
            self._func_name,
            self._properties,
            n_samples=self._n_samples,
            seed=self._seed,
        )
        return jnp.float32(result.energy)

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Property tests are not differentiable. Return zeros.

        Spec: REQ-CODE-006
        """
        return jnp.zeros_like(x)


def format_violations_for_llm(result: PropertyTestResult) -> str:
    """Format property violations as LLM feedback.

    **Researcher summary:**
        Converts PropertyTestResult into a human-readable feedback string
        for the iterative refinement loop.

    Spec: REQ-CODE-006
    """
    if not result.violations:
        return ""

    parts = [
        f"Property-based testing found {result.n_failed}/{result.n_tests} failures "
        f"({result.energy:.1%} failure rate):",
    ]

    # Group by property
    by_prop: dict[str, list[PropertyViolation]] = {}
    for v in result.violations:
        by_prop.setdefault(v.property_name, []).append(v)

    for prop_name, vs in by_prop.items():
        parts.append(f"\n  Property: {prop_name}")
        for v in vs[:5]:  # Show up to 5 examples per property
            if v.error:
                parts.append(f"    FAIL: f{v.input_args} raised {v.error}")
            else:
                parts.append(f"    FAIL: f{v.input_args} returned {v.actual}")

    parts.append("\nFix the function to satisfy ALL properties on random inputs.")
    return "\n".join(parts)
