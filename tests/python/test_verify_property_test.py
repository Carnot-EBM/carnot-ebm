"""Tests for property-based testing as energy terms.

Spec coverage: REQ-CODE-006
"""

from __future__ import annotations

import random

import jax.numpy as jnp

from carnot.verify.property_test import (
    PropertyTestConstraint,
    PropertyTestResult,
    PropertyViolation,
    format_violations_for_llm,
    gen_int,
    gen_list_int,
    gen_pair_int,
    gen_pos_int,
    gen_string,
    property_test,
)


# ---------------------------------------------------------------------------
# Tests: Input generators
# ---------------------------------------------------------------------------


class TestGenerators:
    """Tests for random input generators."""

    def test_gen_int(self) -> None:
        """REQ-CODE-006: gen_int produces ints in range."""
        rng = random.Random(42)
        for _ in range(50):
            v = gen_int(rng, -10, 10)
            assert -10 <= v <= 10

    def test_gen_pos_int(self) -> None:
        """REQ-CODE-006: gen_pos_int produces non-negative."""
        rng = random.Random(42)
        for _ in range(50):
            assert gen_pos_int(rng) >= 0

    def test_gen_string(self) -> None:
        """REQ-CODE-006: gen_string produces strings."""
        rng = random.Random(42)
        s = gen_string(rng, max_len=10)
        assert isinstance(s, str)
        assert len(s) <= 10

    def test_gen_list_int(self) -> None:
        """REQ-CODE-006: gen_list_int produces int lists."""
        rng = random.Random(42)
        lst = gen_list_int(rng, max_len=5)
        assert isinstance(lst, list)
        assert len(lst) <= 5

    def test_gen_pair_int(self) -> None:
        """REQ-CODE-006: gen_pair_int produces 2-tuples."""
        rng = random.Random(42)
        a, b = gen_pair_int(rng)
        assert isinstance(a, int)
        assert isinstance(b, int)


# ---------------------------------------------------------------------------
# Tests: property_test
# ---------------------------------------------------------------------------


GOOD_ADD = "def add(a, b):\n    return a + b"
BAD_ADD = "def add(a, b):\n    return a - b"
CRASHING = "def f(x):\n    return 1 / x"


class TestPropertyTest:
    """Tests for the property_test function."""

    def test_commutative_passes(self) -> None:
        """REQ-CODE-006: correct add passes commutativity."""
        result = property_test(
            GOOD_ADD,
            "add",
            [
                {
                    "name": "commutative",
                    "gen_args": lambda rng: gen_pair_int(rng),
                    "check": lambda result, a, b: result == a + b,
                },
            ],
            n_samples=50,
        )
        assert result.n_failed == 0
        assert result.energy == 0.0

    def test_wrong_code_fails(self) -> None:
        """REQ-CODE-006: buggy add fails property tests."""
        result = property_test(
            BAD_ADD,
            "add",
            [
                {
                    "name": "correctness",
                    "gen_args": lambda rng: gen_pair_int(rng, 1, 100),
                    "check": lambda result, a, b: result == a + b,
                },
            ],
            n_samples=50,
        )
        assert result.n_failed > 0
        assert result.energy > 0.0
        assert len(result.violations) > 0

    def test_exception_counted(self) -> None:
        """REQ-CODE-006: exceptions are counted as failures."""
        result = property_test(
            CRASHING,
            "f",
            [
                {
                    "name": "no_crash",
                    "gen_args": lambda rng: (gen_int(rng, -10, 10),),
                    "check": lambda result, x: True,  # just check no exception
                },
            ],
            n_samples=50,
        )
        # Some inputs will be 0, causing ZeroDivisionError
        assert result.n_failed > 0

    def test_multiple_properties(self) -> None:
        """REQ-CODE-006: multiple properties tested independently."""
        result = property_test(
            GOOD_ADD,
            "add",
            [
                {
                    "name": "commutative",
                    "gen_args": lambda rng: gen_pair_int(rng),
                    "check": lambda result, a, b: result == a + b,
                },
                {
                    "name": "identity",
                    "gen_args": lambda rng: (gen_int(rng), 0),
                    "check": lambda result, a, b: result == a,
                },
            ],
            n_samples=20,
        )
        assert result.n_tests == 40  # 20 per property
        assert result.n_failed == 0

    def test_deterministic(self) -> None:
        """REQ-CODE-006: same seed → same result."""
        r1 = property_test(
            GOOD_ADD,
            "add",
            [
                {
                    "name": "t",
                    "gen_args": lambda rng: gen_pair_int(rng),
                    "check": lambda r, a, b: True,
                },
            ],
            n_samples=10,
            seed=42,
        )
        r2 = property_test(
            GOOD_ADD,
            "add",
            [
                {
                    "name": "t",
                    "gen_args": lambda rng: gen_pair_int(rng),
                    "check": lambda r, a, b: True,
                },
            ],
            n_samples=10,
            seed=42,
        )
        assert r1.n_tests == r2.n_tests
        assert r1.n_failed == r2.n_failed

    def test_check_exception_counted(self) -> None:
        """REQ-CODE-006: exception in check predicate counted as failure."""
        result = property_test(
            GOOD_ADD,
            "add",
            [
                {
                    "name": "bad_check",
                    "gen_args": lambda rng: gen_pair_int(rng),
                    "check": lambda result, a, b: 1 / 0,  # always raises
                },
            ],
            n_samples=5,
        )
        assert result.n_failed == 5

    def test_non_tuple_args_wrapped(self) -> None:
        """REQ-CODE-006: gen_args returning non-tuple gets wrapped."""
        result = property_test(
            "def f(x):\n    return x * 2",
            "f",
            [
                {
                    "name": "double",
                    "gen_args": lambda rng: gen_int(rng, 1, 10),  # returns int, not tuple
                    "check": lambda result, x: result == x * 2,
                },
            ],
            n_samples=10,
        )
        assert result.n_failed == 0

    def test_wall_clock_recorded(self) -> None:
        """REQ-CODE-006: timing is recorded."""
        result = property_test(
            GOOD_ADD,
            "add",
            [
                {"name": "t", "gen_args": lambda rng: (1, 2), "check": lambda r, a, b: True},
            ],
            n_samples=5,
        )
        assert result.wall_clock_seconds >= 0.0

    def test_result_defaults(self) -> None:
        """REQ-CODE-006: PropertyTestResult defaults."""
        r = PropertyTestResult()
        assert r.n_tests == 0
        assert r.violations == []

    def test_violation_defaults(self) -> None:
        """REQ-CODE-006: PropertyViolation defaults."""
        v = PropertyViolation(property_name="test", input_args=(1,))
        assert v.error is None


# ---------------------------------------------------------------------------
# Tests: PropertyTestConstraint
# ---------------------------------------------------------------------------


class TestPropertyTestConstraint:
    """Tests for the BaseConstraint wrapper."""

    def test_energy_zero_for_correct(self) -> None:
        """REQ-CODE-006: correct code → energy 0."""
        constraint = PropertyTestConstraint(
            "add_props",
            GOOD_ADD,
            "add",
            [
                {
                    "name": "correct",
                    "gen_args": lambda rng: gen_pair_int(rng),
                    "check": lambda r, a, b: r == a + b,
                }
            ],
            n_samples=20,
        )
        x = jnp.zeros(10)  # dummy embedding
        assert float(constraint.energy(x)) < 0.01

    def test_energy_positive_for_buggy(self) -> None:
        """REQ-CODE-006: buggy code → energy > 0."""
        constraint = PropertyTestConstraint(
            "add_props",
            BAD_ADD,
            "add",
            [
                {
                    "name": "correct",
                    "gen_args": lambda rng: gen_pair_int(rng, 1, 100),
                    "check": lambda r, a, b: r == a + b,
                }
            ],
            n_samples=20,
        )
        x = jnp.zeros(10)
        assert float(constraint.energy(x)) > 0.0

    def test_grad_is_zeros(self) -> None:
        """REQ-CODE-006: gradient is zeros (not differentiable)."""
        constraint = PropertyTestConstraint(
            "t",
            GOOD_ADD,
            "add",
            [{"name": "t", "gen_args": lambda rng: (1, 2), "check": lambda r, a, b: True}],
        )
        grad = constraint.grad_energy(jnp.ones(5))
        assert jnp.allclose(grad, jnp.zeros(5))

    def test_name_and_threshold(self) -> None:
        """REQ-CODE-006: name and threshold properties."""
        constraint = PropertyTestConstraint(
            "my_test",
            GOOD_ADD,
            "add",
            [],
            n_samples=10,
        )
        assert constraint.name == "my_test"
        assert constraint.satisfaction_threshold == 0.01


# ---------------------------------------------------------------------------
# Tests: format_violations_for_llm
# ---------------------------------------------------------------------------


class TestFormatViolations:
    """Tests for LLM feedback formatting."""

    def test_empty_violations(self) -> None:
        """REQ-CODE-006: no violations → empty string."""
        result = PropertyTestResult(n_tests=10, n_passed=10, n_failed=0)
        assert format_violations_for_llm(result) == ""

    def test_violations_formatted(self) -> None:
        """REQ-CODE-006: violations include details."""
        result = PropertyTestResult(
            n_tests=10,
            n_passed=8,
            n_failed=2,
            energy=0.2,
            violations=[
                PropertyViolation("commutative", (3, 5), actual="wrong"),
                PropertyViolation("no_crash", (0,), error="ZeroDivisionError"),
            ],
        )
        feedback = format_violations_for_llm(result)
        assert "2/10 failures" in feedback
        assert "commutative" in feedback
        assert "ZeroDivisionError" in feedback
        assert "Fix the function" in feedback
