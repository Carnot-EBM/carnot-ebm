"""Tests for Python code verification constraints and code-to-embedding.

Spec coverage: REQ-CODE-001, REQ-CODE-002,
               SCENARIO-CODE-001, SCENARIO-CODE-002, SCENARIO-CODE-005
"""

from __future__ import annotations

import jax.numpy as jnp
from carnot.verify.python_types import (
    NoExceptionConstraint,
    ReturnTypeConstraint,
    TestPassConstraint,
    build_code_energy,
    code_to_embedding,
    safe_exec_function,
)

CORRECT_ADD = "def add(a: int, b: int) -> int:\n    return a + b"
BUGGY_ADD = "def add(a: int, b: int) -> int:\n    return a - b"
SYNTAX_ERROR_CODE = "def bad(:\n    return 1"


class TestSafeExecFunction:
    """Tests for safe_exec_function. REQ-CODE-001, SCENARIO-CODE-005."""

    def test_success(self) -> None:
        """REQ-CODE-001: successful execution returns (result, None)."""
        result, error = safe_exec_function(CORRECT_ADD, "add", (1, 2))
        assert result == 3
        assert error is None

    def test_exception(self) -> None:
        """SCENARIO-CODE-005: exception returns (None, error)."""
        code = "def fail(x: int) -> int:\n    raise ValueError('boom')"
        result, error = safe_exec_function(code, "fail", (1,))
        assert result is None
        assert error is not None
        assert isinstance(error, ValueError)

    def test_syntax_error(self) -> None:
        """SCENARIO-CODE-005: syntax error returns (None, SyntaxError)."""
        result, error = safe_exec_function(SYNTAX_ERROR_CODE, "bad", ())
        assert result is None
        assert error is not None
        assert isinstance(error, SyntaxError)

    def test_missing_function(self) -> None:
        """REQ-CODE-001: missing function returns (None, NameError)."""
        result, error = safe_exec_function("x = 1", "missing", ())
        assert result is None
        assert error is not None
        assert isinstance(error, NameError)


class TestReturnTypeConstraint:
    """Tests for ReturnTypeConstraint. REQ-CODE-001."""

    def test_correct_type_zero_energy(self) -> None:
        """REQ-CODE-001: correct return type yields energy 0."""
        c = ReturnTypeConstraint(
            name="type_check",
            code=CORRECT_ADD,
            func_name="add",
            test_inputs=[(1, 2), (0, 0)],
            expected_type=int,
        )
        emb = jnp.zeros(256)
        energy = float(c.energy(emb))
        assert energy == 0.0
        assert c.is_satisfied(emb)

    def test_wrong_type_positive_energy(self) -> None:
        """REQ-CODE-001: wrong return type yields energy > 0."""
        # Function returns None (no return), which is not int
        code = "def add(a: int, b: int) -> int:\n    a + b"
        c = ReturnTypeConstraint(
            name="type_check",
            code=code,
            func_name="add",
            test_inputs=[(1, 2), (0, 0)],
            expected_type=int,
        )
        emb = jnp.zeros(256)
        energy = float(c.energy(emb))
        assert energy > 0.0
        assert not c.is_satisfied(emb)

    def test_empty_inputs(self) -> None:
        """REQ-CODE-001: empty test inputs yields energy 0."""
        c = ReturnTypeConstraint(
            name="type_check",
            code=CORRECT_ADD,
            func_name="add",
            test_inputs=[],
            expected_type=int,
        )
        emb = jnp.zeros(256)
        assert float(c.energy(emb)) == 0.0

    def test_grad_is_zeros(self) -> None:
        """REQ-CODE-001: gradient is zero (not differentiable through exec)."""
        c = ReturnTypeConstraint(
            name="type_check",
            code=CORRECT_ADD,
            func_name="add",
            test_inputs=[(1, 2)],
            expected_type=int,
        )
        emb = jnp.ones(256)
        grad = c.grad_energy(emb)
        assert jnp.allclose(grad, jnp.zeros(256))


class TestNoExceptionConstraint:
    """Tests for NoExceptionConstraint. REQ-CODE-001."""

    def test_no_exception_zero_energy(self) -> None:
        """REQ-CODE-001: no exceptions yields energy 0."""
        c = NoExceptionConstraint(
            name="no_exc",
            code=CORRECT_ADD,
            func_name="add",
            test_inputs=[(1, 2), (0, 0)],
        )
        emb = jnp.zeros(256)
        assert float(c.energy(emb)) == 0.0
        assert c.is_satisfied(emb)

    def test_exception_positive_energy(self) -> None:
        """REQ-CODE-001: exceptions yield energy > 0."""
        code = "def fail(x: int) -> int:\n    raise ValueError('boom')"
        c = NoExceptionConstraint(
            name="no_exc",
            code=code,
            func_name="fail",
            test_inputs=[(1,), (2,)],
        )
        emb = jnp.zeros(256)
        energy = float(c.energy(emb))
        assert energy == 1.0  # All inputs raise
        assert not c.is_satisfied(emb)

    def test_empty_inputs(self) -> None:
        """REQ-CODE-001: empty test inputs yields energy 0."""
        c = NoExceptionConstraint(
            name="no_exc",
            code=CORRECT_ADD,
            func_name="add",
            test_inputs=[],
        )
        emb = jnp.zeros(256)
        assert float(c.energy(emb)) == 0.0

    def test_grad_is_zeros(self) -> None:
        """REQ-CODE-001: gradient is zero (not differentiable through exec)."""
        c = NoExceptionConstraint(
            name="no_exc",
            code=CORRECT_ADD,
            func_name="add",
            test_inputs=[(1, 2)],
        )
        emb = jnp.ones(256)
        grad = c.grad_energy(emb)
        assert jnp.allclose(grad, jnp.zeros(256))


class TestTestPassConstraint:
    """Tests for TestPassConstraint. REQ-CODE-001, SCENARIO-CODE-001, SCENARIO-CODE-002."""

    def test_correct_output_zero_energy(self) -> None:
        """SCENARIO-CODE-001: correct outputs yield energy 0."""
        c = TestPassConstraint(
            name="test_pass",
            code=CORRECT_ADD,
            func_name="add",
            test_cases=[((1, 2), 3), ((0, 0), 0)],
        )
        emb = jnp.zeros(256)
        assert float(c.energy(emb)) == 0.0
        assert c.is_satisfied(emb)

    def test_wrong_output_positive_energy(self) -> None:
        """SCENARIO-CODE-002: wrong outputs yield energy > 0."""
        c = TestPassConstraint(
            name="test_pass",
            code=BUGGY_ADD,
            func_name="add",
            test_cases=[((1, 2), 3), ((0, 0), 0)],
        )
        emb = jnp.zeros(256)
        energy = float(c.energy(emb))
        assert energy > 0.0  # (1, 2) -> -1 != 3
        assert not c.is_satisfied(emb)

    def test_empty_cases(self) -> None:
        """REQ-CODE-001: empty test cases yields energy 0."""
        c = TestPassConstraint(
            name="test_pass",
            code=CORRECT_ADD,
            func_name="add",
            test_cases=[],
        )
        emb = jnp.zeros(256)
        assert float(c.energy(emb)) == 0.0

    def test_grad_is_zeros(self) -> None:
        """REQ-CODE-001: gradient is zero (not differentiable through exec)."""
        c = TestPassConstraint(
            name="test_pass",
            code=CORRECT_ADD,
            func_name="add",
            test_cases=[((1, 2), 3)],
        )
        emb = jnp.ones(256)
        grad = c.grad_energy(emb)
        assert jnp.allclose(grad, jnp.zeros(256))


class TestCodeToEmbedding:
    """Tests for code_to_embedding. REQ-CODE-002."""

    def test_correct_shape(self) -> None:
        """REQ-CODE-002: embedding has shape (vocab_size,)."""
        emb = code_to_embedding(CORRECT_ADD, vocab_size=256)
        assert emb.shape == (256,)
        assert emb.dtype == jnp.float32

    def test_custom_vocab_size(self) -> None:
        """REQ-CODE-002: custom vocab_size works."""
        emb = code_to_embedding(CORRECT_ADD, vocab_size=64)
        assert emb.shape == (64,)

    def test_deterministic(self) -> None:
        """REQ-CODE-002: same code produces same embedding."""
        emb1 = code_to_embedding(CORRECT_ADD)
        emb2 = code_to_embedding(CORRECT_ADD)
        assert jnp.allclose(emb1, emb2)

    def test_different_code_different_embedding(self) -> None:
        """REQ-CODE-002: different code produces different embedding."""
        emb1 = code_to_embedding(CORRECT_ADD)
        emb2 = code_to_embedding(BUGGY_ADD)
        assert not jnp.allclose(emb1, emb2)

    def test_nonzero_counts(self) -> None:
        """REQ-CODE-002: embedding has nonzero token counts."""
        emb = code_to_embedding(CORRECT_ADD)
        assert float(jnp.sum(emb)) > 0.0

    def test_syntax_error_partial(self) -> None:
        """REQ-CODE-002: malformed code returns partial embedding."""
        emb = code_to_embedding(SYNTAX_ERROR_CODE)
        # Should not crash, returns whatever tokens were parsed
        assert emb.shape == (256,)


class TestBuildCodeEnergy:
    """Tests for build_code_energy. REQ-CODE-001."""

    def test_constraint_count(self) -> None:
        """REQ-CODE-001: composed energy has 3 constraints."""
        composed = build_code_energy(CORRECT_ADD, "add", [((1, 2), 3), ((0, 0), 0)])
        assert composed.num_constraints == 3

    def test_correct_code_verified(self) -> None:
        """SCENARIO-CODE-001: correct code passes all constraints."""
        composed = build_code_energy(CORRECT_ADD, "add", [((1, 2), 3), ((0, 0), 0)])
        emb = code_to_embedding(CORRECT_ADD)
        result = composed.verify(emb)
        assert result.is_verified()

    def test_buggy_code_not_verified(self) -> None:
        """SCENARIO-CODE-002: buggy code fails constraints."""
        composed = build_code_energy(BUGGY_ADD, "add", [((1, 2), 3), ((0, 0), 0)])
        emb = code_to_embedding(BUGGY_ADD)
        energy = float(composed.energy(emb))
        assert energy > 0.0
