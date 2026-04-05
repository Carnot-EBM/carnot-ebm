"""Tests for Python code verification constraints and code-to-embedding.

Spec coverage: REQ-CODE-001, REQ-CODE-002 (bag-of-tokens + AST embedding),
               SCENARIO-CODE-001, SCENARIO-CODE-002, SCENARIO-CODE-005
"""

from __future__ import annotations

import jax.numpy as jnp
from carnot.verify.python_types import (
    NoExceptionConstraint,
    ReturnTypeConstraint,
    TestPassConstraint,
    ast_code_to_embedding,
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


COMPLEX_CODE = """
def process(items):
    result = []
    for item in items:
        if item > 0:
            try:
                val = item * 2
                result.append(val)
            except Exception:
                pass
        elif item == 0:
            continue
        else:
            while item < 0:
                item += 1
    return result
"""

NO_RETURN_ADD = "def add(a: int, b: int) -> int:\n    a + b"

CODE_WITH_IMPORTS_AND_BOOLOPS = """
import os
from sys import argv

def check(x):
    if x > 0 and x < 10 or x == 99:
        return True
    return False
"""


class TestAstCodeToEmbedding:
    """Tests for ast_code_to_embedding. REQ-CODE-002."""

    def test_correct_shape(self) -> None:
        """REQ-CODE-002: embedding has shape (feature_dim,)."""
        emb = ast_code_to_embedding(CORRECT_ADD, feature_dim=64)
        assert emb.shape == (64,)
        assert emb.dtype == jnp.float32

    def test_custom_feature_dim(self) -> None:
        """REQ-CODE-002: custom feature_dim works."""
        emb = ast_code_to_embedding(CORRECT_ADD, feature_dim=32)
        assert emb.shape == (32,)

    def test_deterministic(self) -> None:
        """REQ-CODE-002: same code produces same embedding."""
        emb1 = ast_code_to_embedding(CORRECT_ADD)
        emb2 = ast_code_to_embedding(CORRECT_ADD)
        assert jnp.allclose(emb1, emb2)

    def test_different_code_different_embedding(self) -> None:
        """REQ-CODE-002: different code produces different embedding."""
        emb1 = ast_code_to_embedding(CORRECT_ADD)
        emb2 = ast_code_to_embedding(BUGGY_ADD)
        # These share structure but differ in operator — embedding may differ
        # due to AST node count differences
        emb3 = ast_code_to_embedding(COMPLEX_CODE)
        assert not jnp.allclose(emb1, emb3)

    def test_syntax_error_returns_zeros(self) -> None:
        """REQ-CODE-002: syntax error returns zero vector."""
        emb = ast_code_to_embedding(SYNTAX_ERROR_CODE)
        assert emb.shape == (64,)
        assert float(jnp.sum(emb)) == 0.0

    def test_values_in_zero_one(self) -> None:
        """REQ-CODE-002: all values are in [0, 1) range."""
        emb = ast_code_to_embedding(COMPLEX_CODE)
        assert float(jnp.min(emb)) >= 0.0
        assert float(jnp.max(emb)) < 1.0

    def test_nonzero_for_valid_code(self) -> None:
        """REQ-CODE-002: valid code produces nonzero embedding."""
        emb = ast_code_to_embedding(CORRECT_ADD)
        assert float(jnp.sum(emb)) > 0.0

    def test_distinguishes_structural_mutations(self) -> None:
        """REQ-CODE-002: AST embedding detects structural code mutations.

        The AST embedding captures structural features like return-count and
        nesting depth that are directly affected by common code mutations.
        A missing-return bug produces a measurably different AST embedding,
        and structurally very different code (simple vs complex) produces
        large embedding distances.
        """
        # Missing return is a structural mutation: Return node disappears
        correct_ast = ast_code_to_embedding(CORRECT_ADD)
        no_return_ast = ast_code_to_embedding(NO_RETURN_ADD)
        assert not jnp.allclose(correct_ast, no_return_ast)

        # Simple vs complex code should have large structural distance
        complex_ast = ast_code_to_embedding(COMPLEX_CODE)
        dist_simple_complex = float(jnp.linalg.norm(correct_ast - complex_ast))
        dist_simple_buggy = float(jnp.linalg.norm(correct_ast - no_return_ast))
        # Complex code is structurally more different than a minor mutation
        assert dist_simple_complex > dist_simple_buggy

    def test_complex_code_features(self) -> None:
        """REQ-CODE-002: complex code has higher feature values than simple code."""
        simple_emb = ast_code_to_embedding(CORRECT_ADD)
        complex_emb = ast_code_to_embedding(COMPLEX_CODE)
        # Complex code should have more nonzero features
        simple_nonzero = int(jnp.sum(simple_emb > 0))
        complex_nonzero = int(jnp.sum(complex_emb > 0))
        assert complex_nonzero > simple_nonzero

    def test_small_feature_dim_truncates(self) -> None:
        """REQ-CODE-002: feature_dim smaller than raw features truncates correctly."""
        emb = ast_code_to_embedding(CORRECT_ADD, feature_dim=5)
        assert emb.shape == (5,)
        assert float(jnp.sum(emb)) > 0.0

    def test_imports_and_boolean_ops(self) -> None:
        """REQ-CODE-002: code with imports and boolean operators is embedded correctly."""
        emb = ast_code_to_embedding(CODE_WITH_IMPORTS_AND_BOOLOPS)
        assert emb.shape == (64,)
        assert float(jnp.sum(emb)) > 0.0
        # Should differ from simple code due to imports, boolean ops, higher complexity
        simple_emb = ast_code_to_embedding(CORRECT_ADD)
        assert not jnp.allclose(emb, simple_emb)


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
