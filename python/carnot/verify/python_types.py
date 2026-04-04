"""Execution-based Python code constraints as energy terms.

**Researcher summary:**
    Encodes Python function correctness as differentiable energy terms.
    ReturnTypeConstraint checks type annotations, NoExceptionConstraint checks
    for runtime errors, TestPassConstraint checks input/output correctness.
    Energy = fraction of test inputs that fail the check (0.0 = perfect).
    Code-to-embedding via bag-of-tokens frequency vector for learned verification.

**Detailed explanation for engineers:**
    This module bridges Carnot's constraint-based verification with actual Python
    code execution. Instead of checking mathematical properties of a configuration
    vector, these constraints *execute* the Python code on test inputs and measure
    correctness.

    **How it works:**
    1. ``safe_exec_function``: Runs arbitrary Python code in an isolated namespace
       with try/except wrapping. Returns (result, None) on success or (None, error)
       on failure. No signals or subprocesses needed for small functions.

    2. ``ReturnTypeConstraint``: Runs the function on each test input, checks if
       ``isinstance(result, expected_type)``. Energy = fraction that fail the type
       check.

    3. ``NoExceptionConstraint``: Same execution, but counts how many inputs raise
       exceptions. Energy = fraction that raise.

    4. ``TestPassConstraint``: Given (input, expected_output) pairs, runs the
       function and checks equality. Energy = fraction with wrong output.

    5. ``code_to_embedding``: Converts Python source to a fixed-size frequency
       vector using stdlib ``tokenize``. This embedding is what the learned
       verifier (Gibbs model) operates on.

    6. ``build_code_energy``: Composes all three constraints into a single
       ComposedEnergy for holistic verification.

    **Why execution-based?**
    Static analysis can catch some bugs, but execution-based checking is the gold
    standard: if the code produces the right output on test inputs, it works. The
    energy function measures *how wrong* the code is (0.0 to 1.0 scale), which
    is more informative than a binary pass/fail.

    **Why not differentiable?**
    Code execution involves Python's interpreter, which is not a JAX computation.
    So ``grad_energy`` returns zeros. The learned verifier (Gibbs model) provides
    the differentiable signal instead.

Spec: REQ-CODE-001, REQ-CODE-002
"""

from __future__ import annotations

import io
import tokenize
from typing import Any

import jax
import jax.numpy as jnp

from carnot.verify.constraint import BaseConstraint, ComposedEnergy


def safe_exec_function(
    code: str,
    func_name: str,
    args: tuple[Any, ...],
    timeout: float = 1.0,  # noqa: ARG001 — reserved for future use
) -> tuple[Any, Exception | None]:
    """Execute a Python function defined in ``code`` with the given arguments.

    **Researcher summary:**
        Safely executes arbitrary Python code in an isolated namespace.
        Returns (result, None) on success or (None, exception) on failure.

    **Detailed explanation for engineers:**
        This function uses Python's ``exec()`` to define the function in a fresh
        namespace, then calls it with the provided arguments. All exceptions
        (including SyntaxError from bad code) are caught and returned rather
        than raised.

        The ``timeout`` parameter is reserved for future use (e.g., signal-based
        timeout or subprocess isolation). For the small template functions used
        in code verification training, execution is effectively instant.

        **Security note:** This is NOT a production sandbox. It runs code in the
        same process. For untrusted code, use the autoresearch sandbox module
        (``carnot.autoresearch.sandbox``).

    Args:
        code: Python source code defining the function.
        func_name: Name of the function to call after exec.
        args: Positional arguments to pass to the function.
        timeout: Reserved for future timeout support (currently unused).

    Returns:
        Tuple of (result, None) on success, or (None, exception) on failure.

    Spec: REQ-CODE-001
    """
    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)  # noqa: S102 — intentional exec for code verification
    except Exception as e:
        return None, e

    func = namespace.get(func_name)
    if func is None:
        return None, NameError(f"Function '{func_name}' not found in code")

    try:
        result = func(*args)
    except Exception as e:
        return None, e

    return result, None


class ReturnTypeConstraint(BaseConstraint):
    """Energy = fraction of test inputs where return type doesn't match annotation.

    **Researcher summary:**
        Execution-based type constraint. Runs the function on each test input,
        checks isinstance(result, expected_type). Energy in [0, 1].

    **Detailed explanation for engineers:**
        For each test input, we:
        1. Execute the function via safe_exec_function
        2. If execution fails (exception), count as type failure
        3. If result is not isinstance(expected_type), count as failure
        4. Energy = n_failures / n_tests

        This is a "soft" constraint: energy 0.5 means half the test inputs
        produce the wrong type. Energy 0.0 means perfect type correctness.

    Spec: REQ-CODE-001
    """

    def __init__(
        self,
        name: str,
        code: str,
        func_name: str,
        test_inputs: list[tuple[Any, ...]],
        expected_type: type,
    ) -> None:
        """Initialize with code, function name, test inputs, and expected type.

        Args:
            name: Human-readable constraint name.
            code: Python source code defining the function.
            func_name: Name of the function to call.
            test_inputs: List of argument tuples to test.
            expected_type: Expected return type (checked via isinstance).
        """
        self._name = name
        self._code = code
        self._func_name = func_name
        self._test_inputs = test_inputs
        self._expected_type = expected_type

    @property
    def name(self) -> str:
        """Human-readable constraint name."""
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        """Threshold for satisfaction: energy must be exactly 0."""
        return 1e-6

    def energy(self, x: jax.Array) -> jax.Array:  # noqa: ARG002 — x is the embedding
        """Compute type-check energy by executing the code.

        Args:
            x: Code embedding (unused — we execute the actual code instead).

        Returns:
            Scalar energy in [0, 1]: fraction of type check failures.

        Spec: REQ-CODE-001
        """
        if not self._test_inputs:
            return jnp.float32(0.0)  # type: ignore[no-any-return]

        n_failures = 0
        for args in self._test_inputs:
            result, error = safe_exec_function(self._code, self._func_name, args)
            if error is not None or not isinstance(result, self._expected_type):
                n_failures += 1

        return jnp.float32(n_failures / len(self._test_inputs))  # type: ignore[no-any-return]

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient is zero — execution is not differentiable through JAX.

        Spec: REQ-CODE-001
        """
        return jnp.zeros_like(x)


class NoExceptionConstraint(BaseConstraint):
    """Energy = fraction of test inputs that raise exceptions.

    **Researcher summary:**
        Execution-based exception constraint. Runs the function on each test
        input, counts exceptions. Energy in [0, 1].

    **Detailed explanation for engineers:**
        Same pattern as ReturnTypeConstraint, but only checks whether the
        function raises an exception — doesn't care about the return type or
        value. Energy 0.0 means no exceptions on any test input.

    Spec: REQ-CODE-001
    """

    def __init__(
        self,
        name: str,
        code: str,
        func_name: str,
        test_inputs: list[tuple[Any, ...]],
    ) -> None:
        """Initialize with code, function name, and test inputs.

        Args:
            name: Human-readable constraint name.
            code: Python source code defining the function.
            func_name: Name of the function to call.
            test_inputs: List of argument tuples to test.
        """
        self._name = name
        self._code = code
        self._func_name = func_name
        self._test_inputs = test_inputs

    @property
    def name(self) -> str:
        """Human-readable constraint name."""
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        """Threshold for satisfaction: energy must be exactly 0."""
        return 1e-6

    def energy(self, x: jax.Array) -> jax.Array:  # noqa: ARG002 — x is the embedding
        """Compute exception energy by executing the code.

        Args:
            x: Code embedding (unused — we execute the actual code instead).

        Returns:
            Scalar energy in [0, 1]: fraction of inputs that raise exceptions.

        Spec: REQ-CODE-001
        """
        if not self._test_inputs:
            return jnp.float32(0.0)  # type: ignore[no-any-return]

        n_exceptions = 0
        for args in self._test_inputs:
            _, error = safe_exec_function(self._code, self._func_name, args)
            if error is not None:
                n_exceptions += 1

        return jnp.float32(n_exceptions / len(self._test_inputs))  # type: ignore[no-any-return]

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient is zero — execution is not differentiable through JAX.

        Spec: REQ-CODE-001
        """
        return jnp.zeros_like(x)


class TestPassConstraint(BaseConstraint):
    """Energy = fraction of test cases that produce wrong output.

    **Researcher summary:**
        Execution-based correctness constraint. Runs the function on each
        (input, expected_output) pair, checks equality. Energy in [0, 1].

    **Detailed explanation for engineers:**
        For each test case (args, expected):
        1. Execute the function with args
        2. If exception: count as failure
        3. If result != expected: count as failure
        4. Energy = n_failures / n_tests

        Uses Python's ``==`` operator for comparison, which works for ints,
        floats, strings, tuples, etc.

    Spec: REQ-CODE-001
    """

    def __init__(
        self,
        name: str,
        code: str,
        func_name: str,
        test_cases: list[tuple[tuple[Any, ...], Any]],
    ) -> None:
        """Initialize with code, function name, and test cases.

        Args:
            name: Human-readable constraint name.
            code: Python source code defining the function.
            func_name: Name of the function to call.
            test_cases: List of (args_tuple, expected_output) pairs.
        """
        self._name = name
        self._code = code
        self._func_name = func_name
        self._test_cases = test_cases

    @property
    def name(self) -> str:
        """Human-readable constraint name."""
        return self._name

    @property
    def satisfaction_threshold(self) -> float:
        """Threshold for satisfaction: energy must be exactly 0."""
        return 1e-6

    def energy(self, x: jax.Array) -> jax.Array:  # noqa: ARG002 — x is the embedding
        """Compute test-pass energy by executing the code.

        Args:
            x: Code embedding (unused — we execute the actual code instead).

        Returns:
            Scalar energy in [0, 1]: fraction of test cases with wrong output.

        Spec: REQ-CODE-001
        """
        if not self._test_cases:
            return jnp.float32(0.0)  # type: ignore[no-any-return]

        n_failures = 0
        for args, expected in self._test_cases:
            result, error = safe_exec_function(self._code, self._func_name, args)
            if error is not None or result != expected:
                n_failures += 1

        return jnp.float32(n_failures / len(self._test_cases))  # type: ignore[no-any-return]

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient is zero — execution is not differentiable through JAX.

        Spec: REQ-CODE-001
        """
        return jnp.zeros_like(x)


def code_to_embedding(code: str, vocab_size: int = 256) -> jax.Array:
    """Convert Python source code to a bag-of-tokens frequency vector.

    **Researcher summary:**
        Tokenizes Python source via stdlib ``tokenize``, hashes each token
        string to an index in [0, vocab_size), counts frequencies. Returns
        a float32 vector of shape (vocab_size,).

    **Detailed explanation for engineers:**
        This is a simple but effective code embedding:

        1. **Tokenize**: Use Python's stdlib ``tokenize`` module to break the
           source into tokens (keywords, identifiers, operators, literals).
           This gives us syntactically meaningful units, not just characters.

        2. **Hash to index**: For each token string, compute
           ``hash(token_string) % vocab_size``. This maps an unbounded vocabulary
           to a fixed-size vector. Collisions are possible but rare for small
           codebases.

        3. **Count frequencies**: Increment the count at the hashed index.
           The result is a bag-of-tokens: it captures *which* tokens appear
           and how often, but not their order.

        **Why bag-of-tokens?**
        For the code verifier's purposes, the presence/absence and frequency
        of tokens (like ``return``, ``+``, ``-``, variable names) is a strong
        signal for correctness. A function missing ``return`` is probably buggy.
        A function with ``-`` instead of ``+`` in an addition function is buggy.

        **Limitations:**
        - Ignores token order (can't distinguish ``a + b`` from ``b + a``)
        - Hash collisions may map different tokens to the same index
        - Fixed vocab_size limits expressiveness

    Args:
        code: Python source code string.
        vocab_size: Size of the embedding vector. Default 256.

    Returns:
        JAX array of shape (vocab_size,) with float32 dtype.

    Spec: REQ-CODE-002
    """
    counts = [0] * vocab_size

    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in tokens:
            if tok.type in (
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.COMMENT,
            ):
                continue
            idx = hash(tok.string) % vocab_size
            counts[idx] += 1
    except tokenize.TokenError:
        # Malformed code — return the partial tokenization we got
        pass

    return jnp.array(counts, dtype=jnp.float32)


def build_code_energy(
    code: str,
    func_name: str,
    test_cases: list[tuple[tuple[Any, ...], Any]],
    expected_type: type = int,
    vocab_size: int = 256,
) -> ComposedEnergy:
    """Compose ReturnTypeConstraint + NoExceptionConstraint + TestPassConstraint.

    **Researcher summary:**
        Builds a ComposedEnergy from three execution-based constraints that
        together measure Python function correctness: type checking, exception
        freedom, and test-case correctness.

    **Detailed explanation for engineers:**
        This is the convenience function for creating a holistic code verifier.
        It extracts test_inputs from test_cases (just the argument tuples),
        creates all three constraint types, and composes them with equal weight.

        The resulting ComposedEnergy can be used with any Carnot tool:
        - ``energy(embedding)``: Total code quality score
        - ``verify(embedding)``: Per-constraint breakdown
        - ``decompose(embedding)``: Individual constraint energies

    Args:
        code: Python source code defining the function.
        func_name: Name of the function to call.
        test_cases: List of (args_tuple, expected_output) pairs.
        expected_type: Expected return type. Default int.
        vocab_size: Embedding vector size. Default 256.

    Returns:
        ComposedEnergy with three constraints.

    Spec: REQ-CODE-001
    """
    test_inputs = [tc[0] for tc in test_cases]

    composed = ComposedEnergy(input_dim=vocab_size)
    composed.add_constraint(
        ReturnTypeConstraint(
            name="return_type",
            code=code,
            func_name=func_name,
            test_inputs=test_inputs,
            expected_type=expected_type,
        ),
        weight=1.0,
    )
    composed.add_constraint(
        NoExceptionConstraint(
            name="no_exception",
            code=code,
            func_name=func_name,
            test_inputs=test_inputs,
        ),
        weight=1.0,
    )
    composed.add_constraint(
        TestPassConstraint(
            name="test_pass",
            code=code,
            func_name=func_name,
            test_cases=test_cases,
        ),
        weight=1.0,
    )

    return composed
