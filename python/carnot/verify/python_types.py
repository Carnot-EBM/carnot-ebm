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

import ast
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


def _compute_nesting_depth(node: ast.AST, current: int = 0) -> list[int]:
    """Recursively collect nesting depths at each compound statement.

    **Detailed explanation for engineers:**
        Walks the AST tree and tracks nesting depth. Every compound statement
        (function def, for, while, if, try, with) increments the depth counter
        for its children. Returns a list of depths at each compound node,
        which can be used to compute max and mean nesting depth.

    Args:
        node: Current AST node being visited.
        current: Current nesting depth (0 at module level).

    Returns:
        List of integer depths at each compound statement encountered.

    Spec: REQ-CODE-002
    """
    depths: list[int] = []
    nesting_types = (
        ast.FunctionDef, ast.AsyncFunctionDef,
        ast.For, ast.AsyncFor,
        ast.While, ast.If,
        ast.Try, ast.With, ast.AsyncWith,
    )
    for child in ast.iter_child_nodes(node):
        if isinstance(child, nesting_types):
            depths.append(current + 1)
            depths.extend(_compute_nesting_depth(child, current + 1))
        else:
            depths.extend(_compute_nesting_depth(child, current))
    return depths


def ast_code_to_embedding(code: str, feature_dim: int = 64) -> jax.Array:
    """Convert Python source code to a structural feature embedding using AST analysis.

    **Researcher summary:**
        Parses the code into an Abstract Syntax Tree and extracts structural
        features: counts of function defs, calls, loops, conditionals, returns,
        assignments, imports, try/except blocks, nesting depth stats, variable
        count, line count, AST node count, and cyclomatic complexity. Returns a
        normalized float32 vector of shape (feature_dim,).

    **Detailed explanation for engineers:**
        Unlike ``code_to_embedding`` which uses a bag-of-tokens approach (losing
        all structural information), this function parses the code into a Python
        Abstract Syntax Tree (AST) and extracts features that capture the *structure*
        of the code:

        1. **Node type counts**: How many function definitions, function calls,
           for/while loops, if/elif conditionals, return statements, assignments,
           import statements, and try/except blocks appear in the code.

        2. **Nesting depth**: The maximum and mean nesting depth of compound
           statements (functions, loops, conditionals, try blocks). Deeply nested
           code has different structural properties than flat code.

        3. **Variable count**: Number of unique variable names used in the code,
           counted via ``ast.Name`` nodes. This captures code complexity — a
           function using 2 variables is structurally simpler than one using 10.

        4. **Line count**: Number of non-empty lines in the source code.

        5. **AST node count**: Total number of nodes in the AST tree. This is a
           rough measure of code size/complexity at the structural level.

        6. **Cyclomatic complexity**: Approximated as (number of branch points) + 1,
           where branch points are if/elif/for/while/except/and/or. This measures
           how many independent execution paths exist through the code.

        **Normalization:** Raw counts are divided by (1 + max_value_in_vector) to
        map all features to the [0, 1) range. This prevents large counts from
        dominating the embedding.

        **Padding/truncation:** The raw feature vector is either zero-padded (if
        shorter than feature_dim) or truncated (if longer) to exactly feature_dim.

        **Error handling:** If the code has syntax errors and cannot be parsed,
        returns a zero vector of shape (feature_dim,). This is intentionally
        different from valid code (which will have nonzero features), making
        syntax errors detectable from the embedding alone.

        **Why AST over bag-of-tokens?**
        The bag-of-tokens embedding in ``code_to_embedding`` cannot distinguish
        between ``a + b`` and ``b + a``, or between a function with a return
        statement and one without (if the token counts happen to collide). AST
        features capture the *structure* of the code: whether there IS a return
        statement, how deeply nested the logic is, how many branches exist.
        This makes it much better at distinguishing correct code from buggy code
        that has been mutated (e.g., a missing return statement, an extra loop,
        or a swapped conditional).

    Args:
        code: Python source code string.
        feature_dim: Size of the output embedding vector. Default 64.

    Returns:
        JAX array of shape (feature_dim,) with float32 dtype, values in [0, 1).
        Returns zeros for code with syntax errors.

    Spec: REQ-CODE-002
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return jnp.zeros(feature_dim, dtype=jnp.float32)

    # --- Count structural features ---
    n_func_defs = 0
    n_func_calls = 0
    n_for_loops = 0
    n_while_loops = 0
    n_if_stmts = 0
    n_returns = 0
    n_assignments = 0
    n_imports = 0
    n_try_blocks = 0
    n_branches = 0  # for cyclomatic complexity
    unique_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_func_defs += 1
        elif isinstance(node, ast.Call):
            n_func_calls += 1
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            n_for_loops += 1
            n_branches += 1
        elif isinstance(node, ast.While):
            n_while_loops += 1
            n_branches += 1
        elif isinstance(node, ast.If):
            n_if_stmts += 1
            n_branches += 1
        elif isinstance(node, ast.Return):
            n_returns += 1
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            n_assignments += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            n_imports += 1
        elif isinstance(node, ast.Try):
            n_try_blocks += 1
        elif isinstance(node, (ast.And, ast.Or)):
            n_branches += 1

        # Count exception handlers as branches
        if isinstance(node, ast.ExceptHandler):
            n_branches += 1

        # Collect unique variable names
        if isinstance(node, ast.Name):
            unique_names.add(node.id)

    # --- Nesting depth ---
    depths = _compute_nesting_depth(tree)
    max_depth = float(max(depths)) if depths else 0.0
    mean_depth = float(sum(depths) / len(depths)) if depths else 0.0

    # --- Line count (non-empty lines) ---
    line_count = sum(1 for line in code.splitlines() if line.strip())

    # --- Total AST node count ---
    ast_node_count = sum(1 for _ in ast.walk(tree))

    # --- Cyclomatic complexity approximation: branches + 1 ---
    cyclomatic = n_branches + 1

    # --- Assemble raw feature vector ---
    raw_features = [
        float(n_func_defs),
        float(n_func_calls),
        float(n_for_loops),
        float(n_while_loops),
        float(n_if_stmts),
        float(n_returns),
        float(n_assignments),
        float(n_imports),
        float(n_try_blocks),
        max_depth,
        mean_depth,
        float(len(unique_names)),
        float(line_count),
        float(ast_node_count),
        float(cyclomatic),
    ]

    # --- Pad or truncate to feature_dim ---
    if len(raw_features) < feature_dim:
        raw_features.extend([0.0] * (feature_dim - len(raw_features)))
    else:
        raw_features = raw_features[:feature_dim]

    # --- Normalize to [0, 1) range ---
    feature_array = jnp.array(raw_features, dtype=jnp.float32)
    max_val = jnp.max(feature_array)
    feature_array = feature_array / (1.0 + max_val)

    return feature_array


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
