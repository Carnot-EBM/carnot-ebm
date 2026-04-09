#!/usr/bin/env python3
"""Experiment 53: Runtime constraint verification via code instrumentation.

**Researcher summary:**
    Extends Experiment 48's *static* AST-based constraint extraction with
    *dynamic* runtime instrumentation. Inserts isinstance() guards, bound
    checks, and return-type assertions directly into the AST, then executes
    the instrumented code against concrete inputs to catch bugs that static
    analysis misses (off-by-one on edge inputs, division by zero, overflow).

**Detailed explanation for engineers:**
    Experiment 48 parses Python code and extracts constraints statically —
    without ever running the code. This works well for type-annotation
    mismatches and uninitialized variables, but misses bugs that only
    manifest on specific inputs (e.g., division by zero when a list is empty,
    off-by-one that only fails at boundary values).

    This experiment takes the complementary approach:

    1. **instrument_code(code)** — Uses ``ast.NodeTransformer`` to rewrite
       the AST, inserting runtime checks at four points:
       a. **Type checks at function entry**: For every annotated parameter,
          insert ``assert isinstance(param, expected_type)``.
       b. **Bound checks inside for-range loops**: Insert assertions that
          the loop variable stays within the declared range.
       c. **Return type checks**: Before each ``return`` statement, insert
          ``assert isinstance(return_value, expected_type)``.
       d. **Variable initialization tracking**: Maintain a set of assigned
          names and assert before first use (approximated via wrapper).

    2. **execute_instrumented(code, test_inputs)** — Runs the instrumented
       code in a restricted namespace (builtins + math only, no file I/O)
       against each test input. Catches ``AssertionError`` from the
       instrumented checks and reports per-constraint pass/fail.

    3. **Comparison**: Runs both static (Exp 48's ``code_to_constraints``)
       and dynamic instrumentation on 12 scenarios, showing which approach
       catches which bug class. The key insight: static and dynamic are
       complementary — neither subsumes the other.

Usage:
    .venv/bin/python scripts/experiment_53_runtime_constraints.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import ast
import copy
import math
import os
import sys
import textwrap
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from experiment_48_code_constraints import code_to_constraints


# ---------------------------------------------------------------------------
# 1. instrument_code — rewrite AST with runtime checks
# ---------------------------------------------------------------------------

# Map from annotation string to actual Python type names that isinstance()
# understands. We keep this small and safe — only builtins.
_ANNOTATION_TO_TYPES: dict[str, str] = {
    "int": "int",
    "float": "(int, float)",
    "str": "str",
    "bool": "bool",
    "list": "list",
    "dict": "dict",
    "tuple": "tuple",
    "set": "set",
}


def _annotation_name(node: ast.expr) -> str | None:
    """Extract simple type name from an annotation AST node.

    Handles ``ast.Name`` (e.g., ``int``), ``ast.Constant`` (string
    annotations like ``"int"``). Returns None for complex annotations
    (generics, unions, etc.) that we cannot instrument simply.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _RuntimeCheckInserter(ast.NodeTransformer):
    """AST transformer that inserts runtime constraint checks.

    **How it works (step by step):**

    For each ``FunctionDef`` node encountered:

    1. **Parameter type checks**: For every argument with a type annotation
       that maps to a known builtin type, we prepend an ``assert
       isinstance(arg, type), "..."`` statement to the function body. This
       catches callers passing wrong types.

    2. **Return type checks**: If the function has a ``-> T`` return
       annotation, we transform every ``return expr`` into::

           __rt_val__ = expr
           assert isinstance(__rt_val__, T), "..."
           return __rt_val__

       This catches the function *producing* the wrong type at runtime,
       which static analysis only detects for literal constants.

    3. **Loop bound checks**: For ``for i in range(n)`` patterns, we
       insert ``assert 0 <= i < n`` at the top of the loop body.  This
       is technically redundant (Python's ``range`` guarantees it), but
       serves as a demonstration and would catch bugs if someone manually
       manipulated the loop variable inside the body.

    The transformer works on a *copy* of the AST to avoid mutating the
    original.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Insert runtime checks into a function definition."""
        # --- 1. Parameter type checks at function entry ---
        type_checks: list[ast.stmt] = []
        for arg in node.args.args:
            if arg.annotation is None:
                continue
            type_name = _annotation_name(arg.annotation)
            if type_name is None or type_name not in _ANNOTATION_TO_TYPES:
                continue
            # Build: assert isinstance(arg, type), "msg"
            isinstance_types = _ANNOTATION_TO_TYPES[type_name]
            check = _make_isinstance_assert(
                var_name=arg.arg,
                type_expr=isinstance_types,
                msg=f"RUNTIME_TYPE_CHECK: {node.name}() param '{arg.arg}' "
                    f"expected {type_name}, got {{type({arg.arg}).__name__}}",
            )
            type_checks.append(check)

        # --- 2. Return type checks ---
        return_type_name: str | None = None
        if node.returns:
            return_type_name = _annotation_name(node.returns)
            if return_type_name and return_type_name in _ANNOTATION_TO_TYPES:
                node = self._transform_returns(node, return_type_name)

        # --- 3. Loop bound checks ---
        node = self._insert_loop_bound_checks(node)

        # Prepend parameter type checks to the function body.
        if type_checks:
            node.body = type_checks + node.body

        # Fix line numbers so the AST compiles.
        ast.fix_missing_locations(node)

        # Continue transforming nested functions.
        self.generic_visit(node)
        return node

    def _transform_returns(
        self, node: ast.FunctionDef, return_type_name: str
    ) -> ast.FunctionDef:
        """Replace ``return expr`` with type-checked version.

        Transforms each ``return expr`` into::

            __rt_val__ = expr
            assert isinstance(__rt_val__, expected_type), "..."
            return __rt_val__

        This is done by walking the function body and replacing Return
        nodes with a small block. We use a temporary variable name
        ``__rt_val__`` to avoid evaluating the expression twice.
        """
        isinstance_types = _ANNOTATION_TO_TYPES[return_type_name]

        class _ReturnRewriter(ast.NodeTransformer):
            def visit_Return(self, ret: ast.Return) -> list[ast.stmt]:
                if ret.value is None:
                    return [ret]
                # __rt_val__ = <expr>
                assign = ast.Assign(
                    targets=[ast.Name(id="__rt_val__", ctx=ast.Store())],
                    value=ret.value,
                    lineno=ret.lineno,
                    col_offset=ret.col_offset,
                )
                # assert isinstance(__rt_val__, type), "msg"
                check = _make_isinstance_assert(
                    var_name="__rt_val__",
                    type_expr=isinstance_types,
                    msg=f"RUNTIME_RETURN_CHECK: {node.name}() should return "
                        f"{return_type_name}, got {{type(__rt_val__).__name__}}",
                )
                # return __rt_val__
                new_ret = ast.Return(
                    value=ast.Name(id="__rt_val__", ctx=ast.Load()),
                    lineno=ret.lineno,
                    col_offset=ret.col_offset,
                )
                return [assign, check, new_ret]

            def visit_FunctionDef(self, inner: ast.FunctionDef) -> ast.FunctionDef:
                # Don't rewrite returns inside nested functions.
                return inner

        # _ReturnRewriter replaces Return nodes with lists of statements.
        # We need to flatten those lists into the parent body.
        rewriter = _ReturnRewriter()
        new_body: list[ast.stmt] = []
        for stmt in node.body:
            result = rewriter.visit(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
        node.body = new_body
        return node

    def _insert_loop_bound_checks(
        self, node: ast.FunctionDef
    ) -> ast.FunctionDef:
        """Insert bound-check assertions at the top of for-range loops.

        For ``for i in range(n)``, inserts:
            ``assert 0 <= i < n, "RUNTIME_BOUND_CHECK: ..."``

        For ``for i in range(lo, hi)``, inserts:
            ``assert lo <= i < hi, "RUNTIME_BOUND_CHECK: ..."``
        """
        class _LoopBounder(ast.NodeTransformer):
            def visit_For(self, for_node: ast.For) -> ast.For:
                self.generic_visit(for_node)  # recurse first

                if not isinstance(for_node.target, ast.Name):
                    return for_node
                if not (isinstance(for_node.iter, ast.Call)
                        and isinstance(for_node.iter.func, ast.Name)
                        and for_node.iter.func.id == "range"):
                    return for_node

                loop_var = for_node.target.id
                args = for_node.iter.args

                if len(args) == 1:
                    # range(n): assert 0 <= i < n
                    bound_expr = args[0]
                    check_code = (
                        f"assert 0 <= {loop_var} < __bound__, "
                        f"'RUNTIME_BOUND_CHECK: {loop_var} out of range(n)'"
                    )
                    # We need to store the bound value before the assertion
                    # to avoid re-evaluating complex expressions.
                    bound_assign = ast.parse(
                        f"__bound__ = {ast.unparse(bound_expr)}"
                    ).body[0]
                    check = ast.parse(check_code).body[0]
                    for_node.body = [bound_assign, check] + for_node.body
                elif len(args) >= 2:
                    lo_expr = ast.unparse(args[0])
                    hi_expr = ast.unparse(args[1])
                    check_code = (
                        f"assert {lo_expr} <= {loop_var} < {hi_expr}, "
                        f"'RUNTIME_BOUND_CHECK: {loop_var} out of "
                        f"range({lo_expr}, {hi_expr})'"
                    )
                    check = ast.parse(check_code).body[0]
                    for_node.body = [check] + for_node.body

                return for_node

        bounder = _LoopBounder()
        node = bounder.visit(node)
        return node


def _make_isinstance_assert(
    var_name: str, type_expr: str, msg: str
) -> ast.Assert:
    """Build an ``assert isinstance(var, type), f"msg"`` AST node.

    The message is an f-string so it can include ``{type(var).__name__}``
    for informative error reporting at runtime.

    Args:
        var_name: The variable to check (e.g., ``"x"``).
        type_expr: The type(s) to check against (e.g., ``"int"`` or
            ``"(int, float)"``).
        msg: The assertion message template. May contain ``{...}`` for
            f-string interpolation.

    Returns:
        An ``ast.Assert`` node ready to insert into a function body.
    """
    # Build: assert isinstance(var_name, type_expr), f"msg"
    # We use ast.parse for reliability rather than manually constructing
    # the full node tree (which is verbose and error-prone for f-strings).
    code = f'assert isinstance({var_name}, {type_expr}), f"{msg}"'
    return ast.parse(code).body[0]


def instrument_code(code: str) -> str:
    """Parse Python source code and insert runtime constraint checks.

    **How it works:**
    1. Parse *code* into an AST.
    2. Apply ``_RuntimeCheckInserter`` to insert isinstance guards,
       return-type checks, and loop-bound assertions.
    3. Unparse the modified AST back to source code.

    Args:
        code: Valid Python source code string.

    Returns:
        The instrumented source code with runtime checks inserted.
        If parsing fails, returns the original code unchanged.
    """
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return code

    transformer = _RuntimeCheckInserter()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    return ast.unparse(new_tree)


# ---------------------------------------------------------------------------
# 2. execute_instrumented — run instrumented code in a restricted namespace
# ---------------------------------------------------------------------------

def execute_instrumented(
    code: str,
    test_inputs: list[dict[str, Any]],
    target_func: str | None = None,
) -> dict[str, Any]:
    """Execute instrumented code against test inputs and collect results.

    **How it works (step by step):**

    1. Create a restricted namespace containing only safe builtins (no
       ``open``, ``exec``, ``eval``, ``__import__``, ``compile``). Add
       ``math`` as the only importable module.
    2. Execute the instrumented code in this namespace to define functions.
    3. For each test input dict, call the target function (auto-detected
       as the first defined function if not specified).
    4. Catch ``AssertionError`` from instrumented checks and record which
       constraint was violated and what value caused the failure.
    5. Also catch ``ZeroDivisionError``, ``IndexError``, ``TypeError``, and
       ``NameError`` as these represent runtime constraint violations that
       the instrumentation framework should detect.

    Args:
        code: Instrumented Python source code (output of instrument_code).
        test_inputs: List of dicts, each mapping parameter names to values.
        target_func: Name of the function to call. If None, uses the first
            ``def`` found in the code.

    Returns:
        Dict with keys:
        - ``"results"``: list of per-input results
        - ``"n_pass"``: count of inputs that passed all checks
        - ``"n_fail"``: count of inputs that triggered a violation
        - ``"violations"``: list of violation descriptions
    """
    # --- Build restricted namespace ---
    safe_builtins = {
        k: v for k, v in __builtins__.__dict__.items()
        if k not in {
            "open", "exec", "eval", "__import__", "compile",
            "breakpoint", "exit", "quit",
        }
    } if isinstance(__builtins__, type(sys)) else {
        k: v for k, v in __builtins__.items()
        if k not in {
            "open", "exec", "eval", "__import__", "compile",
            "breakpoint", "exit", "quit",
        }
    }

    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "math": math}

    # --- Execute the code to define functions ---
    try:
        exec(code, namespace)  # noqa: S102 — intentional restricted exec
    except Exception as e:
        return {
            "results": [],
            "n_pass": 0,
            "n_fail": 0,
            "violations": [f"Code execution failed: {e}"],
            "exec_error": str(e),
        }

    # --- Find the target function ---
    if target_func is None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                target_func = node.name
                break
    if target_func is None or target_func not in namespace:
        return {
            "results": [],
            "n_pass": 0,
            "n_fail": 0,
            "violations": [f"Function '{target_func}' not found in code"],
        }

    func = namespace[target_func]

    # --- Run test inputs ---
    results: list[dict[str, Any]] = []
    violations: list[str] = []
    n_pass = 0
    n_fail = 0

    for i, inputs in enumerate(test_inputs):
        try:
            result_val = func(**inputs)
            results.append({
                "input_idx": i,
                "inputs": inputs,
                "result": result_val,
                "passed": True,
            })
            n_pass += 1
        except AssertionError as e:
            msg = str(e)
            violations.append(msg)
            results.append({
                "input_idx": i,
                "inputs": inputs,
                "error": msg,
                "error_type": "AssertionError",
                "passed": False,
            })
            n_fail += 1
        except (ZeroDivisionError, IndexError, TypeError, NameError,
                OverflowError, ValueError) as e:
            msg = f"{type(e).__name__}: {e}"
            violations.append(msg)
            results.append({
                "input_idx": i,
                "inputs": inputs,
                "error": msg,
                "error_type": type(e).__name__,
                "passed": False,
            })
            n_fail += 1
        except Exception as e:
            msg = f"Unexpected {type(e).__name__}: {e}"
            violations.append(msg)
            results.append({
                "input_idx": i,
                "inputs": inputs,
                "error": msg,
                "error_type": type(e).__name__,
                "passed": False,
            })
            n_fail += 1

    return {
        "results": results,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# 3. Test scenarios — static vs dynamic comparison
# ---------------------------------------------------------------------------

def get_comparison_scenarios() -> list[dict[str, Any]]:
    """Return 12 scenarios comparing static vs dynamic constraint detection.

    Each scenario includes:
    - ``name``: human-readable description
    - ``code``: Python source code with a potential bug
    - ``test_inputs``: concrete inputs for dynamic execution
    - ``target_func``: which function to call
    - ``bug_type``: classification of the bug
    - ``expected_static``: whether static analysis should catch it
    - ``expected_dynamic``: whether dynamic execution should catch it
    - ``explanation``: why one approach catches it and the other doesn't

    **Bug categories covered:**
    1. Static-only: type annotation mismatch on literals, uninitialized vars
    2. Dynamic-only: edge-case division by zero, off-by-one on empty input,
       large-input overflow
    3. Both catch: always-wrong return type
    4. Neither catches: semantic bugs (correct types, wrong logic)
    """
    return [
        # =====================================================================
        # Category 1: STATIC catches, DYNAMIC may not (depends on inputs)
        # =====================================================================
        {
            "name": "Literal return type mismatch (str vs int)",
            "code": textwrap.dedent("""\
                def get_id() -> int:
                    return "not_a_number"
            """),
            "test_inputs": [{}],
            "target_func": "get_id",
            "bug_type": "type_mismatch_literal",
            "expected_static": True,
            "expected_dynamic": True,
            "explanation": (
                "Static: sees literal str where int declared. "
                "Dynamic: isinstance check catches str return."
            ),
        },
        {
            "name": "Uninitialized variable reference",
            "code": textwrap.dedent("""\
                def compute(x: int) -> int:
                    return x + undefined_var
            """),
            "test_inputs": [{"x": 5}],
            "target_func": "compute",
            "bug_type": "uninitialized_variable",
            "expected_static": True,
            "expected_dynamic": True,
            "explanation": (
                "Static: detects 'undefined_var' never assigned. "
                "Dynamic: NameError at runtime."
            ),
        },

        # =====================================================================
        # Category 2: DYNAMIC catches, STATIC does not
        # =====================================================================
        {
            "name": "Division by zero on empty list",
            "code": textwrap.dedent("""\
                def average(nums: list) -> float:
                    total = 0
                    for i in range(len(nums)):
                        total = total + nums[i]
                    return total / len(nums)
            """),
            "test_inputs": [
                {"nums": [1, 2, 3]},     # passes
                {"nums": []},             # division by zero!
            ],
            "target_func": "average",
            "bug_type": "division_by_zero_edge",
            "expected_static": False,
            "expected_dynamic": True,
            "explanation": (
                "Static: code is syntactically and type-correct. "
                "Dynamic: empty list causes ZeroDivisionError."
            ),
        },
        {
            "name": "Off-by-one index error on boundary input",
            "code": textwrap.dedent("""\
                def last_element(arr: list) -> int:
                    idx = len(arr)
                    return arr[idx]
            """),
            "test_inputs": [
                {"arr": [10, 20, 30]},   # IndexError: index 3
            ],
            "target_func": "last_element",
            "bug_type": "off_by_one",
            "expected_static": False,
            "expected_dynamic": True,
            "explanation": (
                "Static: cannot evaluate len(arr) without running code. "
                "Dynamic: IndexError when idx == len(arr) instead of len-1."
            ),
        },
        {
            "name": "Integer overflow on large input",
            "code": textwrap.dedent("""\
                def factorial(n: int) -> int:
                    result = 1
                    for i in range(1, n + 1):
                        result = result * i
                    return result
            """),
            "test_inputs": [
                {"n": 5},      # 120, fine
                {"n": 1000},   # huge number, still int in Python
                {"n": -1},     # edge case: range(1, 0) is empty, returns 1
            ],
            "target_func": "factorial",
            "bug_type": "large_input_behavior",
            "expected_static": False,
            "expected_dynamic": False,  # Python handles big ints natively
            "explanation": (
                "Static: cannot determine runtime values. "
                "Dynamic: Python has arbitrary-precision ints, so no overflow. "
                "This demonstrates a case where the 'bug' doesn't manifest "
                "in Python (unlike C/Java)."
            ),
        },
        {
            "name": "Type error from wrong argument type at call site",
            "code": textwrap.dedent("""\
                def double(x: int) -> int:
                    return x * 2
            """),
            "test_inputs": [
                {"x": 5},          # passes
                {"x": "hello"},    # TypeError from isinstance check
            ],
            "target_func": "double",
            "bug_type": "wrong_arg_type",
            "expected_static": False,
            "expected_dynamic": True,
            "explanation": (
                "Static: the function itself is correct; the bug is at the "
                "call site. Dynamic: isinstance guard catches wrong type."
            ),
        },
        {
            "name": "Conditional return type depends on input",
            "code": textwrap.dedent("""\
                def parse_value(s: str) -> int:
                    if s.isdigit():
                        return int(s)
                    return s
            """),
            "test_inputs": [
                {"s": "42"},       # returns int, ok
                {"s": "hello"},    # returns str, violates -> int
            ],
            "target_func": "parse_value",
            "bug_type": "conditional_type_violation",
            "expected_static": False,
            "expected_dynamic": True,
            "explanation": (
                "Static: both return paths exist; can't determine which "
                "runs. The literal 's' return isn't a constant, so static "
                "can't type-check it. Dynamic: isinstance catches str return."
            ),
        },

        # =====================================================================
        # Category 3: BOTH static and dynamic catch
        # =====================================================================
        {
            "name": "Always returns wrong type (float vs str)",
            "code": textwrap.dedent("""\
                def get_name() -> str:
                    return 3.14
            """),
            "test_inputs": [{}],
            "target_func": "get_name",
            "bug_type": "always_wrong_return_type",
            "expected_static": True,
            "expected_dynamic": True,
            "explanation": (
                "Static: literal 3.14 is float, not str. "
                "Dynamic: isinstance(3.14, str) fails."
            ),
        },
        {
            "name": "Multiple uninitialized vars always fail",
            "code": textwrap.dedent("""\
                def broken(a: int) -> int:
                    result = a + b + c
                    return result
            """),
            "test_inputs": [{"a": 1}],
            "target_func": "broken",
            "bug_type": "multiple_uninit",
            "expected_static": True,
            "expected_dynamic": True,
            "explanation": (
                "Static: 'b' and 'c' never assigned. "
                "Dynamic: NameError on first use of 'b'."
            ),
        },

        # =====================================================================
        # Category 4: NEITHER catches (limitations)
        # =====================================================================
        {
            "name": "Semantic bug: wrong formula (types correct)",
            "code": textwrap.dedent("""\
                def circle_area(radius: float) -> float:
                    return 2 * 3.14159 * radius
            """),
            "test_inputs": [
                {"radius": 1.0},  # returns 6.28 instead of 3.14
                {"radius": 5.0},  # returns 31.4 instead of 78.5
            ],
            "target_func": "circle_area",
            "bug_type": "semantic_wrong_formula",
            "expected_static": False,
            "expected_dynamic": False,
            "explanation": (
                "Static: types are correct, vars initialized. "
                "Dynamic: returns float as declared, no runtime error. "
                "Bug is SEMANTIC: uses circumference formula (2*pi*r) "
                "instead of area (pi*r^2). Neither approach can detect "
                "wrong math without a specification of expected output."
            ),
        },
        {
            "name": "Off-by-one in accumulation (correct types)",
            "code": textwrap.dedent("""\
                def sum_first_n(n: int) -> int:
                    total = 0
                    for i in range(n):
                        total = total + i
                    return total
            """),
            "test_inputs": [
                {"n": 5},   # returns 10 (0+1+2+3+4), not 15 (1+2+3+4+5)
                {"n": 1},   # returns 0, arguably wrong if expecting 1
            ],
            "target_func": "sum_first_n",
            "bug_type": "semantic_off_by_one",
            "expected_static": False,
            "expected_dynamic": False,
            "explanation": (
                "Static: code is well-typed and vars initialized. "
                "Dynamic: no runtime errors, returns valid int. "
                "Bug: computes 0+1+...+(n-1) instead of 1+2+...+n. "
                "This is a specification ambiguity, not a crash bug."
            ),
        },
        {
            "name": "Logic error in comparison (correct types)",
            "code": textwrap.dedent("""\
                def is_even(n: int) -> bool:
                    return n % 2 == 1
            """),
            "test_inputs": [
                {"n": 4},   # returns False (correct by accident? no, wrong)
                {"n": 3},   # returns True (wrong — 3 is odd)
            ],
            "target_func": "is_even",
            "bug_type": "semantic_logic_error",
            "expected_static": False,
            "expected_dynamic": False,
            "explanation": (
                "Static: bool return matches annotation, vars fine. "
                "Dynamic: no crashes, returns bool as expected. "
                "Bug: checks n%2==1 instead of n%2==0. Pure logic error."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# 4. Run static analysis (from Experiment 48)
# ---------------------------------------------------------------------------

def run_static_analysis(code: str) -> dict[str, Any]:
    """Run Experiment 48's static constraint extraction on code.

    Returns a dict with ``"detected"`` (bool) indicating if any constraint
    violations were found, and ``"constraints"`` listing what was found.
    """
    constraints = code_to_constraints(code)
    n_violated = sum(1 for c in constraints if c.get("satisfied") is False)
    return {
        "detected": n_violated > 0,
        "n_constraints": len(constraints),
        "n_violated": n_violated,
        "constraints": constraints,
    }


# ---------------------------------------------------------------------------
# 5. Run dynamic analysis
# ---------------------------------------------------------------------------

def run_dynamic_analysis(
    code: str,
    test_inputs: list[dict[str, Any]],
    target_func: str,
) -> dict[str, Any]:
    """Instrument code and execute against test inputs.

    Returns a dict with ``"detected"`` (bool) indicating if any runtime
    violations were triggered, plus detailed per-input results.
    """
    instrumented = instrument_code(code)
    result = execute_instrumented(instrumented, test_inputs, target_func)
    return {
        "detected": result["n_fail"] > 0,
        "instrumented_code": instrumented,
        **result,
    }


# ---------------------------------------------------------------------------
# 6. Main — run comparison and print results table
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all comparison scenarios and print static vs dynamic table."""
    print("=" * 78)
    print("EXPERIMENT 53: Runtime Constraint Verification via Code Instrumentation")
    print("  Compare STATIC (AST analysis) vs DYNAMIC (instrumented execution)")
    print("=" * 78)

    start = time.time()
    scenarios = get_comparison_scenarios()

    # Column headers for comparison table.
    print(f"\n{'#':<3} {'Scenario':<45} {'Static':<8} {'Dynamic':<8} "
          f"{'Match?':<7} {'Bug Type'}")
    print("-" * 120)

    n_correct_static = 0
    n_correct_dynamic = 0
    results: list[dict[str, Any]] = []

    for i, scenario in enumerate(scenarios):
        name = scenario["name"]
        code = scenario["code"]

        # Run both analyses.
        static = run_static_analysis(code)
        dynamic = run_dynamic_analysis(
            code, scenario["test_inputs"], scenario["target_func"]
        )

        # Compare against expectations.
        static_correct = static["detected"] == scenario["expected_static"]
        dynamic_correct = dynamic["detected"] == scenario["expected_dynamic"]

        if static_correct:
            n_correct_static += 1
        if dynamic_correct:
            n_correct_dynamic += 1

        static_icon = "✓" if static["detected"] else "·"
        dynamic_icon = "✓" if dynamic["detected"] else "·"
        match_s = "✓" if static_correct else "✗"
        match_d = "✓" if dynamic_correct else "✗"

        print(
            f"{i+1:<3} {name:<45} "
            f"{static_icon}({match_s})  "
            f"{dynamic_icon}({match_d})  "
            f"{'OK' if static_correct and dynamic_correct else 'MISS':<7} "
            f"{scenario['bug_type']}"
        )

        # Show violations for dynamic failures.
        if dynamic["detected"] and dynamic.get("violations"):
            for v in dynamic["violations"][:2]:  # show first 2
                print(f"    └─ {v[:90]}")

        results.append({
            "name": name,
            "static_detected": static["detected"],
            "dynamic_detected": dynamic["detected"],
            "expected_static": scenario["expected_static"],
            "expected_dynamic": scenario["expected_dynamic"],
            "static_correct": static_correct,
            "dynamic_correct": dynamic_correct,
            "bug_type": scenario["bug_type"],
        })

    # --- Summary statistics ---
    elapsed = time.time() - start
    n_total = len(results)
    sep = "=" * 78

    print(f"\n{sep}")
    print(f"EXPERIMENT 53 RESULTS ({elapsed:.1f}s)")
    print(sep)

    print(f"\n  Scenarios tested:           {n_total}")
    print(f"  Static analysis accuracy:   {n_correct_static}/{n_total}")
    print(f"  Dynamic analysis accuracy:  {n_correct_dynamic}/{n_total}")

    # Breakdown by category.
    static_only = sum(
        1 for r in results if r["expected_static"] and not r["expected_dynamic"]
    )
    dynamic_only = sum(
        1 for r in results if r["expected_dynamic"] and not r["expected_static"]
    )
    both_catch = sum(
        1 for r in results if r["expected_static"] and r["expected_dynamic"]
    )
    neither_catch = sum(
        1 for r in results if not r["expected_static"] and not r["expected_dynamic"]
    )

    print(f"\n  Bug detection breakdown:")
    print(f"    Static-only catches:      {static_only}")
    print(f"    Dynamic-only catches:     {dynamic_only}")
    print(f"    Both catch:               {both_catch}")
    print(f"    Neither catches:          {neither_catch}")

    # Overall verdict.
    all_correct = all(
        r["static_correct"] and r["dynamic_correct"] for r in results
    )
    mostly_correct = (n_correct_static + n_correct_dynamic) >= 2 * n_total * 0.8

    print(f"\n  Instrumentation demo:")
    print(f"    Sample instrumented code (scenario 1):")
    sample_instrumented = instrument_code(scenarios[0]["code"])
    for line in sample_instrumented.split("\n")[:10]:
        print(f"      {line}")

    if all_correct:
        print(f"\n  VERDICT: ✅ All predictions match — static + dynamic "
              f"are complementary!")
    elif mostly_correct:
        print(f"\n  VERDICT: ✅ Pipeline mostly works "
              f"({n_correct_static + n_correct_dynamic}/{2 * n_total} correct)")
    else:
        print(f"\n  VERDICT: ❌ Pipeline needs work "
              f"({n_correct_static + n_correct_dynamic}/{2 * n_total})")

    # Key insight.
    print(f"\n  KEY INSIGHT: Static analysis catches {static_only} bug classes "
          f"that don't need runtime")
    print(f"  (literal type mismatches, uninitialized vars). Dynamic analysis")
    print(f"  catches {dynamic_only} additional classes that require actual "
          f"execution (edge-case")
    print(f"  inputs, conditional paths). {neither_catch} scenarios have "
          f"semantic bugs that")
    print(f"  neither approach detects — these require specification-level "
          f"verification")
    print(f"  (the domain of Experiments 42b-47's Ising-based claim checking).")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
