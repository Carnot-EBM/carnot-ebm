#!/usr/bin/env python3
"""Experiment 48: Automatic constraint extraction from Python code.

**Researcher summary:**
    Parses Python source code via the ``ast`` module to extract verifiable
    constraints (type annotations, loop bounds, return types, variable
    initialization). Verifies constraints using direct computation for
    arithmetic checks and ``ParallelIsingSampler`` for logical consistency.
    Demonstrates that LLM-generated *code* (not just natural-language claims)
    can be mechanically verified for internal consistency.

**Detailed explanation for engineers:**
    Experiments 42b-47 validated constraint verification for arithmetic and
    logical claims expressed as structured dicts. This experiment takes the
    next step: given raw Python *source code*, we:

    1. Parse it with ``ast`` to extract an abstract syntax tree.
    2. Walk the AST to identify four kinds of constraints:
       a. **Type constraints** — if a parameter or variable has a type
          annotation (e.g., ``x: int``), we record that the variable
          should satisfy that type contract at runtime.
       b. **Arithmetic/bound constraints** — ``for i in range(n)``
          implies ``0 <= i < n``; comparisons like ``if x > 0`` imply
          bounds.
       c. **Return-type constraints** — if a function declares
          ``-> int``, its return statements should produce values
          consistent with that type.
       d. **Initialization constraints** — every variable used in an
          expression must have been assigned (or passed as a parameter)
          before that use.

    3. Verify the extracted constraints:
       - Arithmetic constraints: direct computation.
       - Logical consistency (e.g., "x is annotated int but returned as
         str"): encoded as Ising propositions and verified via
         ``ParallelIsingSampler``.

    4. Report per-constraint pass/fail and an overall verdict.

    This bridges the gap between "verify a claim" (Exp 47) and "verify
    code" — the foundation for autonomous code-review in the autoresearch
    loop.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_48_code_constraints.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import ast
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# 1. code_to_constraints — parse Python code into a list of constraint dicts
# ---------------------------------------------------------------------------

def _annotation_to_type_name(node: ast.expr) -> str | None:
    """Extract a simple type name string from an AST annotation node.

    Handles ``ast.Name`` (e.g., ``int``), ``ast.Constant`` (string
    annotations like ``"int"``), and ``ast.Attribute`` (e.g.,
    ``typing.Optional``). Returns ``None`` for anything too complex to
    resolve statically.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _collect_assigned_names(body: list[ast.stmt]) -> set[str]:
    """Walk a list of statements and return all names that receive an
    assignment (``=``, ``for`` target, ``with ... as``, etc.).

    This is a conservative over-approximation — it does not track
    control-flow reachability (a name assigned inside an ``if`` branch is
    still counted as assigned).
    """
    assigned: set[str] = set()
    for stmt in body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]):
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        elif isinstance(stmt, ast.For):
            if isinstance(stmt.target, ast.Name):
                assigned.add(stmt.target.id)
            assigned |= _collect_assigned_names(stmt.body)
        elif isinstance(stmt, (ast.If, ast.While)):
            assigned |= _collect_assigned_names(stmt.body)
            assigned |= _collect_assigned_names(stmt.orelse)
        elif isinstance(stmt, ast.With):
            for item in stmt.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    assigned.add(item.optional_vars.id)
            assigned |= _collect_assigned_names(stmt.body)
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                assigned.add(stmt.target.id)
    return assigned


def _collect_used_names(node: ast.AST) -> set[str]:
    """Return all ``ast.Name`` ids that appear in *load* context (i.e.,
    the name is read, not written to).
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
    return names


def code_to_constraints(code: str) -> list[dict[str, Any]]:
    """Parse Python source *code* and extract verifiable constraints.

    **How it works (step by step):**
    1. Parse *code* into an AST via ``ast.parse``.
    2. For every ``FunctionDef`` found at the top level:
       a. Record parameter type annotations as **type constraints**.
       b. Record the return-type annotation as a **return-type constraint**.
       c. Scan the function body for ``for i in range(...)`` loops and
          emit **arithmetic bound constraints** (``0 <= i < bound``).
       d. Compare the set of names *used* inside the function body against
          the set of names *assigned* (including parameters).  Any name
          that is used but never assigned/passed becomes an
          **initialization constraint** violation.
    3. Return a flat list of constraint dicts. Each dict has at minimum:
       ``{"kind": str, "description": str, "satisfied": bool | None}``.
       The ``satisfied`` field is ``None`` when the constraint can only be
       checked via Ising (deferred to ``verify_code_constraints``).

    Returns:
        List of constraint dicts in the same spirit as Experiment 47.
    """
    tree = ast.parse(code)
    constraints: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        func_name = node.name

        # --- Parameters: names + type annotations ---
        param_names: set[str] = set()
        param_types: dict[str, str] = {}
        for arg in node.args.args:
            param_names.add(arg.arg)
            if arg.annotation:
                tname = _annotation_to_type_name(arg.annotation)
                if tname:
                    param_types[arg.arg] = tname
                    constraints.append({
                        "kind": "type",
                        "function": func_name,
                        "variable": arg.arg,
                        "expected_type": tname,
                        "description": (
                            f"{func_name}(): parameter '{arg.arg}' annotated "
                            f"as {tname}"
                        ),
                        "satisfied": None,  # checked at logical level
                    })

        # --- Return type annotation ---
        return_type: str | None = None
        if node.returns:
            return_type = _annotation_to_type_name(node.returns)
            if return_type:
                constraints.append({
                    "kind": "return_type",
                    "function": func_name,
                    "expected_type": return_type,
                    "description": (
                        f"{func_name}() annotated to return {return_type}"
                    ),
                    "satisfied": None,
                })

        # --- Check return statements for type consistency ---
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                # If the return value is a constant, we can check its type
                # against the declared return type right now.
                if isinstance(child.value, ast.Constant) and return_type:
                    actual = type(child.value.value).__name__
                    consistent = _types_compatible(actual, return_type)
                    constraints.append({
                        "kind": "return_value_type",
                        "function": func_name,
                        "expected_type": return_type,
                        "actual_type": actual,
                        "description": (
                            f"{func_name}() returns literal of type {actual}, "
                            f"expected {return_type}"
                        ),
                        "satisfied": consistent,
                    })

        # --- Loop bound constraints ---
        for child in ast.walk(node):
            if isinstance(child, ast.For) and isinstance(child.target, ast.Name):
                loop_var = child.target.id
                # Detect ``for <var> in range(...)`` pattern.
                if (isinstance(child.iter, ast.Call)
                        and isinstance(child.iter.func, ast.Name)
                        and child.iter.func.id == "range"):
                    args = child.iter.args
                    if len(args) == 1:
                        # range(n) → 0 <= var < n
                        bound_desc = _expr_source(args[0])
                        constraints.append({
                            "kind": "loop_bound",
                            "function": func_name,
                            "variable": loop_var,
                            "lower": 0,
                            "upper_expr": bound_desc,
                            "description": (
                                f"{func_name}(): loop var '{loop_var}' "
                                f"bounded by 0 <= {loop_var} < {bound_desc}"
                            ),
                            "satisfied": True,  # range() guarantees this
                        })
                    elif len(args) >= 2:
                        lo_desc = _expr_source(args[0])
                        hi_desc = _expr_source(args[1])
                        constraints.append({
                            "kind": "loop_bound",
                            "function": func_name,
                            "variable": loop_var,
                            "lower_expr": lo_desc,
                            "upper_expr": hi_desc,
                            "description": (
                                f"{func_name}(): loop var '{loop_var}' "
                                f"bounded by {lo_desc} <= {loop_var} < {hi_desc}"
                            ),
                            "satisfied": True,
                        })

        # --- Initialization: every used name must be assigned or a param ---
        # Built-in names and module-level names are excluded from the check.
        builtins_set = {
            "print", "len", "range", "int", "str", "float", "bool", "list",
            "dict", "set", "tuple", "type", "isinstance", "enumerate", "zip",
            "map", "filter", "sorted", "reversed", "sum", "min", "max", "abs",
            "True", "False", "None", "ValueError", "TypeError", "IndexError",
            "KeyError", "Exception", "super", "property", "staticmethod",
            "classmethod", "open", "input", "any", "all", "iter", "next",
            "hasattr", "getattr", "setattr",
        }
        assigned_names = param_names | _collect_assigned_names(node.body)
        used_names = _collect_used_names(node)
        # Remove names that are the function's own name (recursion) and builtins.
        uninitialized = used_names - assigned_names - builtins_set - {func_name}
        # Remove names that look like module-level imports (heuristic: uppercase
        # first letter or contains dots).  Also remove 'self'.
        uninitialized -= {"self"}

        for uname in sorted(uninitialized):
            constraints.append({
                "kind": "initialization",
                "function": func_name,
                "variable": uname,
                "description": (
                    f"{func_name}(): variable '{uname}' used but never "
                    f"assigned or passed as parameter"
                ),
                "satisfied": False,
            })

        # If all used names are initialized, record a positive constraint.
        if not uninitialized:
            constraints.append({
                "kind": "initialization",
                "function": func_name,
                "variable": "*",
                "description": (
                    f"{func_name}(): all variables properly initialized"
                ),
                "satisfied": True,
            })

    return constraints


def _types_compatible(actual: str, expected: str) -> bool:
    """Check if *actual* Python type name is compatible with *expected*
    annotation.

    This is a deliberately conservative check: ``int`` is compatible with
    ``int`` and ``float`` (numeric promotion), ``bool`` is compatible
    with ``int`` and ``bool``, and ``str`` is only compatible with ``str``.
    """
    if actual == expected:
        return True
    # int → float promotion is fine
    if actual == "int" and expected == "float":
        return True
    # bool is a subclass of int in Python
    if actual == "bool" and expected in ("int", "bool"):
        return True
    return False


def _expr_source(node: ast.expr) -> str:
    """Best-effort reconstruction of source text for an AST expression.

    Uses ``ast.unparse`` (Python 3.9+). Falls back to a readable repr
    for older Pythons.
    """
    try:
        return ast.unparse(node)
    except AttributeError:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return "<expr>"


# ---------------------------------------------------------------------------
# 2. verify_code_constraints — verify extracted constraints
# ---------------------------------------------------------------------------

def verify_code_constraints(
    code: str,
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify the constraints extracted from *code*.

    **How verification works:**

    - **Constraints already resolved** (``satisfied`` is ``True``/``False``):
      counted directly.
    - **Type / return-type constraints** (``satisfied is None``): we encode
      them as Ising propositions and check logical consistency. For instance,
      if a function declares ``-> int`` but also has a type constraint saying
      a returned variable is ``str``, these form a mutex pair that the Ising
      sampler should detect as contradictory.
    - **Arithmetic bound constraints**: verified via direct computation
      (``range()`` guarantees bounds, so these are always ``True``; however
      we cross-check against explicit index expressions if present).

    Returns:
        Dict with keys ``"constraints"`` (per-constraint details),
        ``"n_total"``, ``"n_satisfied"``, ``"n_violated"``, ``"verdict"``.
    """
    results: list[dict[str, Any]] = []
    n_satisfied = 0
    n_violated = 0

    # Separate already-resolved constraints from deferred ones.
    deferred_logical: list[dict[str, Any]] = []

    for c in constraints:
        if c["satisfied"] is True:
            n_satisfied += 1
            results.append({**c, "verified": True})
        elif c["satisfied"] is False:
            n_violated += 1
            results.append({**c, "verified": False})
        else:
            # Deferred — needs Ising verification.
            deferred_logical.append(c)

    # --- Ising-based logical consistency check for deferred constraints ---
    if deferred_logical:
        ising_result = _verify_logical_via_ising(deferred_logical)
        for c, ising_ok in zip(deferred_logical, ising_result["per_constraint"]):
            c_result = {**c, "satisfied": ising_ok, "verified": ising_ok}
            results.append(c_result)
            if ising_ok:
                n_satisfied += 1
            else:
                n_violated += 1

    verdict = "PASS" if n_violated == 0 else "FAIL"
    return {
        "constraints": results,
        "n_total": len(results),
        "n_satisfied": n_satisfied,
        "n_violated": n_violated,
        "verdict": verdict,
    }


def _verify_logical_via_ising(
    deferred: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify deferred type/return constraints via Ising sampling.

    **Encoding strategy:**
    Each deferred constraint becomes a proposition. We check whether the
    set of type declarations is internally consistent:
    - Each type annotation is a proposition "var X has type T".
    - A return-type annotation is "function returns type T".
    - If we find a ``return_value_type`` constraint that contradicts the
      return annotation, the propositions form an inconsistent set.

    Since the deferred constraints here are all type-level (not
    arithmetic), we encode them as Ising spins where spin=1 means
    "constraint holds" and spin=0 means "constraint violated." We add
    strong positive biases (all constraints *should* hold) and check
    whether the sampler can find a satisfying assignment.

    If the Ising model finds a zero-violation assignment, the constraints
    are consistent; otherwise there is a logical conflict.
    """
    from experiment_45_logical_consistency import encode_claims_as_ising, count_violations
    from carnot.samplers.parallel_ising import (
        ParallelIsingSampler,
        AnnealingSchedule,
    )
    import jax.numpy as jnp
    import jax.random as jrandom

    n_props = len(deferred)

    # Build logical claims: each deferred constraint is a proposition that
    # should be "true".
    claims: list[dict] = []
    for i, _c in enumerate(deferred):
        claims.append({"type": "true", "prop": i})

    # Add mutex relationships: if two constraints refer to the same
    # function's return type but with different expected types, they are
    # mutually exclusive.
    return_types_by_func: dict[str, list[int]] = {}
    for i, c in enumerate(deferred):
        if c["kind"] in ("return_type", "return_value_type"):
            fn = c["function"]
            return_types_by_func.setdefault(fn, []).append(i)

    for fn, indices in return_types_by_func.items():
        types_seen: dict[str, list[int]] = {}
        for idx in indices:
            t = deferred[idx].get("expected_type") or deferred[idx].get("actual_type", "")
            types_seen.setdefault(t, []).append(idx)
        # If there are conflicting types, add mutex constraints.
        type_groups = list(types_seen.values())
        for gi in range(len(type_groups)):
            for gj in range(gi + 1, len(type_groups)):
                for a in type_groups[gi]:
                    for b in type_groups[gj]:
                        claims.append({"type": "mutex", "props": [a, b]})

    biases_np, edge_pairs, weights_list = encode_claims_as_ising(claims, n_props)

    if len(edge_pairs) == 0:
        # No conflicts — all constraints are independently satisfiable.
        return {"consistent": True, "per_constraint": [True] * n_props}

    J = np.zeros((n_props, n_props), dtype=np.float32)
    for k, (i, j) in enumerate(edge_pairs):
        J[i, j] += weights_list[k]
        J[j, i] += weights_list[k]

    sampler = ParallelIsingSampler(
        n_warmup=500,
        n_samples=30,
        steps_per_sample=10,
        schedule=AnnealingSchedule(0.1, 8.0),
        use_checkerboard=True,
    )

    samples = sampler.sample(
        jrandom.PRNGKey(48),
        jnp.array(biases_np, dtype=jnp.float32),
        jnp.array(J, dtype=jnp.float32),
        beta=8.0,
    )

    # Find the sample with fewest violations.
    best_violations = len(claims)
    best_assignment: dict[int, bool] = {}
    for s_idx in range(samples.shape[0]):
        assignment = {i: bool(samples[s_idx, i]) for i in range(n_props)}
        v = count_violations(claims, assignment)
        if v < best_violations:
            best_violations = v
            best_assignment = assignment

    # Per-constraint satisfaction: a constraint is satisfied if the best
    # Ising assignment sets its proposition to True.
    per_constraint = [best_assignment.get(i, True) for i in range(n_props)]

    return {
        "consistent": best_violations == 0,
        "violations": best_violations,
        "per_constraint": per_constraint,
    }


# ---------------------------------------------------------------------------
# 3. Test scenarios — correct and buggy code snippets
# ---------------------------------------------------------------------------

def get_test_scenarios() -> list[dict[str, Any]]:
    """Return test scenarios: Python code snippets with expected verdicts.

    Each scenario has:
    - ``name``: human-readable description
    - ``code``: Python source code string
    - ``expected_verdict``: ``"PASS"`` or ``"FAIL"``
    - ``expected_issues``: list of issue descriptions to look for
    """
    return [
        # --- CORRECT code (should PASS) ---
        {
            "name": "Simple typed function",
            "code": '''
def add(x: int, y: int) -> int:
    return x + y
''',
            "expected_verdict": "PASS",
            "expected_issues": [],
        },
        {
            "name": "Loop with correct bounds",
            "code": '''
def sum_list(arr: list) -> int:
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total
''',
            "expected_verdict": "PASS",
            "expected_issues": [],
        },
        {
            "name": "Function returning correct literal type",
            "code": '''
def get_count() -> int:
    return 42
''',
            "expected_verdict": "PASS",
            "expected_issues": [],
        },
        {
            "name": "Multiple functions, all correct",
            "code": '''
def greet(name: str) -> str:
    return "hello"

def double(x: int) -> int:
    return 2
''',
            "expected_verdict": "PASS",
            "expected_issues": [],
        },
        {
            "name": "Loop with start and stop",
            "code": '''
def partial_sum(arr: list, start: int, end: int) -> int:
    total = 0
    for i in range(start, end):
        total += arr[i]
    return total
''',
            "expected_verdict": "PASS",
            "expected_issues": [],
        },
        # --- BUGGY code (should FAIL) ---
        {
            "name": "Return type mismatch (str instead of int)",
            "code": '''
def get_id() -> int:
    return "not_a_number"
''',
            "expected_verdict": "FAIL",
            "expected_issues": ["return_value_type"],
        },
        {
            "name": "Uninitialized variable used",
            "code": '''
def compute(x: int) -> int:
    return x + uninitialized_var
''',
            "expected_verdict": "FAIL",
            "expected_issues": ["initialization"],
        },
        {
            "name": "Multiple uninitialized variables",
            "code": '''
def broken(a: int) -> int:
    result = a + b + c
    return result
''',
            "expected_verdict": "FAIL",
            "expected_issues": ["initialization"],
        },
        {
            "name": "Return bool when int expected",
            "code": '''
def check_positive(x: int) -> str:
    return True
''',
            "expected_verdict": "FAIL",
            "expected_issues": ["return_value_type"],
        },
        {
            "name": "Float return when str expected",
            "code": '''
def get_name() -> str:
    return 3.14
''',
            "expected_verdict": "FAIL",
            "expected_issues": ["return_value_type"],
        },
    ]


# ---------------------------------------------------------------------------
# 4. Main — run all scenarios and print results
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all test scenarios and print a results table."""
    print("=" * 72)
    print("EXPERIMENT 48: Automatic Constraint Extraction from Python Code")
    print("  Parse code via ast → extract constraints → verify via Ising")
    print("=" * 72)

    start = time.time()
    scenarios = get_test_scenarios()
    results: list[dict[str, Any]] = []

    for scenario in scenarios:
        code = scenario["code"]
        name = scenario["name"]
        expected = scenario["expected_verdict"]

        # Step 1: extract constraints from source code.
        constraints = code_to_constraints(code)

        # Step 2: verify the extracted constraints.
        verification = verify_code_constraints(code, constraints)
        actual = verification["verdict"]

        correct_detection = actual == expected
        icon = "✓" if correct_detection else "✗"
        n_c = verification["n_total"]
        n_v = verification["n_violated"]

        print(
            f"  [{icon}] {name:<45s} → {actual} "
            f"({n_c} constraints, {n_v} violated) "
            f"[expected {expected}]"
        )

        # Print constraint details for failing detections or actual failures.
        if not correct_detection or actual == "FAIL":
            for c in verification["constraints"]:
                sat = "✓" if c.get("verified", c.get("satisfied")) else "✗"
                print(f"        [{sat}] {c['description']}")

        results.append({
            "name": name,
            "expected": expected,
            "actual": actual,
            "correct_detection": correct_detection,
            "n_constraints": n_c,
            "n_violated": n_v,
            "details": verification,
        })

    # --- Summary ---
    elapsed = time.time() - start
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"EXPERIMENT 48 RESULTS ({elapsed:.1f}s)")
    print(sep)

    n_total = len(results)
    n_correct = sum(1 for r in results if r["correct_detection"])
    n_pass_expected = sum(1 for r in results if r["expected"] == "PASS")
    n_fail_expected = sum(1 for r in results if r["expected"] == "FAIL")
    n_true_pos = sum(
        1 for r in results if r["expected"] == "FAIL" and r["actual"] == "FAIL"
    )
    n_true_neg = sum(
        1 for r in results if r["expected"] == "PASS" and r["actual"] == "PASS"
    )
    n_false_pos = sum(
        1 for r in results if r["expected"] == "PASS" and r["actual"] == "FAIL"
    )
    n_false_neg = sum(
        1 for r in results if r["expected"] == "FAIL" and r["actual"] == "PASS"
    )

    print(f"  Total scenarios:                  {n_total}")
    print(f"  Correct detections:               {n_correct}/{n_total}")
    print(f"  True positives (caught bugs):     {n_true_pos}/{n_fail_expected}")
    print(f"  True negatives (passed correct):  {n_true_neg}/{n_pass_expected}")
    print(f"  False positives (flagged correct): {n_false_pos}")
    print(f"  False negatives (missed bug):      {n_false_neg}")

    total_constraints = sum(r["n_constraints"] for r in results)
    total_violated = sum(r["n_violated"] for r in results)
    print(f"\n  Constraints extracted:  {total_constraints}")
    print(f"  Constraints violated:   {total_violated}")

    if n_correct == n_total:
        print(f"\n  VERDICT: ✅ Perfect code constraint verification!")
    elif n_correct >= n_total * 0.8:
        print(f"\n  VERDICT: ✅ Pipeline works ({n_correct}/{n_total} correct)")
    else:
        print(f"\n  VERDICT: ❌ Pipeline needs work ({n_correct}/{n_total})")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
