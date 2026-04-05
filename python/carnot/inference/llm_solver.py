"""LLM-powered solver for constraint satisfaction problems.

**Researcher summary:**
    Sends SAT and graph coloring problems to an LLM via OpenAI-compatible API,
    parses the response, and feeds it through the verify-and-repair pipeline.
    This is the first concrete "LLM hallucinates -> EBM repairs" measurement.

**Detailed explanation for engineers:**
    This module connects the constraint satisfaction pipeline to a real LLM.
    The workflow is:

    1. **Prompt construction**: Build a clear, structured prompt that presents
       the constraint problem (SAT clauses or graph edges) and asks for a
       specific output format that the parsers can handle.

    2. **LLM call**: Send the prompt to any OpenAI-compatible API endpoint
       (Claude API bridge, vLLM, Ollama, OpenAI, etc.) using the openai SDK.

    3. **Parse + verify + repair**: Feed the LLM's text response through
       parse_llm_sat_assignment() or parse_llm_coloring(), then through
       verify_and_repair().

    The LLM import is lazy (same pattern as hypothesis_generator.py) so the
    module works without the openai package installed — it just returns errors.

Spec: REQ-INFER-006, REQ-INFER-013
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from carnot.inference.multi_start import multi_start_repair
from carnot.inference.verify_and_repair import (
    VerifyRepairResult,
    parse_llm_coloring,
    parse_llm_sat_assignment,
    verify_and_repair,
)
from carnot.verify.graph_coloring import build_coloring_energy
from carnot.verify.sat import SATClause, build_sat_energy

logger = logging.getLogger(__name__)


@dataclass
class LLMSolverConfig:
    """Configuration for the LLM solver.

    **Researcher summary:**
        API endpoint, model name, temperature, and auth for the LLM.

    **Detailed explanation for engineers:**
        - ``api_base``: URL of the OpenAI-compatible API endpoint.
        - ``model``: Model identifier (e.g., "sonnet", "gpt-4").
        - ``api_key``: API key. For Claude bridge, "not-needed".
        - ``temperature``: Low (0.3) for reasoning tasks.

    Spec: REQ-INFER-006
    """

    api_base: str = "http://localhost:8080/v1"
    model: str = "sonnet"
    api_key: str = "not-needed"
    temperature: float = 0.3


def _build_sat_prompt(clauses: list[SATClause], n_vars: int) -> str:
    """Build a prompt presenting a SAT problem to an LLM.

    Spec: REQ-INFER-006
    """
    clause_strs = []
    for i, clause in enumerate(clauses):
        lits = []
        for var_idx, is_negated in clause.literals:
            var_name = f"x{var_idx + 1}"  # 0-based to 1-based for readability
            lits.append(f"NOT {var_name}" if is_negated else var_name)
        clause_strs.append(f"  Clause {i + 1}: ({' OR '.join(lits)})")

    clauses_text = "\n".join(clause_strs)

    return (
        f"Solve this SAT (Boolean Satisfiability) problem.\n\n"
        f"There are {n_vars} variables: x1 through x{n_vars}.\n"
        f"Find an assignment of True/False to each variable that satisfies "
        f"ALL of the following {len(clauses)} clauses:\n\n"
        f"{clauses_text}\n\n"
        f"Respond with EXACTLY this format, one variable per line:\n"
        f"x1=True\n"
        f"x2=False\n"
        f"... (for all {n_vars} variables)\n\n"
        f"Think step by step, then provide the assignment."
    )


def _build_coloring_prompt(
    edges: list[tuple[int, int]],
    n_nodes: int,
    n_colors: int,
) -> str:
    """Build a prompt presenting a graph coloring problem to an LLM.

    Spec: REQ-INFER-006
    """
    edge_strs = [f"  ({a}, {b})" for a, b in edges]
    edges_text = "\n".join(edge_strs)

    return (
        f"Solve this graph coloring problem.\n\n"
        f"There are {n_nodes} nodes (0 to {n_nodes - 1}) and "
        f"{len(edges)} edges.\n"
        f"Available colors: {n_colors} colors (numbered 0 to {n_colors - 1})\n\n"
        f"Edges (connected nodes must have DIFFERENT colors):\n"
        f"{edges_text}\n\n"
        f"Assign a color number to each node.\n\n"
        f"Respond with EXACTLY space-separated color numbers:\n"
        f"0 1 2 0 1 ... (one number per node)\n"
    )


def solve_sat_with_llm(
    config: LLMSolverConfig,
    clauses: list[SATClause],
    n_vars: int,
) -> str:
    """Send a SAT problem to an LLM and return the raw text response.

    **Researcher summary:**
        Constructs a SAT prompt, calls the LLM API, returns raw text.

    Raises:
        ImportError: If openai package is not installed.

    Spec: REQ-INFER-006
    """
    from openai import OpenAI

    prompt = _build_sat_prompt(clauses, n_vars)
    client = OpenAI(base_url=config.api_base, api_key=config.api_key)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise constraint solver. Provide exact "
                "assignments in the requested format.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=config.temperature,
    )
    return response.choices[0].message.content or ""


def solve_coloring_with_llm(
    config: LLMSolverConfig,
    edges: list[tuple[int, int]],
    n_nodes: int,
    n_colors: int,
) -> str:
    """Send a graph coloring problem to an LLM and return raw text response.

    Spec: REQ-INFER-006
    """
    from openai import OpenAI

    prompt = _build_coloring_prompt(edges, n_nodes, n_colors)
    client = OpenAI(base_url=config.api_base, api_key=config.api_key)
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise constraint solver. Provide exact "
                "assignments in the requested format.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=config.temperature,
    )
    return response.choices[0].message.content or ""


def run_llm_sat_experiment(
    config: LLMSolverConfig,
    clauses: list[SATClause],
    n_vars: int,
    repair_step_size: float = 0.1,
    repair_max_steps: int = 200,
    n_starts: int = 1,
) -> VerifyRepairResult:
    """Full pipeline: LLM solves SAT -> parse -> verify -> repair -> certify.

    **Researcher summary:**
        End-to-end: send SAT to LLM, parse, verify, repair, certify.
        When n_starts > 1, uses multi-start repair to escape local minima.

    **Detailed explanation for engineers:**
        When n_starts=1 (default), uses single-start verify_and_repair as before.
        When n_starts > 1, uses multi_start_repair which runs gradient repair
        from N randomly perturbed starting points and selects the lowest-energy
        result. This dramatically improves repair success on hard SAT instances
        where single-start gets stuck in local minima.

    Spec: REQ-INFER-006, REQ-INFER-009, SCENARIO-INFER-007
    """
    try:
        raw_response = solve_sat_with_llm(config, clauses, n_vars)
    except ImportError:
        logger.warning("openai not installed")
        return VerifyRepairResult()
    except Exception:
        logger.exception("LLM API call failed")
        return VerifyRepairResult()

    try:
        assignment = parse_llm_sat_assignment(raw_response, n_vars)
    except ValueError:
        logger.warning("Could not parse LLM response: %s...", raw_response[:200])
        return VerifyRepairResult()

    energy = build_sat_energy(clauses, n_vars)

    def round_fn(x: Any) -> Any:
        return jnp.where(x >= 0.5, 1.0, 0.0)

    if n_starts > 1:
        # Multi-start repair: run from N perturbed starts, pick lowest energy
        result = VerifyRepairResult(initial_assignment=assignment)
        result.initial_verification = energy.verify(assignment)

        if result.initial_verification.verdict.verified:
            # Already correct, no repair needed
            result.repaired_assignment = assignment
            result.repaired_verification = result.initial_verification
            result.rounded_assignment = assignment
            result.rounded_verification = result.initial_verification
            result.n_repair_steps = 0
        else:
            ms_result = multi_start_repair(
                assignment,
                energy,
                n_starts=n_starts,
                step_size=repair_step_size,
                max_repair_steps=repair_max_steps,
                round_fn=round_fn,
            )
            result.repaired_assignment = ms_result.best_x
            result.rounded_assignment = ms_result.best_x
            result.rounded_verification = energy.verify(ms_result.best_x)
            result.repaired_verification = result.rounded_verification
            result.n_repair_steps = len(ms_result.best_history)
            result.repair_trajectory = [float(h.total_energy) for h in ms_result.best_history]

        return result

    return verify_and_repair(
        assignment,
        energy,
        step_size=repair_step_size,
        max_repair_steps=repair_max_steps,
        round_fn=round_fn,
    )


def run_llm_coloring_experiment(
    config: LLMSolverConfig,
    edges: list[tuple[int, int]],
    n_nodes: int,
    n_colors: int,
    repair_step_size: float = 0.1,
    repair_max_steps: int = 200,
) -> VerifyRepairResult:
    """Full pipeline: LLM solves graph coloring -> parse -> verify -> repair.

    Spec: REQ-INFER-006, SCENARIO-INFER-007
    """
    try:
        raw_response = solve_coloring_with_llm(config, edges, n_nodes, n_colors)
    except ImportError:
        logger.warning("openai not installed")
        return VerifyRepairResult()
    except Exception:
        logger.exception("LLM API call failed")
        return VerifyRepairResult()

    try:
        coloring = parse_llm_coloring(raw_response, n_nodes)
    except ValueError:
        logger.warning("Could not parse LLM response: %s...", raw_response[:200])
        return VerifyRepairResult()

    energy = build_coloring_energy(edges, n_nodes, n_colors)

    def round_fn(x: Any) -> Any:
        return jnp.round(jnp.clip(x, 0.0, float(n_colors - 1)))

    return verify_and_repair(
        coloring,
        energy,
        step_size=repair_step_size,
        max_repair_steps=repair_max_steps,
        round_fn=round_fn,
    )


# ---------------------------------------------------------------------------
# EBM-Guided Iterative Refinement
# ---------------------------------------------------------------------------


@dataclass
class RefinementResult:
    """Result of EBM-guided iterative refinement.

    **Researcher summary:**
        Tracks how the LLM improves its answer across iterations, guided
        by EBM violation feedback. The key metric is how many iterations
        it takes to reach energy=0 (all constraints satisfied).

    **Detailed explanation for engineers:**
        This changes the relationship from "LLM then EBM" to "LLM WITH EBM."
        Instead of a single generate-then-check, the EBM feeds violation
        details back to the LLM, which regenerates. This loop continues
        until the EBM certifies correctness (energy=0) or max iterations.

    Spec: REQ-INFER-013
    """

    iterations: int = 0
    final_verified: bool = False
    final_energy: float = 0.0
    energy_trajectory: list[float] = field(default_factory=list)
    violation_trajectory: list[int] = field(default_factory=list)
    final_code: str = ""
    final_response: str = ""


def _build_code_violation_feedback(
    test_results: list[dict[str, Any]],
) -> str:
    """Build feedback for code verification failures.

    Spec: REQ-INFER-013
    """
    failures = [t for t in test_results if not t["passed"]]
    if not failures:
        return ""

    parts = [f"Your code failed {len(failures)} test case(s):"]
    for t in failures[:10]:
        if t.get("error"):
            parts.append(f"  - {t['input']}: raised {t['error']}")
        else:
            parts.append(f"  - {t['input']}: expected {t['expected']}, got {t['actual']}")

    parts.append("")
    parts.append("Fix the function and resubmit. Return ONLY the function definition.")
    return "\n".join(parts)


def iterative_refine_code(
    config: LLMSolverConfig,
    task_description: str,
    func_name: str,
    test_cases: list[tuple[tuple, Any]],
    expected_type: type = int,
    max_iterations: int = 5,
) -> RefinementResult:
    """EBM-guided iterative code refinement.

    **Researcher summary:**
        LLM generates code -> EBM verifies -> if violations, feed them back
        to LLM -> LLM regenerates -> repeat until verified or max iterations.
        This is the core "LLM WITH EBM" loop.

    **Detailed explanation for engineers:**
        The loop:
        1. Ask LLM to implement the function
        2. Execute the code on test cases (EBM verification)
        3. If all tests pass (energy=0): done, return certified result
        4. If tests fail: build a feedback prompt listing exactly which
           tests failed and how, send it back to the LLM
        5. The LLM sees its mistakes and generates a corrected version
        6. Repeat until success or max_iterations

        This is fundamentally different from generate-then-check because
        the EBM actively GUIDES the LLM toward correctness. The LLM
        gets specific, deterministic feedback that it cannot get from
        its own self-evaluation.

    Args:
        config: LLM API configuration.
        task_description: The function specification to implement.
        func_name: Expected function name.
        test_cases: List of (args_tuple, expected_output) pairs.
        expected_type: Expected return type.
        max_iterations: Maximum refinement attempts.

    Returns:
        RefinementResult with trajectory and final verification state.

    Spec: REQ-INFER-013
    """
    import re

    from carnot.verify.python_types import safe_exec_function

    result = RefinementResult()
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a precise Python programmer. Write ONLY the function "
                "definition — no explanation, no imports, no test code. "
                "Just the def statement and its body. "
                "If given feedback about test failures, fix the function."
            ),
        },
        {"role": "user", "content": task_description},
    ]

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai not installed")
        return result

    client = OpenAI(base_url=config.api_base, api_key=config.api_key)

    for iteration in range(max_iterations):
        result.iterations = iteration + 1

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
            )
            raw = response.choices[0].message.content or ""
        except Exception:
            logger.exception("LLM call failed at iteration %d", iteration)
            break

        # Extract code
        match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
            code = match.group(1).strip() if match else raw.strip()

        result.final_code = code
        result.final_response = raw

        # Run tests
        test_results: list[dict[str, Any]] = []
        for args, expected in test_cases:
            actual, error = safe_exec_function(code, func_name, args)
            passed = error is None and actual == expected
            test_results.append(
                {
                    "input": args,
                    "expected": expected,
                    "actual": actual,
                    "error": str(error) if error else None,
                    "passed": passed,
                }
            )

        n_passed = sum(1 for t in test_results if t["passed"])
        n_failed = len(test_cases) - n_passed
        energy = n_failed / max(len(test_cases), 1)

        result.energy_trajectory.append(energy)
        result.violation_trajectory.append(n_failed)
        result.final_energy = energy
        result.final_verified = n_failed == 0

        logger.info(
            "Iteration %d: %d/%d tests passed (energy=%.4f)",
            iteration,
            n_passed,
            len(test_cases),
            energy,
        )

        # If all tests pass, we're done
        if n_failed == 0:
            logger.info("All tests passed at iteration %d", iteration)
            break

        # Build feedback and add to conversation
        feedback = _build_code_violation_feedback(test_results)
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": feedback})

    return result


def iterative_refine_with_properties(
    config: LLMSolverConfig,
    task_description: str,
    func_name: str,
    test_cases: list[tuple[tuple, Any]],
    properties: list[dict[str, Any]],
    expected_type: type = int,
    max_iterations: int = 5,
    property_samples: int = 100,
    property_seed: int = 42,
) -> RefinementResult:
    """EBM-guided iterative refinement with BOTH test cases AND property tests.

    **Researcher summary:**
        Combines deterministic test cases (input/output pairs) with random
        property-based tests (commutativity, idempotency, bounds, etc.).
        After each LLM attempt, runs BOTH kinds of verification and feeds
        ALL failures back to the LLM as a single unified feedback prompt.
        Energy = combined failure rate across both test types.

    **Detailed explanation for engineers:**
        ``iterative_refine_code`` only checks specific input/output pairs —
        the kind of tests that appear in every LLM training set. This function
        adds property-based testing (random inputs checked against invariant
        properties) so the LLM also gets feedback on edge cases, large inputs,
        and structural properties it would never see from hand-written tests.

        The loop per iteration:
        1. Ask LLM to implement the function (or fix it based on feedback).
        2. Run all deterministic test cases — collect pass/fail for each.
        3. Run property-based tests — generate random inputs, check invariants.
        4. Combine failures from both sources into a single feedback prompt.
        5. If zero failures across both: done, certified correct.
        6. Otherwise: feed combined feedback to LLM and repeat.

        Energy is computed as total failures (test cases + property tests)
        divided by total checks (len(test_cases) + n_property_tests). This
        gives a single 0-to-1 score that captures both kinds of correctness.

    Args:
        config: LLM API configuration.
        task_description: The function specification to implement.
        func_name: Expected function name.
        test_cases: List of (args_tuple, expected_output) pairs.
        properties: List of property dicts for property_test(). Each dict has
            ``name``, ``gen_args`` (callable(rng) -> tuple), ``check``
            (callable(result, *args) -> bool), and optional ``description``.
        expected_type: Expected return type.
        max_iterations: Maximum refinement attempts.
        property_samples: Number of random samples per property.
        property_seed: Random seed for reproducible property tests.

    Returns:
        RefinementResult with trajectory and final verification state.
        The violation_trajectory counts combined failures from both sources.

    Spec: REQ-INFER-013
    """
    import re

    from carnot.verify.property_test import format_violations_for_llm, property_test
    from carnot.verify.python_types import safe_exec_function

    result = RefinementResult()
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a precise Python programmer. Write ONLY the function "
                "definition — no explanation, no imports, no test code. "
                "Just the def statement and its body. "
                "If given feedback about test failures, fix the function."
            ),
        },
        {"role": "user", "content": task_description},
    ]

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai not installed")
        return result

    client = OpenAI(base_url=config.api_base, api_key=config.api_key)

    for iteration in range(max_iterations):
        result.iterations = iteration + 1

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
            )
            raw = response.choices[0].message.content or ""
        except Exception:
            logger.exception("LLM call failed at iteration %d", iteration)
            break

        # Extract code from response (may be in markdown code blocks)
        match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            match = re.search(r"```\s*\n(.*?)```", raw, re.DOTALL)
            code = match.group(1).strip() if match else raw.strip()

        result.final_code = code
        result.final_response = raw

        # --- Phase 1: Run deterministic test cases ---
        test_results: list[dict[str, Any]] = []
        for args, expected in test_cases:
            actual, error = safe_exec_function(code, func_name, args)
            passed = error is None and actual == expected
            test_results.append(
                {
                    "input": args,
                    "expected": expected,
                    "actual": actual,
                    "error": str(error) if error else None,
                    "passed": passed,
                }
            )

        n_test_passed = sum(1 for t in test_results if t["passed"])
        n_test_failed = len(test_cases) - n_test_passed

        # --- Phase 2: Run property-based tests ---
        prop_result = property_test(
            code,
            func_name,
            properties,
            n_samples=property_samples,
            seed=property_seed,
        )

        # --- Combine results ---
        total_checks = len(test_cases) + prop_result.n_tests
        total_failures = n_test_failed + prop_result.n_failed
        energy = total_failures / max(total_checks, 1)

        result.energy_trajectory.append(energy)
        result.violation_trajectory.append(total_failures)
        result.final_energy = energy
        result.final_verified = total_failures == 0

        logger.info(
            "Iteration %d: test_cases %d/%d, properties %d/%d (energy=%.4f)",
            iteration,
            n_test_passed,
            len(test_cases),
            prop_result.n_passed,
            prop_result.n_tests,
            energy,
        )

        # If everything passes, we're done
        if total_failures == 0:
            logger.info("All checks passed at iteration %d", iteration)
            break

        # --- Build combined feedback ---
        feedback_parts: list[str] = []

        # Deterministic test failures
        tc_feedback = _build_code_violation_feedback(test_results)
        if tc_feedback:
            feedback_parts.append(tc_feedback)

        # Property-based test failures
        prop_feedback = format_violations_for_llm(prop_result)
        if prop_feedback:
            feedback_parts.append(prop_feedback)

        combined_feedback = "\n\n".join(feedback_parts)
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": combined_feedback})

    return result
