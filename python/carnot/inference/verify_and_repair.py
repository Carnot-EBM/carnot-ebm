"""LLM-EBM verify-and-repair pipeline.

**Researcher summary:**
    Takes LLM text output, parses it into a JAX array, verifies it against
    constraint energy, repairs violations via gradient descent, rounds to
    discrete, and re-verifies. The core "anti-hallucination" loop.

**Detailed explanation for engineers:**
    This module is the glue between LLM text output and the EBM verification
    machinery in ``carnot.verify``. It does NOT call an LLM — it takes the
    LLM's text output as input and processes it through the EBM pipeline.

    The pipeline is:

    1. **Parse**: Convert LLM text to a JAX array (e.g., "x1=True x2=False"
       becomes ``jnp.array([1.0, 0.0])``)
    2. **Verify**: Score against ComposedEnergy → VerificationResult
    3. **Repair**: If violations, run ``repair()`` from ``carnot.verify``
    4. **Round**: Snap continuous values to discrete (e.g., threshold at 0.5)
    5. **Re-verify**: Score the rounded result
    6. **Return**: Complete VerifyRepairResult with all intermediate results

    The parsers support multiple LLM output formats because LLMs produce
    inconsistent formatting. Robustness to format variation is important.

Spec: REQ-INFER-003, REQ-INFER-004, SCENARIO-INFER-005
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import jax
import jax.numpy as jnp

from carnot.verify.constraint import ComposedEnergy, VerificationResult, repair


@dataclass
class VerifyRepairResult:
    """Result of the verify-and-repair pipeline.

    **Researcher summary:**
        Contains initial score, repaired score, rounded score, repair
        trajectory, and step count. Tells you exactly how much the EBM
        improved the LLM's hallucinated answer.

    **Detailed explanation for engineers:**
        Fields:
        - ``initial_assignment``: The parsed LLM output as JAX array
        - ``initial_verification``: VerificationResult before repair
        - ``repaired_assignment``: After gradient repair (continuous)
        - ``repaired_verification``: VerificationResult after repair
        - ``rounded_assignment``: After rounding to discrete
        - ``rounded_verification``: VerificationResult after rounding
        - ``n_repair_steps``: Steps actually taken
        - ``repair_trajectory``: Energy at each repair step

    Spec: REQ-INFER-004
    """

    initial_assignment: Any = None  # jax.Array
    initial_verification: VerificationResult | None = None
    repaired_assignment: Any = None  # jax.Array
    repaired_verification: VerificationResult | None = None
    rounded_assignment: Any = None  # jax.Array
    rounded_verification: VerificationResult | None = None
    n_repair_steps: int = 0
    repair_trajectory: list[float] = field(default_factory=list)


def parse_llm_sat_assignment(text: str, n_vars: int) -> jax.Array:
    """Extract SAT variable assignments from LLM text output.

    **Researcher summary:**
        Parses LLM text into a float array of {0.0, 1.0} values.
        Supports multiple common output formats.

    **Detailed explanation for engineers:**
        LLMs produce inconsistent formatting, so we try multiple patterns:

        1. Space-separated 0/1: "1 0 1 0 1"
        2. Space-separated T/F: "T F T F T" or "True False True"
        3. Named: "x1=True, x2=False" or "x1=1, x2=0"

        We extract all boolean-like values and check that the count
        matches n_vars. Raises ValueError on failure.

    Args:
        text: LLM output text containing variable assignments.
        n_vars: Expected number of variables.

    Returns:
        Float array of shape (n_vars,) with values in {0.0, 1.0}.

    Raises:
        ValueError: If parsing fails or wrong number of variables.

    Spec: REQ-INFER-003
    """
    text = text.strip()

    # Strategy 1: space-separated 0/1
    tokens = text.replace(",", " ").split()
    if all(t in ("0", "1") for t in tokens) and len(tokens) == n_vars:
        return jnp.array([float(t) for t in tokens])

    # Strategy 2: T/F or True/False
    bool_map = {"t": 1.0, "f": 0.0, "true": 1.0, "false": 0.0}
    lower_tokens = [t.lower().rstrip(",") for t in tokens]
    if all(t in bool_map for t in lower_tokens) and len(lower_tokens) == n_vars:
        return jnp.array([bool_map[t] for t in lower_tokens])

    # Strategy 3: named format x1=True or x1=1
    # Find ALL matches, then take the LAST n_vars (LLM may include reasoning before)
    named_pattern = re.findall(
        r"x?\d+\s*[=:]\s*(true|false|[01])",
        text,
        re.IGNORECASE,
    )
    if len(named_pattern) >= n_vars:
        # Take the last n_vars matches (skip reasoning preamble)
        last_n = named_pattern[-n_vars:]
        values = []
        for val in last_n:
            val_lower = val.lower()
            if val_lower in ("true", "1"):
                values.append(1.0)
            else:
                values.append(0.0)
        return jnp.array(values)

    msg = f"Could not parse {n_vars} SAT variables from: {text[:200]}"
    raise ValueError(msg)


def parse_llm_coloring(text: str, n_nodes: int) -> jax.Array:
    """Extract graph coloring from LLM text output.

    **Researcher summary:**
        Parses LLM text into a float array of color indices.

    **Detailed explanation for engineers:**
        Supports:
        1. Space-separated integers: "0 1 2 0 1"
        2. Named color format: "node 0: 1, node 1: 2" (extracts integers)

    Args:
        text: LLM output text containing color assignments.
        n_nodes: Expected number of nodes.

    Returns:
        Float array of shape (n_nodes,) with integer color values.

    Raises:
        ValueError: If parsing fails or wrong number of nodes.

    Spec: REQ-INFER-003
    """
    text = text.strip()

    # Strategy 1: space-separated integers
    tokens = text.replace(",", " ").split()
    try:
        values = [float(int(t)) for t in tokens]
        if len(values) == n_nodes:
            return jnp.array(values)
    except ValueError:
        pass

    # Strategy 2: "node N: C" or "nodeN=C" pattern — take last n_nodes matches
    pattern = re.findall(r"node\s*\d+\s*[=:]\s*(\d+)", text, re.IGNORECASE)
    if len(pattern) >= n_nodes:
        last_n = pattern[-n_nodes:]
        return jnp.array([float(int(v)) for v in last_n])

    msg = f"Could not parse {n_nodes} node colors from: {text[:200]}"
    raise ValueError(msg)


def verify_and_repair(
    assignment: jax.Array,
    energy: ComposedEnergy,
    step_size: float = 0.1,
    max_repair_steps: int = 200,
    round_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> VerifyRepairResult:
    """Full verify-and-repair pipeline.

    **Researcher summary:**
        Verify LLM output → repair violations → round to discrete →
        re-verify → return complete result with trajectory.

    **Detailed explanation for engineers:**
        This is the core "anti-hallucination" function:

        1. Verify the initial assignment against the energy function
        2. If any constraints are violated, run gradient repair (reusing
           ``repair()`` from ``carnot.verify.constraint``)
        3. If a round_fn is provided, snap the repaired result to discrete
           values (e.g., threshold at 0.5 for SAT)
        4. Re-verify the rounded result
        5. Return everything in a VerifyRepairResult

        The repair_trajectory records total energy at each step, allowing
        callers to visualize the improvement.

    Args:
        assignment: Initial assignment (JAX array from LLM output).
        energy: ComposedEnergy encoding the constraints.
        step_size: Gradient descent step size for repair.
        max_repair_steps: Maximum number of repair iterations.
        round_fn: Optional function to snap continuous to discrete.

    Returns:
        VerifyRepairResult with full pipeline results.

    Spec: REQ-INFER-004, SCENARIO-INFER-005
    """
    result = VerifyRepairResult(initial_assignment=assignment)

    # Step 1: Verify initial
    result.initial_verification = energy.verify(assignment)

    # Step 2: Repair if violated
    if result.initial_verification.verdict.verified:
        result.repaired_assignment = assignment
        result.repaired_verification = result.initial_verification
        result.n_repair_steps = 0
    else:
        repaired, history = repair(
            energy,
            assignment,
            step_size=step_size,
            max_steps=max_repair_steps,
        )
        result.repaired_assignment = repaired
        result.repaired_verification = energy.verify(repaired)
        result.n_repair_steps = len(history)

        # Build trajectory from repair history
        result.repair_trajectory = [float(h.total_energy) for h in history]

    # Step 3: Round to discrete
    if round_fn is not None:
        result.rounded_assignment = round_fn(result.repaired_assignment)
        result.rounded_verification = energy.verify(result.rounded_assignment)
    else:
        result.rounded_assignment = result.repaired_assignment
        result.rounded_verification = result.repaired_verification

    return result
