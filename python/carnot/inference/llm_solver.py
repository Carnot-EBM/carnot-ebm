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

Spec: REQ-INFER-006
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

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
) -> VerifyRepairResult:
    """Full pipeline: LLM solves SAT -> parse -> verify -> repair -> certify.

    **Researcher summary:**
        End-to-end: send SAT to LLM, parse, verify, repair, certify.

    Spec: REQ-INFER-006, SCENARIO-INFER-007
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
