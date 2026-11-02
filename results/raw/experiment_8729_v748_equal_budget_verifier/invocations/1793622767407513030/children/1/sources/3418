"""Benchmark harness for LLM-EBM verify-and-repair pipeline.

**Researcher summary:**
    Generates random SAT and graph coloring instances, simulates LLM
    output with random assignments, runs verify-and-repair, and reports
    aggregated statistics (initial vs repaired violations, success rate).

**Detailed explanation for engineers:**
    This module provides the tooling to measure how well the EBM
    verify-and-repair pipeline improves over baseline LLM accuracy.
    Since we don't have a live LLM connection here, we simulate LLM
    output with random assignments (which is actually a reasonable
    lower bound — LLMs on SAT perform near-random for hard instances).

    The benchmark generates random instances, runs the pipeline, and
    collects statistics:
    - Initial violations (before repair)
    - Repaired violations (after repair + rounding)
    - Repair success rate (fraction achieving 0 violations)
    - Energy reduction percentage

Spec: REQ-INFER-005
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import jax.numpy as jnp

from carnot.inference.verify_and_repair import verify_and_repair
from carnot.verify.graph_coloring import build_coloring_energy, coloring_to_array
from carnot.verify.sat import SATClause, assignment_to_array, build_sat_energy


@dataclass
class BenchmarkResult:
    """Results from a single benchmark instance.

    Spec: REQ-INFER-005
    """

    instance_id: str
    n_vars: int
    n_constraints: int
    initial_energy: float
    initial_violations: int
    repaired_energy: float
    repaired_violations: int
    repair_steps: int
    energy_reduction_pct: float


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results.

    Spec: REQ-INFER-005
    """

    n_instances: int
    avg_initial_violations: float
    avg_repaired_violations: float
    repair_success_rate: float
    avg_energy_reduction_pct: float
    results: list[BenchmarkResult] = field(default_factory=list)


def generate_random_sat(
    n_vars: int,
    n_clauses: int,
    clause_size: int = 3,
    seed: int = 42,
) -> list[SATClause]:
    """Generate a random k-SAT instance.

    **Detailed explanation for engineers:**
        Creates ``n_clauses`` clauses, each with ``clause_size`` literals.
        Each literal is a random variable (0..n_vars-1) with 50% chance
        of negation. Uses Python random with a seed for reproducibility.

    Spec: REQ-INFER-005
    """
    rng = random.Random(seed)
    clauses: list[SATClause] = []
    for _ in range(n_clauses):
        lits: list[tuple[int, bool]] = []
        vars_used: set[int] = set()
        while len(lits) < clause_size:
            var = rng.randint(0, n_vars - 1)
            if var not in vars_used:
                vars_used.add(var)
                lits.append((var, rng.random() < 0.5))
        clauses.append(SATClause(lits))
    return clauses


def generate_random_graph(
    n_nodes: int,
    edge_probability: float = 0.3,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Generate a random Erdos-Renyi graph.

    Spec: REQ-INFER-005
    """
    rng = random.Random(seed)
    edges: list[tuple[int, int]] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_probability:
                edges.append((i, j))
    return edges


def generate_random_assignment(n_vars: int, seed: int = 42) -> list[bool]:
    """Generate a random boolean assignment (simulates LLM output).

    Spec: REQ-INFER-005
    """
    rng = random.Random(seed)
    return [rng.random() < 0.5 for _ in range(n_vars)]


def _count_violations(verification: object) -> int:
    """Count failing constraints from a VerificationResult."""
    if verification is None:
        return 0
    verdict = getattr(verification, "verdict", None)
    if verdict is None:
        return 0
    failing = getattr(verdict, "failing", [])
    return len(failing)


def run_sat_benchmark(
    n_instances: int = 10,
    n_vars: int = 10,
    n_clauses: int = 30,
    clause_size: int = 3,
    max_repair_steps: int = 50,
    step_size: float = 0.1,
) -> BenchmarkSummary:
    """Run SAT verify-and-repair benchmark.

    **Detailed explanation for engineers:**
        For each instance:
        1. Generate random 3-SAT
        2. Generate random assignment (simulating LLM)
        3. Build SAT energy
        4. Run verify_and_repair with rounding
        5. Collect metrics

    Spec: REQ-INFER-005
    """
    results: list[BenchmarkResult] = []

    for i in range(n_instances):
        clauses = generate_random_sat(n_vars, n_clauses, clause_size, seed=i)
        raw_assignment = generate_random_assignment(n_vars, seed=i + 1000)
        assignment = assignment_to_array(raw_assignment)
        energy = build_sat_energy(clauses, n_vars)

        def round_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(x >= 0.5, 1.0, 0.0)

        vr = verify_and_repair(
            assignment,
            energy,
            step_size=step_size,
            max_repair_steps=max_repair_steps,
            round_fn=round_fn,
        )

        initial_e = float(vr.initial_verification.total_energy) if vr.initial_verification else 0.0
        repaired_e = float(vr.rounded_verification.total_energy) if vr.rounded_verification else 0.0
        initial_v = _count_violations(vr.initial_verification)
        repaired_v = _count_violations(vr.rounded_verification)

        reduction = ((initial_e - repaired_e) / max(initial_e, 1e-10)) * 100

        results.append(
            BenchmarkResult(
                instance_id=f"sat_{i}",
                n_vars=n_vars,
                n_constraints=n_clauses,
                initial_energy=initial_e,
                initial_violations=initial_v,
                repaired_energy=repaired_e,
                repaired_violations=repaired_v,
                repair_steps=vr.n_repair_steps,
                energy_reduction_pct=reduction,
            )
        )

    return _summarize(results, n_instances)


def run_coloring_benchmark(
    n_instances: int = 10,
    n_nodes: int = 8,
    n_colors: int = 3,
    edge_probability: float = 0.3,
    max_repair_steps: int = 50,
    step_size: float = 0.1,
) -> BenchmarkSummary:
    """Run graph coloring verify-and-repair benchmark.

    Spec: REQ-INFER-005
    """
    results: list[BenchmarkResult] = []

    for i in range(n_instances):
        edges = generate_random_graph(n_nodes, edge_probability, seed=i)
        rng = random.Random(i + 1000)
        raw_coloring = [rng.randint(0, n_colors - 1) for _ in range(n_nodes)]
        assignment = coloring_to_array(raw_coloring)
        energy = build_coloring_energy(edges, n_nodes, n_colors)

        def round_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.round(jnp.clip(x, 0.0, float(n_colors - 1)))

        vr = verify_and_repair(
            assignment,
            energy,
            step_size=step_size,
            max_repair_steps=max_repair_steps,
            round_fn=round_fn,
        )

        initial_e = float(vr.initial_verification.total_energy) if vr.initial_verification else 0.0
        repaired_e = float(vr.rounded_verification.total_energy) if vr.rounded_verification else 0.0
        initial_v = _count_violations(vr.initial_verification)
        repaired_v = _count_violations(vr.rounded_verification)

        reduction = ((initial_e - repaired_e) / max(initial_e, 1e-10)) * 100

        results.append(
            BenchmarkResult(
                instance_id=f"coloring_{i}",
                n_vars=n_nodes,
                n_constraints=len(edges),
                initial_energy=initial_e,
                initial_violations=initial_v,
                repaired_energy=repaired_e,
                repaired_violations=repaired_v,
                repair_steps=vr.n_repair_steps,
                energy_reduction_pct=reduction,
            )
        )

    return _summarize(results, n_instances)


def _summarize(results: list[BenchmarkResult], n_instances: int) -> BenchmarkSummary:
    """Build summary from individual results."""
    if not results:
        return BenchmarkSummary(
            n_instances=0,
            avg_initial_violations=0.0,
            avg_repaired_violations=0.0,
            repair_success_rate=0.0,
            avg_energy_reduction_pct=0.0,
        )

    avg_init = sum(r.initial_violations for r in results) / len(results)
    avg_rep = sum(r.repaired_violations for r in results) / len(results)
    success = sum(1 for r in results if r.repaired_violations == 0) / len(results)
    avg_red = sum(r.energy_reduction_pct for r in results) / len(results)

    return BenchmarkSummary(
        n_instances=n_instances,
        avg_initial_violations=avg_init,
        avg_repaired_violations=avg_rep,
        repair_success_rate=success,
        avg_energy_reduction_pct=avg_red,
        results=results,
    )
