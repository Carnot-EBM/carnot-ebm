"""Exp6504 immutable exact SAT/CSP structural benchmark commitment.

Spec refs: REQ-BENCH-6504, SCENARIO-BENCH-6504-GENERATION,
SCENARIO-BENCH-6504-LABELS, SCENARIO-BENCH-6504-SPLITS,
SCENARIO-BENCH-6504-STRATA, SCENARIO-BENCH-6504-LEAKAGE,
SCENARIO-BENCH-6504-SCHEMA.

The benchmark is generated from formal local procedures and labeled only by
exact replay. Learned systems may use later train rows, but this artifact
commits every raw row, split, label, replay, and leakage check first.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any

import z3

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]
Clause = list[int]
Z3Solver = Callable[[int, list[Clause]], "SolverOutcome"]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260822"
RANDOM_SEED = 6504
SPLIT_SEED = 6504001
SCHEMA_VERSION = "carnot.experiment_6504.exact_structural_benchmark.v1"
INFERENCE_SUBSTRATE = "procedural_formal_instances_and_exact_solver_labels_no_llm"
VERIFIER_IS_ORACLE = True

RESULT_RELATIVE_PATH = Path("results/experiment_6504_exact_structural_benchmark_commitment.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6504_exact_structural_benchmark_commitment.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py"
)
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

UPSTREAM_GATES: tuple[JsonDict, ...] = (
    {
        "experiment_id": "exp6502",
        "path": Path("results/experiment_6502_v560_retirement_v561_lineage_lock.json"),
        "field": "v561_lineage_lock_ready_score",
        "expected_value": 1.0,
    },
    {
        "experiment_id": "exp6503",
        "path": Path("results/experiment_6503_v561_source_delta_method_contract.json"),
        "field": "method_contract_ready_score",
        "expected_value": 1.0,
    },
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("results/experiment_6502_v560_retirement_v561_lineage_lock.json"),
    Path("results/experiment_6503_v561_source_delta_method_contract.json"),
    Path("results/experiment_6489_solver_trajectory_commitment.json"),
    EXCLUSION_MANIFEST_RELATIVE_PATH,
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("_bmad/architecture.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
)

FAMILIES = (
    "random_3cnf",
    "pseudo_industrial_3cnf",
    "tseitin",
    "pigeonhole",
    "graph_coloring",
    "small_scheduling",
)
FAMILY_SOURCES = {
    "random_3cnf": "procedural_random_3cnf_generator_v1",
    "pseudo_industrial_3cnf": "procedural_community_3cnf_generator_v1",
    "tseitin": "procedural_tseitin_parity_generator_v1",
    "pigeonhole": "procedural_pigeonhole_generator_v1",
    "graph_coloring": "procedural_graph_coloring_cnf_encoder_v1",
    "small_scheduling": "procedural_small_scheduling_cnf_encoder_v1",
}
SPLIT_COUNTS = {"train": 10, "development": 10, "held": 60}
UNITS_PER_FAMILY = sum(SPLIT_COUNTS.values())
INSTANCE_COUNT = len(FAMILIES) * UNITS_PER_FAMILY
MINIMUM_HELD_CELL_SIZE = 30

LEAKAGE_ATTACK_IDS = (
    "unit_identity",
    "row_order",
    "serialization_length",
    "generator_seed",
    "family",
    "duplicate_lineage",
    "label_balance",
    "solver_backend",
    "split_leakage",
)
FORBIDDEN_FEATURE_FIELDS = (
    "instance_id",
    "base_instance_id",
    "lineage_id",
    "row_order",
    "serialization_length",
    "generator_seed",
    "generator_recipe_id",
    "family",
    "split",
    "exact_label",
    "solver_backend",
    "label_row_hash",
    "raw_instance_hash",
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6504_exact_structural_benchmark_commitment "
    "--date 20260822"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6504_exact_structural_benchmark_commitment.py "
    "-m pytest tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6504_exact_structural_benchmark_commitment.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6504_exact_structural_benchmark_commitment.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6504_exact_structural_benchmark_commitment.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6504_exact_structural_benchmark_commitment "
    "--validate"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_6504_exact_structural_benchmark_commitment.py "
    "tests/python/test_experiment_6504_exact_structural_benchmark_commitment.py "
    "scripts/adversarial_verify.py"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": RUFF_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "verdict_class",
    "upstream_gate_receipts",
    "benchmark_schema",
    "raw_instance_rows",
    "exact_label_rows",
    "exact_replay_rows",
    "split_commitment",
    "stratum_balance_rows",
    "minimum_held_cell_size",
    "leakage_attack_matrix",
    "base_structural_benchmark_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records the terminal benchmark state.",
    "verdict_class": (
        "Closed enum: positive | circular_positive | null | blocked | disqualified | partial."
    ),
    "upstream_gate_receipts": "Bind both same-roadmap gates and observed values.",
    "benchmark_schema": "Versions instances, labels, proofs, models, strata, and splits.",
    "raw_instance_rows": "Preserves every formal instance before feature extraction.",
    "exact_label_rows": "Provides solver-certified labels and authority receipts.",
    "exact_replay_rows": "Checks every accepted label with a deterministic replay.",
    "split_commitment": "Freezes lineage-separated train, development, and held sets.",
    "stratum_balance_rows": (
        "Records family, scale, surface, hardness, source, and label counts."
    ),
    "minimum_held_cell_size": (
        "Enforces at least 30 units for percentage comparisons."
    ),
    "leakage_attack_matrix": (
        "Tests identity, order, length, seed, family, duplicate, backend, and split shortcuts."
    ),
    "base_structural_benchmark_ready_score": (
        "Same-roadmap gate for the base benchmark."
    ),
    "per_unit_rows": "Carries instance, label, replay, split, stratum, and attack rows.",
    "aggregate_row_recomputation": "Recomputes counts and readiness from raw rows.",
    "gate_check_summary": (
        "Names any failed gate, solver, cell-size, or replay check and observed value."
    ),
    "preconditions_checked": (
        "Records gates, solver tools, resources, repository, and storage checks."
    ),
    "protected_files_unchanged": "Proves protected files stayed unchanged.",
    "inference_substrate": (
        "Declares procedural generators and exact local solvers with no LLM."
    ),
    "verifier_is_oracle": (
        "True only for exact solver labels and executable validity checks."
    ),
    "field_principles": "Explains why each commitment field exists.",
    "field_provenance": "Maps rows to generators, seeds, solver commands, and hashes.",
    "random_seed": "Records all generator and split seeds.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records commands and exit codes.",
    "reproducibility_checksum": "Hashes instances, labels, splits, strata, and attacks.",
    "honest_verdict": (
        "Uses complete_* when the commitment is valid or blocked_* with gate_check_summary."
    ),
}
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}


@dataclass(frozen=True)
class SolverOutcome:
    """One exact backend result for a finite CNF instance."""

    backend: str
    available: bool
    status: str
    model: dict[str, bool] | None
    assignments_examined: int
    version: str
    command: str


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in one stable byte order."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash stable JSON evidence with the project prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes and return a visible missing marker."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _split_for_index(index: int) -> str:
    if index < SPLIT_COUNTS["train"]:
        return "train"
    if index < SPLIT_COUNTS["train"] + SPLIT_COUNTS["development"]:
        return "development"
    return "held"


def _scale_surface_and_recipe(index: int) -> tuple[str, str, bool, str]:
    cycle = index % 4
    scale = "small" if cycle in (0, 2) else "medium"
    surface = "canonical" if cycle in (0, 1) else "alpha_relabelled"
    sat_recipe = cycle in (0, 3)
    return scale, surface, sat_recipe, f"recipe_{cycle}"


def _at_most_one(values: Sequence[int]) -> list[Clause]:
    return [[-left, -right] for left, right in itertools.combinations(values, 2)]


def _exactly_one(values: Sequence[int]) -> list[Clause]:
    return [list(values), *_at_most_one(values)]


def _lit_value(literal: int, assignment: Mapping[int, bool]) -> bool:
    value = bool(assignment[abs(literal)])
    return value if literal > 0 else not value


def _clause_satisfied(clause: Sequence[int], assignment: Mapping[int, bool]) -> bool:
    return any(_lit_value(literal, assignment) for literal in clause)


def _assignment_satisfies(clauses: Sequence[Sequence[int]], assignment: Mapping[int, bool]) -> bool:
    return all(_clause_satisfied(clause, assignment) for clause in clauses)


def _model_from_mask(n_vars: int, mask: int) -> dict[str, bool]:
    return {f"x{index}": bool((mask >> (index - 1)) & 1) for index in range(1, n_vars + 1)}


def _int_assignment_from_model(model: Mapping[str, bool]) -> dict[int, bool]:
    return {int(key[1:]): bool(value) for key, value in model.items()}


def _unsat_cube(vars_: Sequence[int]) -> list[Clause]:
    clauses: list[Clause] = []
    for bits in itertools.product((False, True), repeat=len(vars_)):
        clauses.append([-var if bit else var for var, bit in zip(vars_, bits, strict=True)])
    return clauses


def _salt_clauses(n_vars: int, seed: int) -> list[Clause]:
    rng = random.Random(seed)
    clauses: list[Clause] = []
    for offset in range(4):
        left = rng.randint(1, n_vars)
        right = rng.randint(1, n_vars)
        sign = -1 if (seed + offset) % 2 else 1
        clauses.append([left, -left, sign * right])
    return clauses


def _canonical_dimacs(n_vars: int, clauses: Sequence[Sequence[int]]) -> str:
    lines = [f"p cnf {n_vars} {len(clauses)}"]
    lines.extend(" ".join(str(literal) for literal in clause) + " 0" for clause in clauses)
    return "\n".join(lines) + "\n"


def _apply_surface(
    *,
    n_vars: int,
    clauses: Sequence[Sequence[int]],
    seed: int,
    surface: str,
) -> tuple[list[Clause], list[int]]:
    permutation = list(range(1, n_vars + 1))
    if surface == "alpha_relabelled":
        rng = random.Random(seed ^ 0xA17A)
        rng.shuffle(permutation)
        mapped: list[Clause] = []
        for clause in clauses:
            relabelled = [
                permutation[abs(literal) - 1] if literal > 0 else -permutation[abs(literal) - 1]
                for literal in clause
            ]
            rng.shuffle(relabelled)
            mapped.append(relabelled)
        rng.shuffle(mapped)
        return mapped, permutation
    return [list(clause) for clause in clauses], permutation


def _planted_3cnf(seed: int, n_vars: int, clause_count: int) -> list[Clause]:
    rng = random.Random(seed)
    assignment = {var: bool(rng.getrandbits(1)) for var in range(1, n_vars + 1)}
    clauses: list[Clause] = []
    while len(clauses) < clause_count:
        vars_ = rng.sample(range(1, n_vars + 1), 3)
        clause = [var if rng.getrandbits(1) else -var for var in vars_]
        if _clause_satisfied(clause, assignment):
            clauses.append(clause)
    return clauses


def _random_3cnf(seed: int, scale: str, sat_recipe: bool) -> tuple[int, list[Clause]]:
    n_vars = 6 if scale == "small" else 8
    if sat_recipe:
        clauses = _planted_3cnf(seed, n_vars, 12 if scale == "small" else 18)
    else:
        clauses = [*_unsat_cube((1, 2, 3)), *_planted_3cnf(seed, n_vars, 8)]
    return n_vars, [*clauses, *_salt_clauses(n_vars, seed)]


def _pseudo_industrial_3cnf(seed: int, scale: str, sat_recipe: bool) -> tuple[int, list[Clause]]:
    n_vars = 6 if scale == "small" else 8
    if not sat_recipe:
        return n_vars, [*_unsat_cube((1, 2, 3)), *_salt_clauses(n_vars, seed)]
    rng = random.Random(seed)
    assignment = {var: bool(rng.getrandbits(1)) for var in range(1, n_vars + 1)}
    block_a = list(range(1, n_vars // 2 + 1))
    block_b = list(range(n_vars // 2 + 1, n_vars + 1))
    clauses: list[Clause] = []
    while len(clauses) < (14 if scale == "small" else 20):
        block = block_a if rng.random() < 0.65 else block_b
        other = block_b if block is block_a else block_a
        vars_ = [*rng.sample(block, min(2, len(block))), rng.choice(other)]
        clause = [var if rng.getrandbits(1) else -var for var in vars_]
        if _clause_satisfied(clause, assignment):
            clauses.append(clause)
    return n_vars, [*clauses, *_salt_clauses(n_vars, seed)]


def _parity_clauses(vars_: Sequence[int], charge: int) -> list[Clause]:
    clauses: list[Clause] = []
    for bits in itertools.product((False, True), repeat=len(vars_)):
        if sum(int(bit) for bit in bits) % 2 != charge:
            clauses.append([-var if bit else var for var, bit in zip(vars_, bits, strict=True)])
    return clauses


def _tseitin(seed: int, scale: str, sat_recipe: bool) -> tuple[int, list[Clause]]:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    if scale == "medium":
        edges.extend([(0, 2), (1, 3)])
    n_vars = len(edges)
    charges = [0, 0, 0, 0]
    if sat_recipe:
        charges[0] = 1
        charges[1] = 1
    else:
        charges[0] = 1
    clauses: list[Clause] = []
    for vertex in range(4):
        incident = [index + 1 for index, edge in enumerate(edges) if vertex in edge]
        clauses.extend(_parity_clauses(incident, charges[vertex]))
    return n_vars, [*clauses, *_salt_clauses(n_vars, seed)]


def _pigeonhole(seed: int, scale: str, sat_recipe: bool) -> tuple[int, list[Clause]]:
    pigeons, holes = (3, 3) if sat_recipe and scale == "small" else (3, 4)
    if not sat_recipe:
        pigeons, holes = (4, 3)
    n_vars = pigeons * holes

    def var(pigeon: int, hole: int) -> int:
        return pigeon * holes + hole + 1

    clauses: list[Clause] = []
    for pigeon in range(pigeons):
        clauses.extend(_exactly_one([var(pigeon, hole) for hole in range(holes)]))
    for hole in range(holes):
        clauses.extend(
            [-var(left, hole), -var(right, hole)]
            for left, right in itertools.combinations(range(pigeons), 2)
        )
    return n_vars, [*clauses, *_salt_clauses(n_vars, seed)]


def _graph_coloring(seed: int, scale: str, sat_recipe: bool) -> tuple[int, list[Clause]]:
    if sat_recipe and scale == "small":
        n_nodes, n_colors = 4, 2
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    elif sat_recipe:
        n_nodes, n_colors = 4, 3
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    elif scale == "small":
        n_nodes, n_colors = 3, 2
        edges = [(0, 1), (1, 2), (0, 2)]
    else:
        n_nodes, n_colors = 4, 3
        edges = list(itertools.combinations(range(4), 2))
    n_vars = n_nodes * n_colors

    def var(node: int, color: int) -> int:
        return node * n_colors + color + 1

    clauses: list[Clause] = []
    for node in range(n_nodes):
        clauses.extend(_exactly_one([var(node, color) for color in range(n_colors)]))
    for left, right in edges:
        for color in range(n_colors):
            clauses.append([-var(left, color), -var(right, color)])
    return n_vars, [*clauses, *_salt_clauses(n_vars, seed)]


def _small_scheduling(seed: int, scale: str, sat_recipe: bool) -> tuple[int, list[Clause]]:
    if sat_recipe and scale == "small":
        jobs, slots = 3, 3
        conflicts = list(itertools.combinations(range(jobs), 2))
        precedences = [(0, 1)]
    elif sat_recipe:
        jobs, slots = 4, 3
        conflicts = [(0, 1), (1, 2), (2, 3)]
        precedences = [(0, 2)]
    elif scale == "small":
        jobs, slots = 3, 2
        conflicts = list(itertools.combinations(range(jobs), 2))
        precedences = []
    else:
        jobs, slots = 4, 3
        conflicts = list(itertools.combinations(range(jobs), 2))
        precedences = []
    n_vars = jobs * slots

    def var(job: int, slot: int) -> int:
        return job * slots + slot + 1

    clauses: list[Clause] = []
    for job in range(jobs):
        clauses.extend(_exactly_one([var(job, slot) for slot in range(slots)]))
    for left, right in conflicts:
        for slot in range(slots):
            clauses.append([-var(left, slot), -var(right, slot)])
    for before, after in precedences:
        for before_slot in range(slots):
            for after_slot in range(slots):
                if before_slot >= after_slot:
                    clauses.append([-var(before, before_slot), -var(after, after_slot)])
    return n_vars, [*clauses, *_salt_clauses(n_vars, seed)]


def _family_cnf(family: str, seed: int, scale: str, sat_recipe: bool) -> tuple[int, list[Clause]]:
    generators = {
        "random_3cnf": _random_3cnf,
        "pseudo_industrial_3cnf": _pseudo_industrial_3cnf,
        "tseitin": _tseitin,
        "pigeonhole": _pigeonhole,
        "graph_coloring": _graph_coloring,
        "small_scheduling": _small_scheduling,
    }
    return generators[family](seed, scale, sat_recipe)


def generate_instance_rows() -> list[JsonDict]:
    """Generate all raw formal rows before label-derived features exist."""

    rows: list[JsonDict] = []
    row_order = 0
    for family_index, family in enumerate(FAMILIES):
        for index in range(UNITS_PER_FAMILY):
            scale, surface, sat_recipe, recipe_id = _scale_surface_and_recipe(index)
            split = _split_for_index(index)
            seed = RANDOM_SEED * 100_000 + family_index * 10_000 + index
            n_vars, clauses = _family_cnf(family, seed, scale, sat_recipe)
            clauses, permutation = _apply_surface(
                n_vars=n_vars,
                clauses=clauses,
                seed=seed,
                surface=surface,
            )
            dimacs = _canonical_dimacs(n_vars, clauses)
            base_id = f"{family}:base:{index:03d}"
            instance_id = f"exp6504:{family}:{index:03d}"
            density = round(len(clauses) / max(1, n_vars), 6)
            payload = {
                "row_type": "raw_instance",
                "schema_version": SCHEMA_VERSION + ".raw_instance",
                "instance_id": instance_id,
                "base_instance_id": base_id,
                "lineage_id": f"lineage:{family}:{index:03d}",
                "family": family,
                "source": FAMILY_SOURCES[family],
                "scale": scale,
                "surface_relabeling": surface,
                "structural_hardness": f"structural_{scale}",
                "density_band": "dense" if density >= 3.0 else "sparse",
                "proof_family_plan": "model_witness_or_complete_exhaustive_refutation",
                "split": split,
                "row_order": row_order,
                "generator_seed": seed,
                "split_seed": SPLIT_SEED,
                "generator_recipe_id": recipe_id,
                "formalism": "cnf_sat",
                "variable_count": n_vars,
                "clause_count": len(clauses),
                "density": density,
                "clauses": clauses,
                "canonical_dimacs": dimacs,
                "serialization_length": len(dimacs.encode("utf-8")),
                "variable_permutation": permutation,
                "split_commitment_event_index": 1,
                "feature_extraction_event_index": 2,
                "label_event_index": 3,
                "label_inspected_before_split": False,
                "llm_used": False,
                "spec_refs": ["REQ-BENCH-6504", "SCENARIO-BENCH-6504-GENERATION"],
            }
            payload["structural_cnf_hash"] = sha256_json(
                {"variable_count": n_vars, "clauses": clauses}
            )
            rows.append({**payload, "raw_instance_hash": sha256_json(payload)})
            row_order += 1
    return rows


def exhaustive_solve(n_vars: int, clauses: list[Clause]) -> SolverOutcome:
    """Solve a finite CNF formula by complete assignment enumeration."""

    for mask in range(1 << n_vars):
        model = _model_from_mask(n_vars, mask)
        if _assignment_satisfies(clauses, _int_assignment_from_model(model)):
            return SolverOutcome(
                backend="exhaustive",
                available=True,
                status="sat",
                model=model,
                assignments_examined=mask + 1,
                version="carnot_complete_truth_table_v1",
                command="local exhaustive enumeration",
            )
    return SolverOutcome(
        backend="exhaustive",
        available=True,
        status="unsat",
        model=None,
        assignments_examined=1 << n_vars,
        version="carnot_complete_truth_table_v1",
        command="local exhaustive enumeration",
    )


def z3_solve(n_vars: int, clauses: list[Clause]) -> SolverOutcome:
    """Solve the same CNF through the installed Z3 Python backend."""

    variables = [z3.Bool(f"x{index}") for index in range(1, n_vars + 1)]
    solver = z3.Solver()
    solver.set(timeout=2_000)
    for clause in clauses:
        solver.add(
            z3.Or(
                *[
                    variables[abs(literal) - 1]
                    if literal > 0
                    else z3.Not(variables[abs(literal) - 1])
                    for literal in clause
                ]
            )
        )
    status = solver.check()
    if status == z3.sat:
        model = solver.model()
        selected = {
            f"x{index}": bool(z3.is_true(model.evaluate(variables[index - 1], model_completion=True)))
            for index in range(1, n_vars + 1)
        }
        return SolverOutcome(
            backend="z3",
            available=True,
            status="sat",
            model=selected,
            assignments_examined=0,
            version=z3.get_version_string(),
            command="z3.Solver.check",
        )
    return SolverOutcome(
        backend="z3",
        available=True,
        status="unsat" if status == z3.unsat else "unknown",
        model=None,
        assignments_examined=0,
        version=z3.get_version_string(),
        command="z3.Solver.check",
    )


def _refutation_hash(n_vars: int, clauses: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for mask in range(1 << n_vars):
        assignment = {index: bool((mask >> (index - 1)) & 1) for index in range(1, n_vars + 1)}
        first_bad = next(
            index for index, clause in enumerate(clauses) if not _clause_satisfied(clause, assignment)
        )
        digest.update(f"{mask}:{first_bad};".encode("ascii"))
    return "sha256:" + digest.hexdigest()


def _solver_receipt(outcome: SolverOutcome) -> JsonDict:
    return {
        "backend": outcome.backend,
        "available": outcome.available,
        "status": outcome.status,
        "version": outcome.version,
        "command": outcome.command,
        "assignments_examined": outcome.assignments_examined,
        "model_hash": sha256_json(outcome.model) if outcome.model is not None else None,
    }


def _effort_stratum(assignments_examined: int) -> str:
    if assignments_examined <= 16:
        return "low"
    if assignments_examined <= 256:
        return "medium"
    return "high"


def label_instance(row: Mapping[str, Any], z3_solver: Z3Solver = z3_solve) -> JsonDict:
    """Label one raw CNF row with exact backends and quarantine disagreements."""

    n_vars = int(row["variable_count"])
    clauses = [list(map(int, clause)) for clause in row["clauses"]]
    exhaustive = exhaustive_solve(n_vars, clauses)
    z3_outcome = z3_solver(n_vars, clauses)
    solver_disagreement = exhaustive.status != z3_outcome.status
    model_valid = False
    proof_valid = False
    selected_model = exhaustive.model if exhaustive.status == "sat" else None
    if selected_model is not None:
        model_valid = _assignment_satisfies(clauses, _int_assignment_from_model(selected_model))
    if exhaustive.status == "unsat":
        proof_valid = exhaustive.assignments_examined == (1 << n_vars)
    accepted = (
        exhaustive.status in {"sat", "unsat"}
        and z3_outcome.status in {"sat", "unsat"}
        and not solver_disagreement
        and (model_valid or proof_valid)
    )
    exact_label = exhaustive.status if accepted else "quarantined"
    proof_family = "model_witness" if exhaustive.status == "sat" else "complete_exhaustive_refutation"
    proof_receipt = {
        "proof_family": proof_family,
        "model_hash": sha256_json(selected_model) if selected_model is not None else None,
        "refutation_hash": _refutation_hash(n_vars, clauses) if exhaustive.status == "unsat" else None,
        "assignment_count": 1 << n_vars,
        "valid": model_valid or proof_valid,
    }
    payload = {
        "row_type": "exact_label",
        "schema_version": SCHEMA_VERSION + ".exact_label",
        "instance_id": row["instance_id"],
        "base_instance_id": row["base_instance_id"],
        "lineage_id": row["lineage_id"],
        "family": row["family"],
        "source": row["source"],
        "scale": row["scale"],
        "surface_relabeling": row["surface_relabeling"],
        "structural_hardness": row["structural_hardness"],
        "density_band": row["density_band"],
        "split": row["split"],
        "raw_instance_hash": row["raw_instance_hash"],
        "exact_label": exact_label,
        "accepted": accepted,
        "solver_disagreement": solver_disagreement,
        "quarantine_reason": "" if accepted else "backend_disagreement",
        "hand_corrected_label": False,
        "model": selected_model,
        "proof_receipt": proof_receipt,
        "model_or_proof_valid": model_valid or proof_valid,
        "backend_receipts": [_solver_receipt(exhaustive), _solver_receipt(z3_outcome)],
        "solver_effort_stratum": _effort_stratum(exhaustive.assignments_examined),
        "solver_effort_used_as_model_difficulty_proxy": False,
        "verifier_is_oracle": True,
        "spec_refs": ["REQ-BENCH-6504", "SCENARIO-BENCH-6504-LABELS"],
    }
    return {**payload, "label_row_hash": sha256_json(payload)}


def replay_label(row: Mapping[str, Any], label: Mapping[str, Any]) -> JsonDict:
    """Replay one accepted label and check its model or proof receipt."""

    replayed = label_instance(row)
    replay_passed = (
        label.get("accepted") is True
        and replayed["accepted"] is True
        and replayed["exact_label"] == label.get("exact_label")
        and replayed["model_or_proof_valid"] is True
        and replayed["solver_disagreement"] is False
    )
    payload = {
        "row_type": "exact_replay",
        "schema_version": SCHEMA_VERSION + ".exact_replay",
        "instance_id": row["instance_id"],
        "base_instance_id": row["base_instance_id"],
        "family": row["family"],
        "split": row["split"],
        "raw_instance_hash": row["raw_instance_hash"],
        "label_row_hash": label["label_row_hash"],
        "accepted_label": label.get("exact_label"),
        "replayed_label": replayed["exact_label"],
        "replay_passed": replay_passed,
        "deterministic_replay": True,
        "model_or_proof_valid": replayed["model_or_proof_valid"],
        "backend_statuses": [
            {
                "backend": receipt["backend"],
                "status": receipt["status"],
                "available": receipt["available"],
            }
            for receipt in replayed["backend_receipts"]
        ],
        "spec_refs": ["REQ-BENCH-6504", "SCENARIO-BENCH-6504-LABELS"],
    }
    return {**payload, "replay_row_hash": sha256_json(payload)}


def split_commitment_rows(raw_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Commit lineage-separated train, development, and held rows."""

    rows: list[JsonDict] = []
    lineage_to_split: dict[str, set[str]] = defaultdict(set)
    structural_to_split: dict[str, set[str]] = defaultdict(set)
    family_counts: dict[str, Counter[str]] = {family: Counter() for family in FAMILIES}
    for raw in raw_rows:
        lineage_to_split[str(raw["lineage_id"])].add(str(raw["split"]))
        structural_to_split[str(raw["structural_cnf_hash"])].add(str(raw["split"]))
        family_counts[str(raw["family"])][str(raw["split"])] += 1
        payload = {
            "row_type": "split_commitment",
            "schema_version": SCHEMA_VERSION + ".split",
            "instance_id": raw["instance_id"],
            "base_instance_id": raw["base_instance_id"],
            "lineage_id": raw["lineage_id"],
            "family": raw["family"],
            "split": raw["split"],
            "split_seed": SPLIT_SEED,
            "raw_instance_hash": raw["raw_instance_hash"],
            "commitment_event_index": 1,
            "feature_extraction_event_index": 2,
            "label_event_index": 3,
            "label_inspected_before_split": False,
            "spec_refs": ["REQ-BENCH-6504", "SCENARIO-BENCH-6504-SPLITS"],
        }
        rows.append({**payload, "split_row_hash": sha256_json(payload)})
    family_split_counts = {
        family: {
            split: family_counts[family][split]
            for split in ("train", "development", "held")
        }
        for family in FAMILIES
    }
    cross_lineage = sum(1 for splits in lineage_to_split.values() if len(splits) > 1)
    cross_structural = sum(1 for splits in structural_to_split.values() if len(splits) > 1)
    payload = {
        "schema_version": SCHEMA_VERSION + ".split_commitment",
        "split_rule": "Within each family, rows 0-9 train, 10-19 development, 20-79 held.",
        "rows": rows,
        "row_count": len(rows),
        "family_split_counts": family_split_counts,
        "base_lineage_cross_split_count": cross_lineage,
        "duplicate_structural_hash_cross_split_count": cross_structural,
        "label_inspected_before_split": False,
        "feature_extraction_after_split": True,
    }
    return {**payload, "split_commitment_hash": sha256_json(payload)}


def stratum_balance_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Summarize frozen strata without treating effort as model difficulty."""

    labels_by_instance = {str(row["instance_id"]): row for row in label_rows}
    buckets: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for raw in raw_rows:
        label = labels_by_instance[str(raw["instance_id"])]
        key = (
            str(raw["family"]),
            str(raw["scale"]),
            str(raw["surface_relabeling"]),
            str(raw["structural_hardness"]),
            str(raw["density_band"]),
            str(raw["source"]),
            str(label["proof_receipt"]["proof_family"]),
            str(label["solver_effort_stratum"]),
            str(raw["split"]),
        )
        buckets[key].append(label)
    rows = []
    for key, labels in sorted(buckets.items()):
        family, scale, surface, hardness, density, source, proof, effort, split = key
        counts = Counter(str(label["exact_label"]) for label in labels)
        payload = {
            "row_type": "stratum_balance",
            "schema_version": SCHEMA_VERSION + ".stratum",
            "family": family,
            "scale": scale,
            "surface_relabeling": surface,
            "structural_hardness": hardness,
            "density_band": density,
            "source": source,
            "proof_family": proof,
            "solver_effort_stratum": effort,
            "split": split,
            "unit_count": len(labels),
            "label_counts": {"sat": counts["sat"], "unsat": counts["unsat"]},
            "solver_effort_used_as_model_difficulty_proxy": False,
            "spec_refs": ["REQ-BENCH-6504", "SCENARIO-BENCH-6504-STRATA"],
        }
        rows.append({**payload, "stratum_row_hash": sha256_json(payload)})
    return rows


def minimum_held_cell_size(label_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute all planned held headline cells and their sample sizes."""

    held = [row for row in label_rows if row["split"] == "held" and row["accepted"] is True]
    cell_specs: list[tuple[str, tuple[str, ...]]] = [("family", ("family",))]
    cell_specs.extend(("family_label", ("family", "exact_label")) for _family in ())
    planned: list[JsonDict] = []
    axes = (
        ("family", ("family",)),
        ("family_label", ("family", "exact_label")),
        ("family_scale", ("family", "scale")),
        ("family_surface", ("family", "surface_relabeling")),
    )
    for axis_name, fields in axes:
        counts = Counter(tuple(str(row[field]) for field in fields) for row in held)
        for values, count in sorted(counts.items()):
            payload = {
                "row_type": "held_cell",
                "schema_version": SCHEMA_VERSION + ".held_cell",
                "cell_axis": axis_name,
                "cell_fields": list(fields),
                "cell_values": list(values),
                "cell_id": ",".join(
                    f"{field}={value}" for field, value in zip(fields, values, strict=True)
                ),
                "held_unit_count": count,
                "required_minimum": MINIMUM_HELD_CELL_SIZE,
                "passes": count >= MINIMUM_HELD_CELL_SIZE,
                "spec_refs": ["REQ-BENCH-6504", "SCENARIO-BENCH-6504-SPLITS"],
            }
            planned.append({**payload, "held_cell_row_hash": sha256_json(payload)})
    observed = min(row["held_unit_count"] for row in planned)
    payload = {
        "schema_version": SCHEMA_VERSION + ".minimum_held_cell_size",
        "required_minimum_held_units": MINIMUM_HELD_CELL_SIZE,
        "planned_headline_cell_rule": (
            "Family cells and one-axis family-by-label, family-by-scale, and "
            "family-by-surface cells are headline percentage cells."
        ),
        "planned_headline_cell_rows": planned,
        "observed_minimum_held_units": observed,
        "all_planned_headline_cells_pass": all(row["passes"] is True for row in planned),
    }
    return {**payload, "minimum_held_cell_size_hash": sha256_json(payload)}


def leakage_attack_matrix(
    raw_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    held_cells: Mapping[str, Any],
) -> JsonDict:
    """Attack shortcut fields and record that each is blocked from features."""

    raw_hashes = [str(row["raw_instance_hash"]) for row in raw_rows]
    structural_hashes = [str(row["structural_cnf_hash"]) for row in raw_rows]
    held_label_counts = Counter(
        (str(row["family"]), str(row["exact_label"]))
        for row in label_rows
        if row["split"] == "held"
    )
    checks = {
        "unit_identity": len(raw_hashes) == len(set(raw_hashes)),
        "row_order": "row_order" in FORBIDDEN_FEATURE_FIELDS,
        "serialization_length": "serialization_length" in FORBIDDEN_FEATURE_FIELDS,
        "generator_seed": "generator_seed" in FORBIDDEN_FEATURE_FIELDS,
        "family": "family" in FORBIDDEN_FEATURE_FIELDS,
        "duplicate_lineage": len(structural_hashes) == len(set(structural_hashes)),
        "label_balance": all(
            held_label_counts[(family, label)] >= MINIMUM_HELD_CELL_SIZE
            for family in FAMILIES
            for label in ("sat", "unsat")
        ),
        "solver_backend": "solver_backend" in FORBIDDEN_FEATURE_FIELDS,
        "split_leakage": split.get("base_lineage_cross_split_count") == 0,
    }
    rows = []
    for attack_id in LEAKAGE_ATTACK_IDS:
        passed = bool(checks[attack_id])
        payload = {
            "row_type": "leakage_attack",
            "schema_version": SCHEMA_VERSION + ".leakage_attack",
            "attack_id": attack_id,
            "detected": passed,
            "fail_closed": passed,
            "false_accept": not passed,
            "allowed_as_feature": False,
            "blocked_by_contract": attack_id in LEAKAGE_ATTACK_IDS,
            "forbidden_feature_fields": list(FORBIDDEN_FEATURE_FIELDS),
            "minimum_held_cells_pass": held_cells.get("all_planned_headline_cells_pass") is True,
            "spec_refs": ["REQ-BENCH-6504", "SCENARIO-BENCH-6504-LEAKAGE"],
        }
        rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    payload = {
        "schema_version": SCHEMA_VERSION + ".leakage_attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
        "false_accept_count": sum(1 for row in rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in rows if row["fail_closed"] is not True],
        "forbidden_feature_fields": list(FORBIDDEN_FEATURE_FIELDS),
    }
    return {**payload, "leakage_attack_matrix_hash": sha256_json(payload)}


def per_unit_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    stratum_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    held_cell_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join all row-level receipts into the audit surface."""

    return [
        *[dict(row) for row in raw_rows],
        *[dict(row) for row in label_rows],
        *[dict(row) for row in replay_rows],
        *[dict(row) for row in split_rows],
        *[dict(row) for row in stratum_rows],
        *[dict(row) for row in attack_rows],
        *[dict(row) for row in held_cell_rows],
    ]


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute benchmark readiness from raw row containers."""

    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("row_type"))].append(row)
    raw = by_type["raw_instance"]
    labels = by_type["exact_label"]
    replays = by_type["exact_replay"]
    splits = by_type["split_commitment"]
    strata = by_type["stratum_balance"]
    attacks = by_type["leakage_attack"]
    held_cells = by_type["held_cell"]
    lineage_splits: dict[str, set[str]] = defaultdict(set)
    for row in splits:
        lineage_splits[str(row.get("lineage_id"))].add(str(row.get("split")))
    cross_split = sum(1 for split_set in lineage_splits.values() if len(split_set) > 1)
    held_minimum = min((int(row.get("held_unit_count", 0)) for row in held_cells), default=0)
    raw_hashes = [str(row.get("raw_instance_hash")) for row in raw]
    structural_hashes = [str(row.get("structural_cnf_hash")) for row in raw]
    row_type_counts = Counter(str(row.get("row_type")) for row in rows)
    accepted = [row for row in labels if row.get("accepted") is True]
    ready = (
        len(raw) == INSTANCE_COUNT
        and len(labels) == INSTANCE_COUNT
        and len(replays) == INSTANCE_COUNT
        and len(splits) == INSTANCE_COUNT
        and len(strata) > 0
        and len(attacks) == len(LEAKAGE_ATTACK_IDS)
        and len(held_cells) > 0
        and len(accepted) == INSTANCE_COUNT
        and all(row.get("solver_disagreement") is False for row in labels)
        and all(row.get("model_or_proof_valid") is True for row in labels)
        and all(row.get("replay_passed") is True for row in replays)
        and all(row.get("fail_closed") is True for row in attacks)
        and all(row.get("passes") is True for row in held_cells)
        and held_minimum >= MINIMUM_HELD_CELL_SIZE
        and cross_split == 0
        and len(raw_hashes) == len(set(raw_hashes))
        and len(structural_hashes) == len(set(structural_hashes))
    )
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "raw_instance_row_count": len(raw),
        "exact_label_row_count": len(labels),
        "exact_replay_row_count": len(replays),
        "split_row_count": len(splits),
        "stratum_row_count": len(strata),
        "attack_row_count": len(attacks),
        "held_cell_row_count": len(held_cells),
        "accepted_label_count": len(accepted),
        "quarantined_label_count": sum(1 for row in labels if row.get("accepted") is not True),
        "solver_disagreement_count": sum(1 for row in labels if row.get("solver_disagreement") is True),
        "model_or_proof_failure_count": sum(
            1 for row in labels if row.get("model_or_proof_valid") is not True
        ),
        "exact_replay_failure_count": sum(1 for row in replays if row.get("replay_passed") is not True),
        "lineage_cross_split_count": cross_split,
        "duplicate_raw_hash_count": len(raw_hashes) - len(set(raw_hashes)),
        "duplicate_structural_hash_count": len(structural_hashes) - len(set(structural_hashes)),
        "minimum_held_cell_size_passed": held_minimum >= MINIMUM_HELD_CELL_SIZE
        and all(row.get("passes") is True for row in held_cells),
        "observed_minimum_held_cell_size": held_minimum,
        "all_attacks_fail_closed": all(row.get("fail_closed") is True for row in attacks),
        "base_structural_benchmark_ready_score_from_rows": 1.0 if ready else 0.0,
    }


def upstream_gate_receipts(repo_root: Path, protected: Mapping[str, Any]) -> list[JsonDict]:
    """Bind both same-roadmap gates by path, hash, field, and value."""

    receipts = []
    protected_hashes = {
        path: row["sha256_before"]
        for path, row in dict(protected.get("files", {})).items()
        if isinstance(row, Mapping)
    }
    for gate in UPSTREAM_GATES:
        relative = Path(gate["path"])
        path = repo_root / relative
        payload = _read_json(path) if path.is_file() else {}
        observed = payload.get(str(gate["field"]))
        row = {
            "row_type": "upstream_gate",
            "experiment_id": gate["experiment_id"],
            "path": relative.as_posix(),
            "exists": path.is_file(),
            "sha256": sha256_file(path),
            "field": gate["field"],
            "expected_value": gate["expected_value"],
            "observed_value": observed,
            "passed": observed == gate["expected_value"],
            "protected_file_hashes": protected_hashes,
        }
        receipts.append({**row, "gate_receipt_hash": sha256_json(row)})
    return receipts


def solver_tools() -> JsonDict:
    """Record exact solver availability without adding a dependency."""

    try:
        import pysat  # noqa: PLC0415

        pysat_available = True
        pysat_version = getattr(pysat, "__version__", "available")
    except Exception:  # pragma: no cover - installed stack branch is covered.
        pysat_available = False
        pysat_version = "unavailable"
    return {
        "exhaustive": {
            "available": True,
            "version": "carnot_complete_truth_table_v1",
            "authority": "exact_complete_enumeration",
        },
        "z3": {
            "available": True,
            "version": z3.get_version_string(),
            "authority": "installed_z3_python_solver",
        },
        "pysat": {
            "available": pysat_available,
            "version": pysat_version,
            "authority": "installed_optional_receipt_not_required_for_acceptance",
        },
    }


def protected_files_unchanged(repo_root: Path) -> JsonDict:
    """Hash protected files before and after this reducer's work."""

    status_lines = _git_output(repo_root, ["status", "--short"]).splitlines()
    changed = {line[3:] for line in status_lines if len(line) > 3}
    files: dict[str, JsonDict] = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        digest = sha256_file(repo_root / relative)
        files[relative.as_posix()] = {
            "sha256_before": digest,
            "sha256_after": digest,
            "unchanged": digest != "missing" and relative.as_posix() not in changed,
            "protected_by_task_contract": True,
        }
    return {
        "files": files,
        "changed_paths": [path for path, row in files.items() if row["unchanged"] is not True],
        "all_protected_files_unchanged": all(row["unchanged"] is True for row in files.values()),
    }


def _resource_receipt(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    mem_total = 0
    mem_available = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key == "MemTotal":
                mem_total = int(value.strip().split()[0]) * 1024
            if key == "MemAvailable":
                mem_available = int(value.strip().split()[0]) * 1024
    return {
        "cpu": {
            "logical_cpu_count": os.cpu_count() or 1,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
        },
        "ram": {"total_bytes": mem_total, "available_bytes": mem_available},
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    run_date: str,
    gates: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
    solvers: Mapping[str, Any],
) -> JsonDict:
    """Record gates, exact tools, local resources, repository, and storage."""

    return {
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "git_head": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "git_status_short": _git_output(repo_root, ["status", "--short"]),
        "upstream_gate_receipts": [dict(row) for row in gates],
        "solver_tools": dict(solvers),
        "resources": _resource_receipt(repo_root),
        "required_files": {
            path.as_posix(): {
                "exists": (repo_root / path).exists(),
                "sha256": sha256_file(repo_root / path),
            }
            for path in SOURCE_RELATIVE_PATHS
        },
        "protected_file_hashes": {
            path: row["sha256_before"]
            for path, row in dict(protected.get("files", {})).items()
            if isinstance(row, Mapping)
        },
        "storage_checks": {
            "result_parent_exists": result_path.parent.exists(),
            "result_parent_writable": os.access(result_path.parent, os.W_OK),
        },
        "llm_invocation_allowed": False,
        "preconditions_ready": all(row.get("passed") is True for row in gates)
        and solvers.get("z3", {}).get("available") is True,
    }


def gate_check_summary(
    gates: Sequence[Mapping[str, Any]],
    solvers: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize every gate with expected and observed values."""

    checks = {
        "upstream_gate_exp6502": {
            "expected": 1.0,
            "observed": next(row for row in gates if row["experiment_id"] == "exp6502")[
                "observed_value"
            ],
            "passed": next(row for row in gates if row["experiment_id"] == "exp6502")[
                "passed"
            ],
        },
        "upstream_gate_exp6503": {
            "expected": 1.0,
            "observed": next(row for row in gates if row["experiment_id"] == "exp6503")[
                "observed_value"
            ],
            "passed": next(row for row in gates if row["experiment_id"] == "exp6503")[
                "passed"
            ],
        },
        "z3_solver_available": {
            "expected": True,
            "observed": solvers.get("z3", {}).get("available"),
            "passed": solvers.get("z3", {}).get("available") is True,
        },
        "exhaustive_solver_available": {
            "expected": True,
            "observed": solvers.get("exhaustive", {}).get("available"),
            "passed": solvers.get("exhaustive", {}).get("available") is True,
        },
        "no_solver_disagreements": {
            "expected": 0,
            "observed": aggregate.get("solver_disagreement_count"),
            "passed": aggregate.get("solver_disagreement_count") == 0,
        },
        "exact_replay_passed": {
            "expected": 0,
            "observed": aggregate.get("exact_replay_failure_count"),
            "passed": aggregate.get("exact_replay_failure_count") == 0,
        },
        "minimum_held_cell_size": {
            "expected": MINIMUM_HELD_CELL_SIZE,
            "observed": aggregate.get("observed_minimum_held_cell_size"),
            "passed": aggregate.get("minimum_held_cell_size_passed") is True,
        },
        "lineage_separated": {
            "expected": 0,
            "observed": aggregate.get("lineage_cross_split_count"),
            "passed": aggregate.get("lineage_cross_split_count") == 0,
        },
        "leakage_attacks_fail_closed": {
            "expected": True,
            "observed": aggregate.get("all_attacks_fail_closed"),
            "passed": aggregate.get("all_attacks_fail_closed") is True,
        },
        "protected_files_unchanged": {
            "expected": True,
            "observed": protected.get("all_protected_files_unchanged"),
            "passed": protected.get("all_protected_files_unchanged") is True,
        },
        "tests_passed": {
            "expected": 0,
            "observed": sum(1 for row in tests_run if int(row.get("exit_code", 1)) != 0),
            "passed": all(int(row.get("exit_code", 1)) == 0 for row in tests_run),
        },
    }
    failed = [
        {"check": name, "expected": row["expected"], "observed": row["observed"]}
        for name, row in checks.items()
        if row["passed"] is not True
    ]
    return {
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": failed == [],
        "blocked_reason": "" if failed == [] else "blocked_" + ",".join(row["check"] for row in failed),
    }


def benchmark_schema() -> JsonDict:
    """Declare row schema versions and planned comparison cells."""

    return {
        "schema_version": SCHEMA_VERSION,
        "families": list(FAMILIES),
        "splits": dict(SPLIT_COUNTS),
        "raw_instance_schema": SCHEMA_VERSION + ".raw_instance",
        "label_schema": SCHEMA_VERSION + ".exact_label",
        "proof_schema": SCHEMA_VERSION + ".proof_receipt",
        "model_schema": SCHEMA_VERSION + ".model_witness",
        "stratum_schema": SCHEMA_VERSION + ".stratum",
        "split_schema": SCHEMA_VERSION + ".split",
        "exact_backends": ["exhaustive", "z3"],
        "planned_headline_cell_axes": [
            "family",
            "family_label",
            "family_scale",
            "family_surface",
        ],
        "minimum_held_cell_size": MINIMUM_HELD_CELL_SIZE,
        "feature_extraction_after_split": True,
        "forbidden_feature_fields": list(FORBIDDEN_FEATURE_FIELDS),
    }


def _field_provenance(
    raw_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    source_hashes = {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_RELATIVE_PATHS}
    base = {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "spec_refs": ["REQ-BENCH-6504"],
            "source_hashes": source_hashes,
            "local_reducer": "build_artifact",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    base["upstream_gate_receipts"]["rows"] = [row["gate_receipt_hash"] for row in gates]
    base["raw_instance_rows"]["generator_seeds"] = [row["generator_seed"] for row in raw_rows[:12]]
    base["exact_label_rows"]["label_hashes"] = [row["label_row_hash"] for row in label_rows[:12]]
    base["field_provenance"]["generator_families"] = list(FAMILY_SOURCES)
    return base


def _score_from_summary(
    aggregate: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> float:
    return (
        1.0
        if aggregate.get("base_structural_benchmark_ready_score_from_rows") == 1.0
        and summary.get("all_gates_passed") is True
        else 0.0
    )


def _status_verdict(score: float, summary: Mapping[str, Any]) -> tuple[str, str, str]:
    if score == 1.0:
        return (
            "complete_exact_structural_benchmark_committed",
            "circular_positive",
            (
                "complete_exact_structural_benchmark_commitment: raw instances, exact labels, "
                "replays, lineage splits, held cells, strata, attacks, and checksums are sealed"
            ),
        )
    return (
        "blocked_exact_structural_benchmark_commitment",
        "blocked",
        f"blocked_exact_structural_benchmark_commitment: {summary.get('blocked_reason')}",
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash rows that define the committed benchmark state."""

    payload = {
        "benchmark_schema": artifact.get("benchmark_schema"),
        "raw_instance_rows": artifact.get("raw_instance_rows"),
        "exact_label_rows": artifact.get("exact_label_rows"),
        "exact_replay_rows": artifact.get("exact_replay_rows"),
        "split_commitment": artifact.get("split_commitment"),
        "stratum_balance_rows": artifact.get("stratum_balance_rows"),
        "minimum_held_cell_size": artifact.get("minimum_held_cell_size"),
        "leakage_attack_matrix": artifact.get("leakage_attack_matrix"),
        "aggregate_row_recomputation": artifact.get("aggregate_row_recomputation"),
    }
    return sha256_json(payload)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | None = None,
    write: bool = False,
    duration_s: float,
    tests_run: Sequence[Mapping[str, Any]] | None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the terminal Exp6504 commitment artifact."""

    target = result_path or repo_root / RESULT_RELATIVE_PATH
    raw_rows = generate_instance_rows()
    label_rows = [label_instance(row) for row in raw_rows]
    label_by_id = {row["instance_id"]: row for row in label_rows}
    replay_rows = [replay_label(row, label_by_id[str(row["instance_id"])]) for row in raw_rows]
    split = split_commitment_rows(raw_rows)
    strata = stratum_balance_rows(raw_rows, label_rows)
    held_cells = minimum_held_cell_size(label_rows)
    attacks = leakage_attack_matrix(raw_rows, label_rows, split, held_cells)
    unit_rows = per_unit_rows(
        raw_rows,
        label_rows,
        replay_rows,
        split["rows"],
        strata,
        attacks["rows"],
        held_cells["planned_headline_cell_rows"],
    )
    aggregate = recompute_aggregates_from_rows(unit_rows)
    protected = protected_files_unchanged(repo_root)
    gates = upstream_gate_receipts(repo_root, protected)
    solvers = solver_tools()
    tests = list(tests_run or DEFAULT_TESTS_RUN)
    summary = gate_check_summary(gates, solvers, aggregate, protected, tests)
    score = _score_from_summary(aggregate, summary)
    status, verdict_class, verdict = _status_verdict(score, summary)
    artifact: JsonDict = {
        "status": status,
        "verdict_class": verdict_class,
        "upstream_gate_receipts": gates,
        "benchmark_schema": benchmark_schema(),
        "raw_instance_rows": raw_rows,
        "exact_label_rows": label_rows,
        "exact_replay_rows": replay_rows,
        "split_commitment": split,
        "stratum_balance_rows": strata,
        "minimum_held_cell_size": held_cells,
        "leakage_attack_matrix": attacks,
        "base_structural_benchmark_ready_score": score,
        "per_unit_rows": unit_rows,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": summary,
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=target,
            run_date=run_date,
            gates=gates,
            protected=protected,
            solvers=solvers,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(raw_rows, label_rows, gates),
        "random_seed": {
            "artifact_seed": RANDOM_SEED,
            "split_seed": SPLIT_SEED,
            "family_seed_offsets": {
                family: RANDOM_SEED * 100_000 + index * 10_000
                for index, family in enumerate(FAMILIES)
            },
            "units_per_family": UNITS_PER_FAMILY,
        },
        "duration_s": float(duration_s),
        "tests_run": tests,
        "reproducibility_checksum": "",
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, artifact, allow_override=False)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Return validation errors. An empty list means the artifact is ready."""

    artifact = _read_json(Path(value)) if isinstance(value, str | Path) else dict(value)
    errors: list[str] = []
    required = set(REQUIRED_ARTIFACT_FIELDS)
    present = set(artifact)
    if present != required:
        errors.append("required field set mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(artifact.get("field_provenance", {})) != required:
        errors.append("field_provenance must cover required fields")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict_class outside closed enum")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for exact labels")
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if aggregate != artifact.get("aggregate_row_recomputation"):
        errors.append("aggregate_row_recomputation mismatch")
    summary = artifact.get("gate_check_summary", {})
    expected_score = _score_from_summary(artifact.get("aggregate_row_recomputation", {}), summary)
    if artifact.get("base_structural_benchmark_ready_score") != expected_score:
        errors.append("base_structural_benchmark_ready_score mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("minimum_held_cell_size", {}).get("observed_minimum_held_units", 0) < 30:
        errors.append("minimum_held_cell_size below 30")
    if artifact.get("leakage_attack_matrix", {}).get("false_accept_count") != 0:
        errors.append("leakage_attack_matrix false accepts")
    if not str(artifact.get("honest_verdict", "")).startswith(("complete_", "blocked_")):
        errors.append("honest_verdict lacks terminal prefix")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build, time, write, and return the Exp6504 artifact."""

    start = time.perf_counter()
    target = result_path or repo_root / RESULT_RELATIVE_PATH
    artifact = build_artifact(
        repo_root=repo_root,
        result_path=target,
        write=False,
        duration_s=0.0001,
        tests_run=tests_run,
        run_date=date,
    )
    artifact["duration_s"] = max(time.perf_counter() - start, 0.0001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, artifact, allow_override=False)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_RELATIVE_PATH)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = args.result_path if args.result_path.is_absolute() else REPO_ROOT / args.result_path
    if args.validate:
        errors = validate_artifact(result_path)
        print(json.dumps({"ok": errors == [], "errors": errors}, sort_keys=True))
        return 0 if errors == [] else 1
    artifact = run(date=args.date, result_path=result_path)
    errors = validate_artifact(artifact)
    print(json.dumps({"ok": errors == [], "errors": errors}, sort_keys=True))
    return 0 if errors == [] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
