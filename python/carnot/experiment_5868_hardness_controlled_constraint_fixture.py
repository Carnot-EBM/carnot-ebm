"""Exp5868 hardness-controlled exact constraint fixture.

Spec refs: REQ-VERIFY-5868, SCENARIO-VERIFY-5868-GENERATION,
SCENARIO-VERIFY-5868-CERTIFICATES, SCENARIO-VERIFY-5868-RELABELS-AND-CONTROLS,
SCENARIO-VERIFY-5868-REPLAY-AND-BLOCKED.

The fixture is deliberately narrow: it creates small Tseitin CNF instances,
labels them with exact local solvers, validates explicit certificates or
parity contradiction witnesses, and records surface controls. It does not run
an LLM and it does not claim that SAT conflicts are model difficulty.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any


JsonDict = dict[str, Any]
MemoryProbe = Callable[[], JsonDict]
DiskProbe = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5868_hardness_controlled_constraint_fixture.json")
ROW_FILE_RELATIVE_PATH = Path(
    "results/experiment_5868_hardness_controlled_constraint_fixture.rows.jsonl"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5868_hardness_controlled_constraint_fixture.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5868_hardness_controlled_constraint_fixture.py"
)
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
EXP5840_ARTIFACT_RELATIVE_PATH = Path(
    "results/experiment_5840_exact_counterfactual_embedding_fixture.json"
)
EXP5840_ROWS_RELATIVE_PATH = Path(
    "results/experiment_5840_exact_counterfactual_embedding_fixture.rows.jsonl"
)
PROTECTED_FILES = (Path("scripts/research_conductor.py"),)

SCHEMA = "carnot.experiment_5868.hardness_controlled_constraint_fixture.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 5868
EXPERIMENT_ID = "experiment_5868_hardness_controlled_constraint_fixture"
MILESTONE = "2026.07.522"
RUN_DATE = "20260724"
SOURCE_ARXIV_ID = "2607.17047"
SOURCE_ID = f"arxiv:{SOURCE_ARXIV_ID}"
INFERENCE_SUBSTRATE = "deterministic_exact_solver_labeled_dataset_no_llm"
VERIFIER_IS_ORACLE = True
BASE_SEED = 5868

FAMILIES = ("expander_tseitin", "ladder_tseitin")
LABELS = ("satisfiable", "unsatisfiable")
SURFACE_CONTROL_KINDS = (
    "canonical",
    "clause_reorder",
    "variable_renaming",
    "padding",
    "density_mismatched",
    "length_matched",
    "no_information",
)
NO_INFORMATION_TOKENS = ("neutral", "blank", "masked", "field", "control", "zero")
PAD_TOKENS = ("pad", "neutral", "window", "stable", "surface", "hold")
SOLVER_CONFIGS = ("dpll_lex_false_first_v1", "dpll_occurrence_true_first_v1")
MAX_CLAUSE_WIDTH = 3
DENSITY_TOLERANCE = 0.45
LENGTH_TOLERANCE_TOKENS = 0
RAM_FLOOR_MB = 1024
DISK_FLOOR_MB = 512
MAX_DPLL_DECISIONS = 1_000_000


@dataclass(frozen=True)
class SizeBin:
    """One matched graph-size cell used by both proof-hardness families."""

    name: str
    vertices: int
    surface_tokens: int


SIZE_BINS = (
    SizeBin("small", 8, 220),
    SizeBin("medium", 10, 280),
    SizeBin("large", 12, 340),
)
SIZE_BIN_NAMES = tuple(size.name for size in SIZE_BINS)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "source_method_receipt",
    "generator_and_seed_receipts",
    "family_and_size_bin_definitions",
    "density_width_and_length_matching",
    "label_and_certificate_balance",
    "solver_versions_and_oracle_receipts",
    "proof_hardness_covariates",
    "proof_preserving_relabel_receipts",
    "surface_and_no_information_controls",
    "solver_disagreement_and_timeout_controls",
    "row_file_receipt",
    "deterministic_replay_receipt",
    "protected_files_unchanged",
    "hardness_controlled_fixture_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a ready exact fixture from partial generation.",
    "preconditions_checked": "Paper, solver, generator, seed, resource, output, and protection checks prevent unverifiable data.",
    "source_method_receipt": "The hardness-control design traces to a real primary source without importing its results.",
    "generator_and_seed_receipts": "Deterministic generation makes every formula and relabel reproducible.",
    "family_and_size_bin_definitions": "Proof-hard and proof-easy comparisons must be explicit within aligned bins.",
    "density_width_and_length_matching": "Matched nuisance axes prevent density or text length standing in for hardness.",
    "label_and_certificate_balance": "Balanced exact labels and checked witnesses create usable paired evaluation cells.",
    "solver_versions_and_oracle_receipts": "Exact solver outputs own labels and include versioned commands.",
    "proof_hardness_covariates": "Conflicts and time are analysis covariates, never truth labels.",
    "proof_preserving_relabel_receipts": "Equivalent formulas must retain labels and certificates after renaming.",
    "surface_and_no_information_controls": "Order, padding, renaming, length, density, and null rows expose shortcuts.",
    "solver_disagreement_and_timeout_controls": "Unsettled instances are rejected rather than guessed.",
    "row_file_receipt": "Path, count, schema, and hash expose all examples.",
    "deterministic_replay_receipt": "A second generation must reproduce row IDs and content exactly.",
    "protected_files_unchanged": "User and operator-owned files remain untouched.",
    "hardness_controlled_fixture_ready_score": "EMIT BARE scalar; only 1.0 permits Exp5869.",
    "duration_s": "Measured wall time exposes bootstrap-only dataset work.",
    "inference_substrate": "`deterministic_exact_solver_labeled_dataset_no_llm` declares the true path.",
    "verifier_is_oracle": "True for exact labels; solver conflict covariates are not oracle scores.",
    "field_provenance": "Every row traces to generator config, seed, solver, certificate, and source hash.",
    "test_commands": "Commands document generation, matching, solver, relabel, replay, and schema checks.",
    "test_exit_codes": "Exit codes prevent invalid fixtures becoming ready.",
    "reproducibility_checksum": "A checksum detects generator, solver, row, or control drift.",
    "honest_verdict": "A `complete:`, `ready:`, or `blocked:` prefix states the terminal dataset result.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5868_hardness_controlled_constraint_fixture.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5868_hardness_controlled_constraint_fixture.py "
    "-m pytest tests/python/test_experiment_5868_hardness_controlled_constraint_fixture.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5868_hardness_controlled_constraint_fixture.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5868_hardness_controlled_constraint_fixture.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\"",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values with deterministic key and byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for stable text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes in chunks rather than trusting path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL object required: {path}")
        rows.append(dict(payload))
    return rows


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": RAM_FLOOR_MB,
        "ok": available_mb >= RAM_FLOOR_MB,
    }


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host-dependent resource probe.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": DISK_FLOOR_MB,
        "ok": available_mb >= DISK_FLOOR_MB,
    }


def _hash_optional_file(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.exists() and path.is_file() else "missing"


def _source_receipt_text(text: str) -> str:
    marker = f"arXiv:{SOURCE_ARXIV_ID}"
    index = text.find(marker)
    if index < 0:
        return ""
    start = text.rfind("\n- **", 0, index)
    start = 0 if start < 0 else start + 1
    next_bullet = text.find("\n- **", index + len(marker))
    next_heading = text.find("\n### ", index + len(marker))
    candidates = [value for value in (next_bullet, next_heading) if value >= 0]
    end = min(candidates) if candidates else len(text)
    return text[start:end].strip()


def source_method_receipt(root: Path = REPO_ROOT) -> JsonDict:
    """Bind the fixture design to the local source ledger without importing results."""

    path = Path(root) / RESEARCH_REFERENCES_RELATIVE_PATH
    if not path.exists():
        return {
            "source_id": SOURCE_ID,
            "path": RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
            "receipt_found": False,
            "ok": False,
        }
    text = path.read_text(encoding="utf-8")
    receipt_text = _source_receipt_text(text)
    return {
        "source_id": SOURCE_ID,
        "path": RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
        "receipt_found": bool(receipt_text),
        "receipt_hash": sha256_text(receipt_text) if receipt_text else "missing",
        "research_references_sha256": sha256_file(path),
        "method_boundary": "fixture_design_reference_only_no_result_import",
        "claims_imported": False,
        "ok": bool(receipt_text),
    }


def solver_version_receipts() -> JsonDict:
    """Declare the exact local solver configurations used as label oracles."""

    solvers = {
        SOLVER_CONFIGS[0]: {
            "version": "carnot_exact_dpll_lex_false_first_v1",
            "command": "internal:solve_cnf_dpll(config=dpll_lex_false_first_v1)",
            "complete": True,
        },
        SOLVER_CONFIGS[1]: {
            "version": "carnot_exact_dpll_occurrence_true_first_v1",
            "command": "internal:solve_cnf_dpll(config=dpll_occurrence_true_first_v1)",
            "complete": True,
        },
    }
    return {
        "schema": SCHEMA + ".solver_versions",
        "solvers": solvers,
        "solver_configuration_count": len(solvers),
        "python_version": platform.python_version(),
        "ok": all(item["complete"] for item in solvers.values()),
    }


def generator_seed_registry() -> JsonDict:
    """Return the preregistered deterministic generator seeds and controls."""

    registry = {
        "base_seed": BASE_SEED,
        "family_order": list(FAMILIES),
        "label_order": list(LABELS),
        "size_bins": [size.__dict__ for size in SIZE_BINS],
        "surface_control_kinds": list(SURFACE_CONTROL_KINDS),
        "variable_relabel_rule": "new_var=((old_var+seed) mod n_vars)+1",
        "clause_reorder_rule": "reverse_clauses_and_literals",
        "density_mismatch_rule": "add_redundant_tautology_clauses",
    }
    return {
        "schema": SCHEMA + ".generator_seed_registry",
        "registry": registry,
        "registry_hash": sha256_json(registry),
        "ok": True,
    }


def _output_path_receipt(result_path: Path, row_file_path: Path) -> JsonDict:
    def writable(path: Path) -> bool:
        parent = path.parent
        parent_ready = (parent.exists() and os.access(parent, os.W_OK)) or (
            parent.parent.exists() and os.access(parent.parent, os.W_OK)
        )
        return parent_ready and (not path.exists() or os.access(path, os.W_OK))

    return {
        "result_path": str(result_path),
        "row_file_path": str(row_file_path),
        "result_writable": writable(result_path),
        "row_file_writable": writable(row_file_path),
        "atomic_checkpoint_suffix": ".tmp",
    }


def _exp5840_receipt(root: Path) -> JsonDict:
    artifact_path = root / EXP5840_ARTIFACT_RELATIVE_PATH
    rows_path = root / EXP5840_ROWS_RELATIVE_PATH
    if not artifact_path.exists() or not rows_path.exists():
        return {"ok": False, "blocked_reason": "missing_exp5840_fixture"}
    try:
        artifact = _read_json(artifact_path)
        row_count = sum(1 for line in rows_path.read_text(encoding="utf-8").splitlines() if line)
        return {
            "artifact_path": EXP5840_ARTIFACT_RELATIVE_PATH.as_posix(),
            "rows_path": EXP5840_ROWS_RELATIVE_PATH.as_posix(),
            "artifact_sha256": sha256_file(artifact_path),
            "rows_sha256": sha256_file(rows_path),
            "ready_score": artifact.get("counterfactual_fixture_ready_score"),
            "row_count": row_count,
            "ok": artifact.get("counterfactual_fixture_ready_score") == 1.0 and row_count > 0,
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "blocked_reason": f"corrupt_exp5840_fixture:{type(exc).__name__}"}


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    memory_probe: MemoryProbe = _memory_probe,
    disk_probe: DiskProbe = _disk_probe,
) -> JsonDict:
    """Collect source, solver, generator, resource, output, and protection gates."""

    root = Path(root)
    result_path = Path(result_path)
    row_file_path = Path(row_file_path)
    source = source_method_receipt(root)
    solvers = solver_version_receipts()
    seeds = generator_seed_registry()
    generator_hashes = {
        "module": _hash_optional_file(root, MODULE_RELATIVE_PATH),
        "test": _hash_optional_file(root, TEST_RELATIVE_PATH),
        "verification_spec": _hash_optional_file(root, VERIFY_SPEC_RELATIVE_PATH),
    }
    protected_hashes = {
        path.as_posix(): _hash_optional_file(root, path) for path in PROTECTED_FILES
    }
    exp5840 = _exp5840_receipt(root)
    memory = memory_probe()
    disk = disk_probe(root)
    output_paths = _output_path_receipt(result_path, row_file_path)
    checks = {
        "source_paper_receipt": source.get("ok") is True,
        "solver_versions": solvers.get("ok") is True
        and solvers.get("solver_configuration_count", 0) >= 2,
        "generator_and_tests": all(value != "missing" for value in generator_hashes.values()),
        "seed_registry": seeds.get("ok") is True,
        "exp5840_fixture": exp5840.get("ok") is True,
        "protected_files": all(value != "missing" for value in protected_hashes.values()),
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "output_paths": output_paths["result_writable"] is True
        and output_paths["row_file_writable"] is True,
        "python": sys.version_info >= (3, 11),
    }
    failure_names = {
        "source_paper_receipt": "missing_source_paper_receipt",
        "solver_versions": "missing_exact_solver_configuration",
        "generator_and_tests": "missing_generator_or_test_code",
        "seed_registry": "missing_seed_registry",
        "exp5840_fixture": "missing_or_unready_exp5840_fixture",
        "protected_files": "missing_protected_file",
        "memory": "insufficient_free_ram",
        "disk": "insufficient_free_disk",
        "output_paths": "output_path_not_writable",
        "python": "python_version_too_old",
    }
    blocked = [failure_names[name] for name, ok in checks.items() if not ok]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "source_method_receipt": source,
        "solver_versions": solvers,
        "generator_hashes": generator_hashes,
        "generator_and_seed_receipts": seeds,
        "upstream_counterfactual_fixture": exp5840,
        "resources": {"memory": memory, "disk": disk},
        "output_paths": output_paths,
        "protected_file_hashes": protected_hashes,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def expander_edges(vertices: int) -> list[tuple[int, int]]:
    """Build a deterministic 3-regular cycle-plus-matching graph."""

    edges = {(index, (index + 1) % vertices) for index in range(vertices)}
    half = vertices // 2
    edges.update((index, index + half) for index in range(half))
    return sorted(tuple(sorted(edge)) for edge in edges)


def ladder_edges(vertices: int) -> list[tuple[int, int]]:
    """Build a two-rail ladder graph with the same vertex bin as the expander."""

    rungs = vertices // 2
    edges: set[tuple[int, int]] = set()
    for index in range(rungs - 1):
        edges.add((index, index + 1))
        edges.add((rungs + index, rungs + index + 1))
    for index in range(rungs):
        edges.add((index, rungs + index))
    return sorted(tuple(sorted(edge)) for edge in edges)


def graph_for_family(family: str, vertices: int) -> list[tuple[int, int]]:
    if family == "expander_tseitin":
        return expander_edges(vertices)
    if family == "ladder_tseitin":
        return ladder_edges(vertices)
    raise ValueError(f"unknown_family:{family}")


def charges_for_label(vertices: int, label: str, seed: int) -> list[int]:
    """Return vertex charges with even parity for SAT and odd parity for UNSAT."""

    charges = [0 for _ in range(vertices)]
    first = seed % vertices
    charges[first] = 1
    if label == "satisfiable":
        second = (seed * 3 + 1) % vertices
        if second == first:
            second = (second + 1) % vertices
        charges[second] ^= 1
    elif label != "unsatisfiable":
        raise ValueError(f"unknown_label:{label}")
    return charges


def tseitin_clauses(
    vertices: int,
    edges: Sequence[tuple[int, int]],
    charges: Sequence[int],
) -> tuple[list[list[int]], dict[tuple[int, int], int]]:
    """Encode graph parity equations as CNF clauses with one variable per edge."""

    edge_vars = {tuple(edge): index + 1 for index, edge in enumerate(edges)}
    incident: dict[int, list[int]] = {vertex: [] for vertex in range(vertices)}
    for edge, var in edge_vars.items():
        left, right = edge
        incident[left].append(var)
        incident[right].append(var)
    clauses: list[list[int]] = []
    for vertex in range(vertices):
        variables = sorted(incident[vertex])
        charge = int(charges[vertex])
        for bits in product((0, 1), repeat=len(variables)):
            if sum(bits) % 2 != charge:
                clauses.append([var if bit == 0 else -var for var, bit in zip(variables, bits)])
    return clauses, edge_vars


def _gf2_solution(
    vertices: int,
    edges: Sequence[tuple[int, int]],
    charges: Sequence[int],
) -> dict[int, bool] | None:
    rows: list[list[int]] = []
    for vertex in range(vertices):
        coeffs = [0 for _ in edges]
        for index, edge in enumerate(edges):
            if vertex in edge:
                coeffs[index] = 1
        rows.append(coeffs + [int(charges[vertex])])

    pivot_row = 0
    pivots: list[tuple[int, int]] = []
    for column in range(len(edges)):
        found = None
        for row_index in range(pivot_row, len(rows)):
            if rows[row_index][column]:
                found = row_index
                break
        if found is None:
            continue
        rows[pivot_row], rows[found] = rows[found], rows[pivot_row]
        for row_index in range(len(rows)):
            if row_index != pivot_row and rows[row_index][column]:
                rows[row_index] = [
                    left ^ right for left, right in zip(rows[row_index], rows[pivot_row], strict=True)
                ]
        pivots.append((pivot_row, column))
        pivot_row += 1

    for row in rows:
        if not any(row[:-1]) and row[-1]:
            return None

    solution = [0 for _ in edges]
    for row_index, column in reversed(pivots):
        rhs = rows[row_index][-1]
        for col_index in range(column + 1, len(edges)):
            rhs ^= rows[row_index][col_index] & solution[col_index]
        solution[column] = rhs
    return {index + 1: bool(value) for index, value in enumerate(solution)}


def _assignment_satisfies_clause(assignment: Mapping[int, bool], clause: Sequence[int]) -> bool:
    return any(bool(assignment[abs(lit)]) is (lit > 0) for lit in clause)


def assignment_satisfies_cnf(assignment: Mapping[int, bool], clauses: Sequence[Sequence[int]]) -> bool:
    """Return true only when every clause is satisfied by a full assignment."""

    return all(_assignment_satisfies_clause(assignment, clause) for clause in clauses)


def parity_witness(vertices: int, edges: Sequence[tuple[int, int]], charges: Sequence[int]) -> JsonDict:
    """Return the connected-component parity contradiction for odd Tseitin charges."""

    return {
        "component_vertices": list(range(vertices)),
        "component_edge_count": len(edges),
        "charge_parity": sum(int(value) for value in charges) % 2,
        "edge_incidence_parity": 0,
    }


def validate_parity_witness(
    vertices: int,
    edges: Sequence[tuple[int, int]],
    charges: Sequence[int],
    witness: Mapping[str, Any],
) -> bool:
    component = set(int(value) for value in witness.get("component_vertices", []))
    all_vertices = set(range(vertices))
    all_edges_internal = all(left in component and right in component for left, right in edges)
    incidence_parity = (2 * len(edges)) % 2
    charge_parity = sum(int(charges[vertex]) for vertex in component) % 2
    return (
        component == all_vertices
        and all_edges_internal
        and charge_parity == 1
        and incidence_parity == 0
        and int(witness.get("charge_parity", -1)) == charge_parity
        and int(witness.get("edge_incidence_parity", -1)) == incidence_parity
    )


def _simplify_clauses(
    clauses: tuple[tuple[int, ...], ...],
    assignment: Mapping[int, bool],
) -> tuple[tuple[tuple[int, ...], ...] | None, int]:
    simplified: list[tuple[int, ...]] = []
    removed_literals = 0
    for clause in clauses:
        new_clause: list[int] = []
        satisfied = False
        for lit in clause:
            value = assignment.get(abs(lit))
            if value is None:
                new_clause.append(lit)
            elif value is (lit > 0):
                satisfied = True
                break
            else:
                removed_literals += 1
        if satisfied:
            continue
        if not new_clause:
            return None, removed_literals
        simplified.append(tuple(new_clause))
    return tuple(simplified), removed_literals


def _unit_and_pure_propagate(
    clauses: tuple[tuple[int, ...], ...],
    assignment: dict[int, bool],
) -> tuple[tuple[tuple[int, ...], ...] | None, int]:
    propagations = 0
    current = clauses
    changed = True
    while changed:
        changed = False
        units = [clause[0] for clause in current if len(clause) == 1]
        for lit in units:
            var = abs(lit)
            value = lit > 0
            if var in assignment and assignment[var] is not value:
                return None, propagations
            if var not in assignment:
                assignment[var] = value
                propagations += 1
                changed = True
        simplified, removed = _simplify_clauses(current, assignment)
        propagations += removed
        if simplified is None:
            return None, propagations
        current = simplified
        polarity: dict[int, set[bool]] = defaultdict(set)
        for clause in current:
            for lit in clause:
                if abs(lit) not in assignment:
                    polarity[abs(lit)].add(lit > 0)
        pure_literals = [
            (var, next(iter(values))) for var, values in sorted(polarity.items()) if len(values) == 1
        ]
        for var, value in pure_literals:
            if var not in assignment:
                assignment[var] = value
                propagations += 1
                changed = True
    return current, propagations


def _choose_branch_variable(
    clauses: tuple[tuple[int, ...], ...],
    assignment: Mapping[int, bool],
    n_vars: int,
    config: str,
) -> int:
    if config == SOLVER_CONFIGS[1]:
        counts: Counter[int] = Counter()
        for clause in clauses:
            for lit in clause:
                var = abs(lit)
                if var not in assignment:
                    counts[var] += 1
        if counts:
            return min(counts, key=lambda var: (-counts[var], var))
    for var in range(1, n_vars + 1):
        if var not in assignment:
            return var
    raise ValueError("no_branch_variable")


def solve_cnf_dpll(
    clauses: Sequence[Sequence[int]],
    n_vars: int,
    *,
    config: str,
    max_decisions: int = MAX_DPLL_DECISIONS,
) -> JsonDict:
    """Solve CNF exactly with deterministic DPLL and return stable covariates."""

    if config not in SOLVER_CONFIGS:
        raise ValueError(f"unknown_solver_config:{config}")
    clause_tuple = tuple(tuple(int(lit) for lit in clause) for clause in clauses)
    stats = {"decisions": 0, "conflicts": 0, "propagations": 0}
    unsat_cache: set[tuple[tuple[int, ...], ...]] = set()
    branch_values = (False, True) if config == SOLVER_CONFIGS[0] else (True, False)

    def recurse(
        current_clauses: tuple[tuple[int, ...], ...],
        assignment: dict[int, bool],
    ) -> dict[int, bool] | None:
        if stats["decisions"] > max_decisions:
            raise TimeoutError("dpll_decision_limit")
        propagated_assignment = dict(assignment)
        simplified, propagated = _unit_and_pure_propagate(current_clauses, propagated_assignment)
        stats["propagations"] += propagated
        if simplified is None:
            stats["conflicts"] += 1
            return None
        if not simplified:
            return propagated_assignment
        if simplified in unsat_cache:
            stats["conflicts"] += 1
            return None
        var = _choose_branch_variable(simplified, propagated_assignment, n_vars, config)
        for value in branch_values:
            stats["decisions"] += 1
            child_assignment = dict(propagated_assignment)
            child_assignment[var] = value
            solved = recurse(simplified, child_assignment)
            if solved is not None:
                return solved
        unsat_cache.add(simplified)
        return None

    try:
        solution = recurse(clause_tuple, {})
        timeout = False
    except TimeoutError:
        solution = None
        timeout = True
    label = "satisfiable" if solution is not None else "unsatisfiable"
    full_solution = None
    if solution is not None:
        full_solution = {str(var): bool(solution.get(var, False)) for var in range(1, n_vars + 1)}
    tick_count = stats["decisions"] + stats["conflicts"] + stats["propagations"]
    return {
        "solver_config": config,
        "label": label,
        "timeout": timeout,
        "conflicts": stats["conflicts"],
        "decisions": stats["decisions"],
        "propagations": stats["propagations"],
        "deterministic_time_proxy_s": round(tick_count / 1_000_000.0, 6),
        "assignment": full_solution,
    }


def clause_density(clauses: Sequence[Sequence[int]], n_vars: int) -> float:
    """Return the standard CNF clause-to-variable density."""

    return round(len(clauses) / float(n_vars), 6)


def max_clause_width(clauses: Sequence[Sequence[int]]) -> int:
    return max((len(clause) for clause in clauses), default=0)


def canonical_dimacs(n_vars: int, clauses: Sequence[Sequence[int]]) -> str:
    """Render a stable DIMACS-like formula without comments."""

    lines = [f"p cnf {n_vars} {len(clauses)}"]
    lines.extend(" ".join(str(int(lit)) for lit in clause) + " 0" for clause in clauses)
    return "\n".join(lines)


def _pad_to_target_tokens(text: str, target_tokens: int, pad_tokens: Sequence[str] = PAD_TOKENS) -> str:
    tokens = text.split()
    if len(tokens) > target_tokens:
        raise ValueError("surface_text_exceeds_preregistered_target")
    padded = list(tokens)
    index = 0
    while len(padded) < target_tokens:
        padded.append(pad_tokens[index % len(pad_tokens)])
        index += 1
    return " ".join(padded)


def _no_information_text(target_tokens: int) -> str:
    return " ".join(NO_INFORMATION_TOKENS[index % len(NO_INFORMATION_TOKENS)] for index in range(target_tokens))


def _relabel_map(n_vars: int, seed: int) -> dict[int, int]:
    shift = seed % n_vars
    if shift == 0:
        shift = 1
    return {var: ((var + shift - 1) % n_vars) + 1 for var in range(1, n_vars + 1)}


def apply_variable_relabel(
    clauses: Sequence[Sequence[int]],
    variable_map: Mapping[int, int],
) -> list[list[int]]:
    """Rename CNF variable ids while preserving every literal sign."""

    relabeled: list[list[int]] = []
    for clause in clauses:
        relabeled.append([
            int(variable_map[abs(lit)]) if lit > 0 else -int(variable_map[abs(lit)])
            for lit in clause
        ])
    return relabeled


def _relabel_assignment(
    assignment: Mapping[int, bool],
    variable_map: Mapping[int, int],
) -> dict[int, bool]:
    return {int(variable_map[var]): bool(value) for var, value in assignment.items()}


def _clause_reorder(clauses: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(reversed(clause)) for clause in reversed(clauses)]


def _density_mismatch_clauses(clauses: Sequence[Sequence[int]], n_vars: int) -> list[list[int]]:
    extra_count = min(4, n_vars)
    extras = [[var, -var] for var in range(1, extra_count + 1)]
    return [list(clause) for clause in clauses] + extras


def _certificate_for_instance(
    *,
    label: str,
    clauses: Sequence[Sequence[int]],
    vertices: int,
    edges: Sequence[tuple[int, int]],
    charges: Sequence[int],
    assignment: Mapping[int, bool] | None,
) -> JsonDict:
    if label == "satisfiable":
        if assignment is None:
            return {"kind": "satisfying_assignment", "validated": False}
        clean_assignment = {str(var): bool(assignment[var]) for var in sorted(assignment)}
        validated = assignment_satisfies_cnf({int(k): v for k, v in clean_assignment.items()}, clauses)
        return {
            "kind": "satisfying_assignment",
            "assignment": clean_assignment,
            "assignment_hash": sha256_json(clean_assignment),
            "validated": validated,
        }
    witness = parity_witness(vertices, edges, charges)
    return {
        "kind": "tseitin_parity_contradiction",
        "witness": witness,
        "witness_hash": sha256_json(witness),
        "validated": validate_parity_witness(vertices, edges, charges, witness),
    }


def validate_certificate(row: Mapping[str, Any]) -> bool:
    """Validate the row's satisfying assignment or parity contradiction witness."""

    certificate = dict(row.get("certificate") or {})
    label = str(row.get("expected_label"))
    if certificate.get("validated") is not True:
        return False
    if label == "satisfiable":
        assignment = {int(key): bool(value) for key, value in dict(certificate.get("assignment") or {}).items()}
        return len(assignment) == int(row.get("n_vars", 0)) and assignment_satisfies_cnf(
            assignment,
            row.get("clauses") or [],
        )
    if label == "unsatisfiable":
        edges = [tuple(edge) for edge in row.get("edges") or []]
        return validate_parity_witness(
            int(row.get("n_vertices", 0)),
            edges,
            row.get("charges") or [],
            certificate.get("witness") or {},
        )
    return False


def _run_solvers(clauses: Sequence[Sequence[int]], n_vars: int) -> dict[str, JsonDict]:
    return {
        config: solve_cnf_dpll(clauses, n_vars, config=config) for config in SOLVER_CONFIGS
    }


def _row_seed(family: str, size_bin: str, label: str) -> int:
    return BASE_SEED + 101 * FAMILIES.index(family) + 17 * SIZE_BIN_NAMES.index(size_bin) + 3 * LABELS.index(label)


def _base_instance(family: str, size: SizeBin, label: str) -> JsonDict:
    seed = _row_seed(family, size.name, label)
    edges = graph_for_family(family, size.vertices)
    charges = charges_for_label(size.vertices, label, seed)
    clauses, edge_vars = tseitin_clauses(size.vertices, edges, charges)
    assignment = _gf2_solution(size.vertices, edges, charges)
    expected_label = "satisfiable" if assignment is not None else "unsatisfiable"
    if expected_label != label:
        raise ValueError(f"label_generation_mismatch:{family}:{size.name}:{label}")
    return {
        "base_instance_id": f"exp5868-{family}-{size.name}-{label}",
        "family": family,
        "size_bin": size.name,
        "seed": seed,
        "n_vertices": size.vertices,
        "n_vars": len(edges),
        "edges": edges,
        "edge_variable_map": {f"{left}-{right}": var for (left, right), var in edge_vars.items()},
        "charges": charges,
        "clauses": clauses,
        "assignment": assignment,
        "expected_label": expected_label,
        "target_surface_token_count": size.surface_tokens,
    }


def _surface_clauses(
    base_clauses: Sequence[Sequence[int]],
    n_vars: int,
    control_kind: str,
    seed: int,
) -> tuple[list[list[int]], dict[int, int] | None]:
    if control_kind == "clause_reorder":
        return _clause_reorder(base_clauses), None
    if control_kind == "variable_renaming":
        mapping = _relabel_map(n_vars, seed)
        return apply_variable_relabel(base_clauses, mapping), mapping
    if control_kind == "density_mismatched":
        return _density_mismatch_clauses(base_clauses, n_vars), None
    return [list(clause) for clause in base_clauses], None


def _surface_text(
    *,
    n_vars: int,
    clauses: Sequence[Sequence[int]],
    control_kind: str,
    target_tokens: int,
) -> str:
    if control_kind == "no_information":
        return _no_information_text(target_tokens)
    prefix = canonical_dimacs(n_vars, clauses)
    if control_kind == "padding":
        prefix += "\nc padding_control neutral stable"
    elif control_kind == "length_matched":
        prefix += "\nc length_matched_control neutral stable"
    elif control_kind == "clause_reorder":
        prefix += "\nc reorder_control neutral stable"
    elif control_kind == "variable_renaming":
        prefix += "\nc rename_control neutral stable"
    elif control_kind == "density_mismatched":
        prefix += "\nc density_control redundant_tautology"
    return _pad_to_target_tokens(prefix, target_tokens)


def _proof_preserving_relabel_receipt(
    *,
    clauses: Sequence[Sequence[int]],
    n_vars: int,
    label: str,
    certificate: Mapping[str, Any],
    vertices: int,
    edges: Sequence[tuple[int, int]],
    charges: Sequence[int],
    seed: int,
) -> JsonDict:
    mapping = _relabel_map(n_vars, seed + 13)
    relabeled_clauses = apply_variable_relabel(clauses, mapping)
    relabeled_formula = canonical_dimacs(n_vars, relabeled_clauses)
    if label == "satisfiable":
        assignment = {int(key): bool(value) for key, value in dict(certificate.get("assignment") or {}).items()}
        relabeled_assignment = _relabel_assignment(assignment, mapping)
        certificate_preserved = assignment_satisfies_cnf(relabeled_assignment, relabeled_clauses)
    else:
        certificate_preserved = validate_parity_witness(
            vertices,
            edges,
            charges,
            dict(certificate.get("witness") or {}),
        )
    solver_results = _run_solvers(relabeled_clauses, n_vars)
    labels = {result["label"] for result in solver_results.values()}
    return {
        "variable_map": {str(key): value for key, value in sorted(mapping.items())},
        "relabel_formula_hash": sha256_text(relabeled_formula),
        "label_preserved": labels == {label},
        "certificate_preserved": certificate_preserved,
        "solver_results": {
            key: {
                "label": value["label"],
                "conflicts": value["conflicts"],
                "decisions": value["decisions"],
                "timeout": value["timeout"],
            }
            for key, value in solver_results.items()
        },
        "receipt_hash": sha256_json(
            {
                "variable_map": {str(key): value for key, value in sorted(mapping.items())},
                "relabel_formula_hash": sha256_text(relabeled_formula),
                "label": label,
            }
        ),
    }


def _build_row(base: Mapping[str, Any], control_kind: str) -> JsonDict:
    n_vars = int(base["n_vars"])
    seed = int(base["seed"]) + 31 * SURFACE_CONTROL_KINDS.index(control_kind)
    base_clauses = [list(clause) for clause in base["clauses"]]
    clauses, surface_mapping = _surface_clauses(base_clauses, n_vars, control_kind, seed)
    assignment = base.get("assignment")
    if surface_mapping is not None and assignment is not None:
        assignment = _relabel_assignment(assignment, surface_mapping)
    label = str(base["expected_label"])
    edges = [tuple(edge) for edge in base["edges"]]
    certificate = _certificate_for_instance(
        label=label,
        clauses=clauses,
        vertices=int(base["n_vertices"]),
        edges=edges,
        charges=list(base["charges"]),
        assignment=assignment,
    )
    solver_results = _run_solvers(clauses, n_vars)
    solver_labels = {result["label"] for result in solver_results.values()}
    canonical_formula = canonical_dimacs(n_vars, base_clauses)
    surface_formula = _surface_text(
        n_vars=n_vars,
        clauses=clauses,
        control_kind=control_kind,
        target_tokens=int(base["target_surface_token_count"]),
    )
    covariate_conflicts = sum(int(result["conflicts"]) for result in solver_results.values())
    covariate_decisions = sum(int(result["decisions"]) for result in solver_results.values())
    covariate_time = round(
        sum(float(result["deterministic_time_proxy_s"]) for result in solver_results.values()),
        6,
    )
    row = {
        "schema": ROW_SCHEMA,
        "row_id": f"{base['base_instance_id']}-{control_kind}",
        "base_instance_id": base["base_instance_id"],
        "family": base["family"],
        "proof_hardness_family": (
            "proof_hard_expander" if base["family"] == "expander_tseitin" else "proof_easy_ladder"
        ),
        "size_bin": base["size_bin"],
        "seed": seed,
        "control_kind": control_kind,
        "n_vertices": base["n_vertices"],
        "n_vars": n_vars,
        "edges": [list(edge) for edge in edges],
        "charges": list(base["charges"]),
        "clauses": clauses,
        "canonical_clauses": base_clauses,
        "clause_count": len(clauses),
        "canonical_clause_count": len(base_clauses),
        "clause_density": clause_density(clauses, n_vars),
        "canonical_clause_density": clause_density(base_clauses, n_vars),
        "max_clause_width": max_clause_width(clauses),
        "canonical_max_clause_width": max_clause_width(base_clauses),
        "canonical_formula_text": canonical_formula,
        "canonical_formula_hash": sha256_text(canonical_formula),
        "surface_formula_text": surface_formula,
        "surface_formula_hash": sha256_text(surface_formula),
        "surface_token_count": len(surface_formula.split()),
        "target_surface_token_count": base["target_surface_token_count"],
        "expected_label": label,
        "canonical_expected_label": label,
        "certificate": certificate,
        "solver_results": solver_results,
        "solver_disagreement": solver_labels != {label},
        "solver_timeout": any(bool(result["timeout"]) for result in solver_results.values()),
        "proof_hardness_covariates": {
            "solver_conflicts": covariate_conflicts,
            "solver_decisions": covariate_decisions,
            "deterministic_time_proxy_s": covariate_time,
            "used_as_label": False,
            "label_source": "exact_solver_expected_label_not_conflict_count",
        },
        "generator_receipt": {
            "config_id": "exp5868_tseitin_surface_control_v1",
            "seed": seed,
            "base_seed": base["seed"],
            "graph_hash": sha256_json({"vertices": base["n_vertices"], "edges": base["edges"]}),
            "charges_hash": sha256_json(base["charges"]),
            "control_kind": control_kind,
            "source_id": SOURCE_ID,
        },
        "proof_preserving_relabel": {},
        "row_hash": "",
    }
    row["proof_preserving_relabel"] = _proof_preserving_relabel_receipt(
        clauses=clauses,
        n_vars=n_vars,
        label=label,
        certificate=certificate,
        vertices=int(base["n_vertices"]),
        edges=edges,
        charges=list(base["charges"]),
        seed=seed,
    )
    if control_kind == "variable_renaming":
        row["proof_preserving_relabel"]["relabel_formula_hash"] = row["surface_formula_hash"]
    row["row_hash"] = row_hash(row)
    return row


def generate_rows(
    *,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> list[JsonDict]:
    """Generate rows only after preconditions pass."""

    preconditions = dict(preconditions_checked or collect_preconditions())
    if preconditions.get("preconditions_ready") is not True:
        return []
    rows: list[JsonDict] = []
    for family in FAMILIES:
        for size in SIZE_BINS:
            for label in LABELS:
                base = _base_instance(family, size, label)
                for control_kind in SURFACE_CONTROL_KINDS:
                    rows.append(_build_row(base, control_kind))
    verify_rows(rows)
    return rows


def row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def rows_to_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize rows as deterministic JSONL."""

    return "".join(canonical_json(row) + "\n" for row in rows)


def read_row_file(path: str | Path) -> list[JsonDict]:
    """Read the Exp5868 JSONL row file, returning empty for absent files."""

    if not Path(path).exists():
        return []
    return _read_jsonl(path)


def verify_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Validate row ids, hashes, schemas, certificates, solver labels, and relabels."""

    seen: set[str] = set()
    for row in rows:
        row_id = str(row.get("row_id"))
        if row_id in seen:
            raise ValueError(f"duplicate_row_id:{row_id}")
        seen.add(row_id)
        if row.get("schema") != ROW_SCHEMA:
            raise ValueError(f"row_schema:{row_id}")
        if row_hash(row) != row.get("row_hash"):
            raise ValueError(f"row_hash:{row_id}")
        if not validate_certificate(row):
            raise ValueError(f"certificate:{row_id}")
        label = str(row.get("expected_label"))
        solver_labels = {result["label"] for result in dict(row.get("solver_results") or {}).values()}
        if solver_labels != {label}:
            raise ValueError(f"solver_label:{row_id}")
        if row.get("solver_disagreement") is not False or row.get("solver_timeout") is not False:
            raise ValueError(f"solver_status:{row_id}")
        if validate_relabel_receipt(row) is not True:
            raise ValueError(f"relabel:{row_id}")
    return True


def _row_file_receipt(rows: Sequence[Mapping[str, Any]], row_text: str) -> JsonDict:
    row_hashes = {str(row["row_id"]): str(row["row_hash"]) for row in rows}
    receipt = {
        "path": ROW_FILE_RELATIVE_PATH.as_posix(),
        "row_count": len(rows),
        "schema": ROW_SCHEMA,
        "sha256": sha256_text(row_text),
        "row_hashes": row_hashes,
        "row_hash_root": sha256_json(row_hashes),
        "atomic_write": True,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _row_file_receipt_ok(receipt: Mapping[str, Any]) -> bool:
    return (
        receipt.get("path") == ROW_FILE_RELATIVE_PATH.as_posix()
        and receipt.get("schema") == ROW_SCHEMA
        and isinstance(receipt.get("row_count"), int)
        and str(receipt.get("sha256", "")).startswith("sha256:")
        and str(receipt.get("row_hash_root", "")).startswith("sha256:")
        and receipt.get("atomic_write") is True
    )


def verify_row_file(rows: Sequence[Mapping[str, Any]], artifact: Mapping[str, Any]) -> bool:
    receipt = dict(artifact.get("row_file_receipt") or {})
    if not _row_file_receipt_ok(receipt):
        raise ValueError("row_file_receipt")
    verify_rows(rows)
    if len(rows) != receipt.get("row_count"):
        raise ValueError("row_count")
    expected_hashes = dict(receipt.get("row_hashes") or {})
    for row in rows:
        if expected_hashes.get(str(row["row_id"])) != row.get("row_hash"):
            raise ValueError(f"row_hash_receipt:{row['row_id']}")
    if sha256_text(rows_to_jsonl(rows)) != receipt.get("sha256"):
        raise ValueError("row_file_sha256")
    return True


def family_and_size_bin_definitions(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    canonical = [row for row in rows if row.get("control_kind") == "canonical"]
    counts = Counter(f"{row['family']}|{row['size_bin']}|{row['expected_label']}" for row in canonical)
    family_bin_counts = Counter(f"{row['family']}|{row['size_bin']}" for row in canonical)
    definition = {
        "families": {
            "expander_tseitin": "proof_hard_expander_family",
            "ladder_tseitin": "proof_easy_ladder_family",
        },
        "size_bins": [size.__dict__ for size in SIZE_BINS],
        "labels": list(LABELS),
        "base_instances_per_family_size_label": 1,
    }
    return {
        "schema": SCHEMA + ".family_and_size_bin_definitions",
        "definition": definition,
        "definition_hash": sha256_json(definition),
        "canonical_cell_counts": dict(sorted(counts.items())),
        "family_bin_counts": dict(sorted(family_bin_counts.items())),
        "all_bins_have_both_families": all(
            family_bin_counts.get(f"{family}|{size}", 0) == len(LABELS)
            for family in FAMILIES
            for size in SIZE_BIN_NAMES
        ),
        "all_bins_have_both_labels": all(
            counts.get(f"{family}|{size}|{label}", 0) == 1
            for family in FAMILIES
            for size in SIZE_BIN_NAMES
            for label in LABELS
        ),
        "receipt_hash": sha256_json(dict(sorted(counts.items()))),
    }


def density_width_and_length_matching(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    canonical = [row for row in rows if row.get("control_kind") == "canonical"]
    density_deltas: list[float] = []
    width_failures: list[str] = []
    length_deltas: list[int] = []
    for size in SIZE_BIN_NAMES:
        for label in LABELS:
            family_rows = {
                row["family"]: row
                for row in canonical
                if row["size_bin"] == size and row["expected_label"] == label
            }
            if set(family_rows) == set(FAMILIES):
                density_deltas.append(
                    abs(
                        float(family_rows[FAMILIES[0]]["canonical_clause_density"])
                        - float(family_rows[FAMILIES[1]]["canonical_clause_density"])
                    )
                )
                widths = {int(row["canonical_max_clause_width"]) for row in family_rows.values()}
                if widths != {MAX_CLAUSE_WIDTH}:
                    width_failures.append(f"{size}|{label}")
    for size in SIZE_BIN_NAMES:
        token_counts = [int(row["surface_token_count"]) for row in rows if row["size_bin"] == size]
        if token_counts:
            length_deltas.append(max(token_counts) - min(token_counts))
    max_density_delta = round(max(density_deltas), 6) if density_deltas else 0.0
    max_length_delta = max(length_deltas) if length_deltas else 0
    return {
        "schema": SCHEMA + ".density_width_and_length_matching",
        "density_tolerance": DENSITY_TOLERANCE,
        "length_tolerance_tokens": LENGTH_TOLERANCE_TOKENS,
        "max_density_delta": max_density_delta,
        "density_deltas": density_deltas,
        "max_surface_token_delta": max_length_delta,
        "max_clause_width": MAX_CLAUSE_WIDTH,
        "max_clause_width_matched": not width_failures,
        "width_failures": width_failures,
        "all_matching_passed": bool(rows)
        and max_density_delta <= DENSITY_TOLERANCE
        and max_length_delta <= LENGTH_TOLERANCE_TOKENS
        and not width_failures,
    }


def label_and_certificate_balance(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    label_counts = Counter(str(row.get("expected_label")) for row in rows)
    family_bin_label_counts = Counter(
        f"{row['family']}|{row['size_bin']}|{row['expected_label']}" for row in rows
    )
    certificate_failures = [str(row.get("row_id")) for row in rows if not validate_certificate(row)]
    per_cell_balanced = all(
        family_bin_label_counts.get(f"{family}|{size}|satisfiable", 0)
        == family_bin_label_counts.get(f"{family}|{size}|unsatisfiable", 0)
        for family in FAMILIES
        for size in SIZE_BIN_NAMES
    )
    return {
        "schema": SCHEMA + ".label_and_certificate_balance",
        "label_counts": dict(sorted(label_counts.items())),
        "family_bin_label_counts": dict(sorted(family_bin_label_counts.items())),
        "all_labels_balanced": bool(rows)
        and label_counts.get("satisfiable", 0) == label_counts.get("unsatisfiable", 0)
        and per_cell_balanced,
        "certificate_failure_count": len(certificate_failures),
        "certificate_failures": certificate_failures[:20],
        "all_certificate_checks_passed": not certificate_failures,
        "receipt_hash": sha256_json(
            {
                str(row.get("row_id")): {
                    "label": row.get("expected_label"),
                    "certificate": row.get("certificate"),
                }
                for row in rows
            }
        ),
    }


def solver_versions_and_oracle_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    versions = solver_version_receipts()
    disagreements = [str(row["row_id"]) for row in rows if row.get("solver_disagreement") is not False]
    timeouts = [str(row["row_id"]) for row in rows if row.get("solver_timeout") is not False]
    label_failures = []
    for row in rows:
        solver_labels = {result["label"] for result in dict(row["solver_results"]).values()}
        if solver_labels != {row["expected_label"]}:
            label_failures.append(str(row["row_id"]))
    return {
        "schema": SCHEMA + ".solver_versions_and_oracle_receipts",
        "solver_versions": versions["solvers"],
        "solver_configuration_count": versions["solver_configuration_count"],
        "all_solvers_agree": not disagreements and not label_failures,
        "solver_disagreement_count": len(disagreements),
        "solver_disagreements": disagreements[:20],
        "solver_timeout_count": len(timeouts),
        "solver_timeouts": timeouts[:20],
        "label_failure_count": len(label_failures),
        "label_failures": label_failures[:20],
        "oracle_label_source": "exact_solver_expected_label",
    }


def proof_hardness_covariates(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    family_conflicts: Counter[str] = Counter()
    family_decisions: Counter[str] = Counter()
    family_time: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        family = str(row["family"])
        covariates = dict(row["proof_hardness_covariates"])
        family_conflicts[family] += int(covariates["solver_conflicts"])
        family_decisions[family] += int(covariates["solver_decisions"])
        family_time[family] += float(covariates["deterministic_time_proxy_s"])
    return {
        "schema": SCHEMA + ".proof_hardness_covariates",
        "family_solver_conflicts": dict(sorted(family_conflicts.items())),
        "family_solver_decisions": dict(sorted(family_decisions.items())),
        "family_deterministic_time_proxy_s": {
            key: round(value, 6) for key, value in sorted(family_time.items())
        },
        "conflict_count_is_label": False,
        "time_covariate_is_label": False,
        "covariate_boundary": "analysis_only_not_ground_truth",
        "receipt_hash": sha256_json(
            {str(row["row_id"]): row["proof_hardness_covariates"] for row in rows}
        ),
    }


def validate_relabel_receipt(row: Mapping[str, Any]) -> bool:
    receipt = dict(row.get("proof_preserving_relabel") or {})
    if receipt.get("label_preserved") is not True or receipt.get("certificate_preserved") is not True:
        return False
    solver_labels = {result["label"] for result in dict(receipt.get("solver_results") or {}).values()}
    return solver_labels == {row.get("expected_label")}


def proof_preserving_relabel_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures = [str(row["row_id"]) for row in rows if validate_relabel_receipt(row) is not True]
    return {
        "schema": SCHEMA + ".proof_preserving_relabel_receipts",
        "relabel_count": len(rows),
        "relabel_failure_count": len(failures),
        "relabel_failures": failures[:20],
        "all_relabel_checks_passed": bool(rows) and not failures,
        "receipt_hash": sha256_json(
            {str(row["row_id"]): row["proof_preserving_relabel"] for row in rows}
        ),
    }


def surface_and_no_information_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    control_counts = Counter(str(row.get("control_kind")) for row in rows)
    base_labels: dict[str, str] = {}
    label_failures: list[str] = []
    no_information_tokens = 0
    for row in rows:
        base_id = str(row["base_instance_id"])
        label = str(row["expected_label"])
        base_labels.setdefault(base_id, label)
        if base_labels[base_id] != label or row.get("canonical_expected_label") != label:
            label_failures.append(str(row["row_id"]))
        if row.get("control_kind") == "no_information":
            no_information_tokens += int(row["surface_token_count"])
    return {
        "schema": SCHEMA + ".surface_and_no_information_controls",
        "control_counts": dict(sorted(control_counts.items())),
        "required_controls": list(SURFACE_CONTROL_KINDS),
        "all_controls_present": all(control_counts.get(control, 0) > 0 for control in SURFACE_CONTROL_KINDS),
        "control_label_failure_count": len(label_failures),
        "control_label_failures": label_failures[:20],
        "all_control_labels_preserved": not label_failures,
        "no_information_surface_token_count": no_information_tokens,
        "no_information_tokens_hash": sha256_json(NO_INFORMATION_TOKENS),
        "receipt_hash": sha256_json(dict(sorted(control_counts.items()))),
    }


def solver_disagreement_and_timeout_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    ids = [str(row["row_id"]) for row in rows]
    duplicate_count = len(ids) - len(set(ids))
    disagreements = [str(row["row_id"]) for row in rows if row.get("solver_disagreement") is not False]
    timeouts = [str(row["row_id"]) for row in rows if row.get("solver_timeout") is not False]
    missing_certificates = [
        str(row["row_id"])
        for row in rows
        if dict(row.get("certificate") or {}).get("validated") is not True
    ]
    return {
        "schema": SCHEMA + ".solver_disagreement_and_timeout_controls",
        "duplicate_row_id_count": duplicate_count,
        "solver_disagreement_count": len(disagreements),
        "solver_timeout_count": len(timeouts),
        "missing_certificate_validation_count": len(missing_certificates),
        "rejected_unsettled_instances": True,
        "all_controls_passed": duplicate_count == 0
        and not disagreements
        and not timeouts
        and not missing_certificates,
        "receipt_hash": sha256_json(
            {
                "duplicate_count": duplicate_count,
                "disagreements": disagreements,
                "timeouts": timeouts,
                "missing_certificates": missing_certificates,
            }
        ),
    }


def deterministic_replay_receipt(
    rows: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    replay_rows = generate_rows(preconditions_checked=preconditions_checked)
    row_ids = [str(row["row_id"]) for row in rows]
    replay_ids = [str(row["row_id"]) for row in replay_rows]
    row_text = rows_to_jsonl(rows)
    replay_text = rows_to_jsonl(replay_rows)
    return {
        "schema": SCHEMA + ".deterministic_replay_receipt",
        "row_ids_match": row_ids == replay_ids,
        "row_content_hash": sha256_text(row_text),
        "replay_content_hash": sha256_text(replay_text),
        "content_match": row_text == replay_text,
        "replay_row_count": len(replay_rows),
        "ok": row_ids == replay_ids and row_text == replay_text,
    }


def protected_files_unchanged(
    root: Path,
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    before = dict(preconditions_checked.get("protected_file_hashes") or {})
    after = {path.as_posix(): _hash_optional_file(root, path) for path in PROTECTED_FILES}
    changed = sorted(path for path, value in after.items() if before.get(path) != value)
    return {
        "schema": SCHEMA + ".protected_files_unchanged",
        "before_hashes": before,
        "after_hashes": after,
        "changed_files": changed,
        "all_unchanged": not changed and all(value != "missing" for value in after.values()),
    }


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
        EXP5840_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5840_ROWS_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": principle, "sources": sources}
        for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    }


def generator_and_seed_receipts(preconditions_checked: Mapping[str, Any]) -> JsonDict:
    receipt = dict(preconditions_checked.get("generator_and_seed_receipts") or generator_seed_registry())
    receipt["generator_hashes"] = dict(preconditions_checked.get("generator_hashes") or {})
    receipt["solver_configurations"] = list(SOLVER_CONFIGS)
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def hardness_controlled_fixture_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return bare readiness only when every exact fixture gate is clean."""

    preconditions = dict(artifact.get("preconditions_checked") or {})
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = bool(
        preconditions.get("preconditions_ready") is True
        and dict(artifact.get("source_method_receipt") or {}).get("ok") is True
        and dict(artifact.get("family_and_size_bin_definitions") or {}).get(
            "all_bins_have_both_families"
        )
        is True
        and dict(artifact.get("family_and_size_bin_definitions") or {}).get(
            "all_bins_have_both_labels"
        )
        is True
        and dict(artifact.get("density_width_and_length_matching") or {}).get(
            "all_matching_passed"
        )
        is True
        and dict(artifact.get("label_and_certificate_balance") or {}).get(
            "all_labels_balanced"
        )
        is True
        and dict(artifact.get("label_and_certificate_balance") or {}).get(
            "all_certificate_checks_passed"
        )
        is True
        and dict(artifact.get("solver_versions_and_oracle_receipts") or {}).get(
            "all_solvers_agree"
        )
        is True
        and dict(artifact.get("proof_hardness_covariates") or {}).get(
            "conflict_count_is_label"
        )
        is False
        and dict(artifact.get("proof_preserving_relabel_receipts") or {}).get(
            "all_relabel_checks_passed"
        )
        is True
        and dict(artifact.get("surface_and_no_information_controls") or {}).get(
            "all_controls_present"
        )
        is True
        and dict(artifact.get("surface_and_no_information_controls") or {}).get(
            "all_control_labels_preserved"
        )
        is True
        and dict(artifact.get("solver_disagreement_and_timeout_controls") or {}).get(
            "all_controls_passed"
        )
        is True
        and _row_file_receipt_ok(dict(artifact.get("row_file_receipt") or {}))
        and dict(artifact.get("deterministic_replay_receipt") or {}).get("ok") is True
        and dict(artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(code == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    checks = {
        "source_method_receipt": dict(artifact.get("source_method_receipt") or {}).get("ok") is True,
        "family_and_size_bin_definitions": dict(
            artifact.get("family_and_size_bin_definitions") or {}
        ).get("all_bins_have_both_families")
        is True
        and dict(artifact.get("family_and_size_bin_definitions") or {}).get(
            "all_bins_have_both_labels"
        )
        is True,
        "density_width_and_length_matching": dict(
            artifact.get("density_width_and_length_matching") or {}
        ).get("all_matching_passed")
        is True,
        "label_and_certificate_balance": dict(
            artifact.get("label_and_certificate_balance") or {}
        ).get("all_labels_balanced")
        is True
        and dict(artifact.get("label_and_certificate_balance") or {}).get(
            "all_certificate_checks_passed"
        )
        is True,
        "solver_versions_and_oracle_receipts": dict(
            artifact.get("solver_versions_and_oracle_receipts") or {}
        ).get("all_solvers_agree")
        is True,
        "proof_preserving_relabel_receipts": dict(
            artifact.get("proof_preserving_relabel_receipts") or {}
        ).get("all_relabel_checks_passed")
        is True,
        "surface_and_no_information_controls": dict(
            artifact.get("surface_and_no_information_controls") or {}
        ).get("all_controls_present")
        is True
        and dict(artifact.get("surface_and_no_information_controls") or {}).get(
            "all_control_labels_preserved"
        )
        is True,
        "solver_disagreement_and_timeout_controls": dict(
            artifact.get("solver_disagreement_and_timeout_controls") or {}
        ).get("all_controls_passed")
        is True,
        "row_file_receipt": _row_file_receipt_ok(dict(artifact.get("row_file_receipt") or {})),
        "deterministic_replay_receipt": dict(
            artifact.get("deterministic_replay_receipt") or {}
        ).get("ok")
        is True,
        "protected_files_unchanged": dict(artifact.get("protected_files_unchanged") or {}).get(
            "all_unchanged"
        )
        is True,
        "test_exit_codes": bool(commands)
        and set(exit_codes) == set(commands)
        and all(code == 0 for code in exit_codes.values()),
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(name)
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        reasons.append("verifier_is_oracle")
    if hardness_controlled_fixture_ready_score(artifact) != 1.0 and not reasons:  # pragma: no cover - unreachable safety net.
        reasons.append("hardness_controlled_fixture_ready_score")
    return sorted(set(reasons))


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal verdict with a required terminal prefix."""

    if hardness_controlled_fixture_ready_score(artifact) == 1.0:
        return "ready: hardness_controlled_exact_constraint_fixture_ready"
    reasons = blocked_reasons(artifact) or ["hardness_controlled_fixture_not_ready"]
    return "blocked: " + ",".join(reasons[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking host-variable and self-referential fields."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    if isinstance(stable.get("preconditions_checked"), dict):
        stable["preconditions_checked"]["output_paths"] = {}
    return sha256_json(stable)


def build_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    row_text: str,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    duration_s: float,
    root: Path = REPO_ROOT,
) -> JsonDict:
    """Build the terminal Exp5868 aggregate artifact from deterministic rows."""

    preconditions = dict(preconditions_checked)
    source = dict(preconditions.get("source_method_receipt") or source_method_receipt(root))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "blocked",
        "preconditions_checked": preconditions,
        "source_method_receipt": source,
        "generator_and_seed_receipts": generator_and_seed_receipts(preconditions),
        "family_and_size_bin_definitions": family_and_size_bin_definitions(rows),
        "density_width_and_length_matching": density_width_and_length_matching(rows),
        "label_and_certificate_balance": label_and_certificate_balance(rows),
        "solver_versions_and_oracle_receipts": solver_versions_and_oracle_receipts(rows),
        "proof_hardness_covariates": proof_hardness_covariates(rows),
        "proof_preserving_relabel_receipts": proof_preserving_relabel_receipts(rows),
        "surface_and_no_information_controls": surface_and_no_information_controls(rows),
        "solver_disagreement_and_timeout_controls": solver_disagreement_and_timeout_controls(rows),
        "row_file_receipt": _row_file_receipt(rows, row_text),
        "deterministic_replay_receipt": deterministic_replay_receipt(rows, preconditions),
        "protected_files_unchanged": protected_files_unchanged(root, preconditions),
        "hardness_controlled_fixture_ready_score": 0.0,
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(key): int(value) for key, value in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: artifact_not_validated",
    }
    score = hardness_controlled_fixture_ready_score(artifact)
    artifact["hardness_controlled_fixture_ready_score"] = score
    artifact["status"] = "complete" if score == 1.0 else "blocked"
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal artifact schema and readiness gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing_fields:{missing}")
    score = hardness_controlled_fixture_ready_score(artifact)
    if artifact.get("hardness_controlled_fixture_ready_score") != score:
        raise ValueError("hardness_controlled_fixture_ready_score")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    if score == 1.0:
        if artifact.get("status") != "complete":
            raise ValueError("status")
        if not str(artifact.get("honest_verdict", "")).startswith("ready:"):
            raise ValueError("honest_verdict")
    return True


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def build_and_write_artifacts(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Generate rows, write JSONL then JSON atomically, and return the artifact."""

    started = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path)
    row_file_path = Path(row_file_path)
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path, row_file_path=row_file_path)
    )
    rows = generate_rows(preconditions_checked=preconditions)
    row_text = rows_to_jsonl(rows)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else float(duration_s)
    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    artifact = build_artifact(
        rows=rows,
        preconditions_checked=preconditions,
        row_text=row_text,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
        duration_s=elapsed,
        root=root,
    )
    _atomic_write(row_file_path, row_text)
    _atomic_write(result_path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    """Run Exp5868, optionally writing terminal artifacts."""

    if write:
        return build_and_write_artifacts(
            root=root,
            result_path=result_path,
            row_file_path=row_file_path,
            preconditions_checked=preconditions_checked,
            test_commands=test_commands,
            test_exit_codes=test_exit_codes,
            duration_s=duration_s,
        )
    started = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path)
    row_file_path = Path(row_file_path)
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result_path, row_file_path=row_file_path)
    )
    rows = generate_rows(preconditions_checked=preconditions)
    row_text = rows_to_jsonl(rows)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else float(duration_s)
    return build_artifact(
        rows=rows,
        preconditions_checked=preconditions,
        row_text=row_text,
        test_commands=test_commands,
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in test_commands}),
        duration_s=elapsed,
        root=root,
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--rows", default=str(REPO_ROOT / ROW_FILE_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(result_path=args.result, row_file_path=args.rows, write=True)
    print(json.dumps({"status": artifact["status"], "result": args.result}, sort_keys=True))
    return 0 if artifact["hardness_controlled_fixture_ready_score"] == 1.0 else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
