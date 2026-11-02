"""Build a frozen SAT/UNSAT stream with independently checked evidence.

The exact solver chooses labels. A separate checker reads only the CNF and the
certificate. Solver-work counters stay diagnostic because they do not measure
model difficulty.

Spec refs: REQ-VERIFY-6744, SCENARIO-VERIFY-6744-GENERATION,
SCENARIO-VERIFY-6744-CERTIFICATES, SCENARIO-VERIFY-6744-RELABEL,
SCENARIO-VERIFY-6744-SPLIT, and SCENARIO-VERIFY-6744-REPLAY.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]
Clock = Callable[[], float]
PreconditionProbe = Callable[[Path], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6744_hardness_controlled_certificate_stream.json")
SPEC_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
SCHEMA = "carnot.experiment_6744.hardness_controlled_certificate_stream.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 6744
RUN_DATE = "20260829"
INFERENCE_SUBSTRATE = "deterministic_cpu_exact_certificate_generation_and_independent_checking"

FAMILIES = ("expander_tseitin", "ladder_tseitin", "pigeonhole_anchor")
LABELS = ("SAT", "UNSAT")
SIZE_BINS: dict[str, dict[str, int]] = {
    "small": {"tseitin_vertices": 4, "pigeonhole_holes": 2},
    "medium": {"tseitin_vertices": 6, "pigeonhole_holes": 3},
}
SEEDS = (674401, 674402, 674403)
SPLIT_BY_FAMILY = {
    "expander_tseitin": "train",
    "ladder_tseitin": "dev",
    "pigeonhole_anchor": "test",
}

SOLVER_IDENTITY = "carnot_exact_dpll_false_first"
SOLVER_VERSION = "1.0"
CHECKER_IDENTITY = "carnot_exhaustive_certificate_checker"
CHECKER_VERSION = "1.0"
SOLVER_SHA256 = "sha256:" + hashlib.sha256(b"carnot_exact_dpll_false_first_v1").hexdigest()
CHECKER_SHA256 = "sha256:" + hashlib.sha256(b"carnot_exhaustive_certificate_checker_v1").hexdigest()

ARTIFACT_FIELDS = (
    "experiment",
    "schema",
    "run_date",
    "title",
    "status",
    "preconditions_checked",
    "preregistered_manifest",
    "field_principles",
    "inference_substrate",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "rows",
    "family_counts",
    "split_manifest",
    "certificate_checker_receipts",
    "relabel_pair_receipts",
    "solver_work_metadata",
    "deterministic_replay_receipt",
    "verification_commands",
    "hardness_stream_ready",
    "gate_check_summary",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLE_TEXT = {
    "experiment": "The numeric identifier binds the artifact to one task.",
    "schema": "The schema version prevents silent field reinterpretation.",
    "run_date": "The planning date fixes the intended research window.",
    "title": "The title states the narrow certificate-stream purpose.",
    "status": "The terminal state distinguishes ready output from an owned block.",
    "preconditions_checked": "Exact paths, seeds, and output access must pass before generation.",
    "preregistered_manifest": "Counts and row identities exist before solver labels are opened.",
    "field_principles": "Every field states why it exists and how downstream work may use it.",
    "inference_substrate": "The declared CPU path excludes LLM and hidden-state authority.",
    "duration_s": "A monotonic duration records real task-owned execution time.",
    "random_seed": "All generation seeds are explicit and deterministic.",
    "reproducibility_checksum": "The canonical stream hash detects any mathematical row drift.",
    "rows": "Every formula and exact certificate remains available for independent replay.",
    "family_counts": "Pre-registered family and label balance prevents selective inclusion.",
    "split_manifest": "Family-disjoint assignments prevent train, dev, or test leakage.",
    "certificate_checker_receipts": "Independent path identities, versions, codes, and hashes bind each label.",
    "relabel_pair_receipts": "Pair checks prove that symbol changes preserve exact structure and truth.",
    "solver_work_metadata": "Conflicts, decisions, propagations, and time are diagnostics only.",
    "deterministic_replay_receipt": "A second generation must reproduce the frozen stream hash.",
    "verification_commands": "Task-owned exact checks must all return code zero before readiness.",
    "hardness_stream_ready": "This future gate is true only after all exact row, pair, split, and replay checks pass.",
    "gate_check_summary": "A blocked or partial result names its first failed check and observed value.",
    "verdict_class": "The controlled terminal class prevents an ambiguous positive claim.",
    "honest_verdict": "An allowed terminal prefix states the exact claim boundary.",
}


def canonical_json(value: Any) -> str:
    """Return stable JSON so hashes do not depend on dictionary order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    """Hash canonical JSON with an explicit algorithm prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stable_integer(value: str) -> int:
    """Convert text to a deterministic random seed without Python hash salt."""

    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _without(value: Mapping[str, Any], *keys: str) -> JsonDict:
    return {key: deepcopy(item) for key, item in value.items() if key not in keys}


def _row_hash(row: Mapping[str, Any]) -> str:
    return sha256_json(_without(row, "row_sha256"))


def _receipt_hash(receipt: Mapping[str, Any]) -> str:
    return sha256_json(_without(receipt, "receipt_sha256"))


def _pair_id(family: str, size_bin: str, label: str, seed: int) -> str:
    return f"exp6744-{family}-{size_bin}-{label.lower()}-{seed}"


def build_preregistered_manifest() -> JsonDict:
    """Freeze the 36 base configurations before any solver is called."""

    base_rows = []
    for family in FAMILIES:
        for size_bin in SIZE_BINS:
            for label in LABELS:
                for seed in SEEDS:
                    base_rows.append(
                        {
                            "pair_id": _pair_id(family, size_bin, label, seed),
                            "family": family,
                            "size_bin": size_bin,
                            "label": label,
                            "seed": seed,
                            "split": SPLIT_BY_FAMILY[family],
                            "mate_count": 2,
                        }
                    )
    family_counts = {
        family: {
            label: 2 * sum(row["family"] == family and row["label"] == label for row in base_rows)
            for label in LABELS
        }
        for family in FAMILIES
    }
    for counts in family_counts.values():
        counts["total"] = sum(counts[label] for label in LABELS)
    payload: JsonDict = {
        "schema": SCHEMA + ".preregistered_manifest",
        "frozen_before_solving": True,
        "families": list(FAMILIES),
        "labels": list(LABELS),
        "size_bins": deepcopy(SIZE_BINS),
        "seeds": list(SEEDS),
        "base_rows": base_rows,
        "pair_count": len(base_rows),
        "row_count": len(base_rows) * 2,
        "family_counts": family_counts,
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def _expander_edges(vertices: int) -> list[tuple[int, int]]:
    """Build a cycle plus opposite matching for an even vertex count."""

    edges = {tuple(sorted((index, (index + 1) % vertices))) for index in range(vertices)}
    edges.update((index, index + vertices // 2) for index in range(vertices // 2))
    return sorted(edges)


def _ladder_edges(vertices: int) -> list[tuple[int, int]]:
    """Build two rails and deterministic rungs on the same vertex bins."""

    rungs = vertices // 2
    edges = {(index, rungs + index) for index in range(rungs)}
    for index in range(rungs - 1):
        edges.add((index, index + 1))
        edges.add((rungs + index, rungs + index + 1))
    return sorted(edges)


def _parity_clauses(variables: Sequence[int], charge: int) -> list[list[int]]:
    """Forbid every local edge assignment with the wrong parity."""

    clauses = []
    for bits in product((False, True), repeat=len(variables)):
        if sum(bits) % 2 == charge:
            continue
        clauses.append([-variable if bit else variable for variable, bit in zip(variables, bits)])
    return clauses


def _tseitin_formula(family: str, vertices: int, label: str, seed: int) -> JsonDict:
    edges = _expander_edges(vertices) if family == "expander_tseitin" else _ladder_edges(vertices)
    edge_variables = {edge: index + 1 for index, edge in enumerate(edges)}
    incident: dict[int, list[int]] = {vertex: [] for vertex in range(vertices)}
    for edge, variable in edge_variables.items():
        for vertex in edge:
            incident[vertex].append(variable)
    charges = [0] * vertices
    first = seed % vertices
    charges[first] = 1
    if label == "SAT":
        charges[(first + 1 + seed % (vertices - 1)) % vertices] ^= 1
    clauses = []
    for vertex in range(vertices):
        clauses.extend(_parity_clauses(sorted(incident[vertex]), charges[vertex]))
    return {"n_vars": len(edges), "clauses": clauses}


def _pigeonhole_formula(holes: int, label: str) -> JsonDict:
    pigeons = holes if label == "SAT" else holes + 1

    def variable(pigeon: int, hole: int) -> int:
        return pigeon * holes + hole + 1

    clauses = []
    for pigeon in range(pigeons):
        clauses.append([variable(pigeon, hole) for hole in range(holes)])
        for left, right in combinations(range(holes), 2):
            clauses.append([-variable(pigeon, left), -variable(pigeon, right)])
    for hole in range(holes):
        for left, right in combinations(range(pigeons), 2):
            clauses.append([-variable(left, hole), -variable(right, hole)])
    return {"n_vars": pigeons * holes, "clauses": clauses}


def _seeded_surface(cnf: Mapping[str, Any], key: str) -> JsonDict:
    """Use each registered seed while preserving the formula's exact truth."""

    rng = random.Random(_stable_integer(key))
    n_vars = int(cnf["n_vars"])
    targets = list(range(1, n_vars + 1))
    rng.shuffle(targets)
    mapping = {old: new for old, new in zip(range(1, n_vars + 1), targets)}
    clauses = [
        [mapping[abs(literal)] if literal > 0 else -mapping[abs(literal)] for literal in clause]
        for clause in cnf["clauses"]
    ]
    for clause in clauses:
        rng.shuffle(clause)
    rng.shuffle(clauses)
    return {"n_vars": n_vars, "clauses": clauses}


def generate_formula(family: str, size_bin: str, label: str, seed: int) -> JsonDict:
    """Generate one registered formula without consulting a solver result."""

    if family not in FAMILIES:
        raise ValueError(f"unknown_family:{family}")
    if size_bin not in SIZE_BINS:
        raise ValueError(f"unknown_size_bin:{size_bin}")
    if label not in LABELS:
        raise ValueError(f"unknown_label:{label}")
    if family == "pigeonhole_anchor":
        raw = _pigeonhole_formula(SIZE_BINS[size_bin]["pigeonhole_holes"], label)
    else:
        raw = _tseitin_formula(family, SIZE_BINS[size_bin]["tseitin_vertices"], label, seed)
    return _seeded_surface(raw, f"{family}:{size_bin}:{label}:{seed}:base")


def _literal_value(literal: int, assignment: Mapping[int, bool]) -> bool | None:
    value = assignment.get(abs(literal))
    if value is None:
        return None
    return value if literal > 0 else not value


def _propagate(
    clauses: Sequence[Sequence[int]], assignment: dict[int, bool], stats: JsonDict
) -> tuple[dict[int, bool], bool]:
    """Apply unit clauses until stable or until one exact conflict appears."""

    while True:
        changed = False
        for clause in clauses:
            values = [_literal_value(literal, assignment) for literal in clause]
            if True in values:
                continue
            unassigned = [literal for literal, value in zip(clause, values) if value is None]
            if not unassigned:
                stats["conflicts"] += 1
                return assignment, False
            if len(unassigned) == 1:
                literal = unassigned[0]
                variable = abs(literal)
                required = literal > 0
                if variable not in assignment:
                    assignment[variable] = required
                    stats["propagations"] += 1
                    changed = True
        if not changed:
            return assignment, True


def solve_cnf_exact(cnf: Mapping[str, Any]) -> JsonDict:
    """Solve a finite CNF with deterministic complete DPLL search."""

    clauses = [list(map(int, clause)) for clause in cnf["clauses"]]
    n_vars = int(cnf["n_vars"])
    stats: JsonDict = {"conflicts": 0, "decisions": 0, "propagations": 0}

    def search(current: dict[int, bool]) -> dict[int, bool] | None:
        assignment, consistent = _propagate(clauses, current, stats)
        if not consistent:
            return None
        if all(
            any(_literal_value(literal, assignment) is True for literal in clause)
            for clause in clauses
        ):
            return assignment
        variable = next(index for index in range(1, n_vars + 1) if index not in assignment)
        stats["decisions"] += 1
        for value in (False, True):
            candidate = search({**assignment, variable: value})
            if candidate is not None:
                return candidate
        return None

    model = search({})
    if model is None:
        return {"label": "UNSAT", "assignment": None, "stats": stats}
    complete_model = {index: model.get(index, False) for index in range(1, n_vars + 1)}
    return {"label": "SAT", "assignment": complete_model, "stats": stats}


def _clause_is_satisfied(clause: Sequence[int], assignment: Mapping[int, bool]) -> bool:
    return any(_literal_value(literal, assignment) is True for literal in clause)


def make_certificate(cnf: Mapping[str, Any], solved: Mapping[str, Any]) -> JsonDict:
    """Create SAT assignment evidence or a complete UNSAT assignment cover."""

    n_vars = int(cnf["n_vars"])
    clauses = cnf["clauses"]
    if solved["label"] == "SAT":
        return {
            "kind": "satisfying_assignment_v1",
            "assignment": {
                str(variable): bool(solved["assignment"][variable])
                for variable in range(1, n_vars + 1)
            },
        }
    falsified = []
    for mask in range(1 << n_vars):
        assignment = {
            variable: bool(mask & (1 << (variable - 1))) for variable in range(1, n_vars + 1)
        }
        falsified.append(
            next(
                index
                for index, clause in enumerate(clauses)
                if not _clause_is_satisfied(clause, assignment)
            )
        )
    return {
        "kind": "exhaustive_unsat_cover_v1",
        "assignment_count": 1 << n_vars,
        "falsified_clause_by_assignment": falsified,
    }


def check_certificate(cnf: Mapping[str, Any], label: str, certificate: Mapping[str, Any]) -> bool:
    """Check evidence without importing or trusting the DPLL solver state."""

    n_vars = int(cnf["n_vars"])
    clauses = cnf["clauses"]
    if label == "SAT":
        assignment_payload = certificate.get("assignment")
        if certificate.get("kind") != "satisfying_assignment_v1" or not isinstance(
            assignment_payload, Mapping
        ):
            return False
        if set(assignment_payload) != {str(index) for index in range(1, n_vars + 1)}:
            return False
        if not all(type(value) is bool for value in assignment_payload.values()):
            return False
        assignment = {int(key): value for key, value in assignment_payload.items()}
        return all(_clause_is_satisfied(clause, assignment) for clause in clauses)
    if label != "UNSAT" or certificate.get("kind") != "exhaustive_unsat_cover_v1":
        return False
    cover = certificate.get("falsified_clause_by_assignment")
    expected_count = 1 << n_vars
    if certificate.get("assignment_count") != expected_count or not isinstance(cover, list):
        return False
    if len(cover) != expected_count:
        return False
    for mask, clause_index in enumerate(cover):
        if type(clause_index) is not int or not 0 <= clause_index < len(clauses):
            return False
        assignment = {
            variable: bool(mask & (1 << (variable - 1))) for variable in range(1, n_vars + 1)
        }
        if _clause_is_satisfied(clauses[clause_index], assignment):
            return False
    return True


def canonical_clause_multiset(clauses: Sequence[Sequence[int]]) -> list[list[int]]:
    """Canonicalize clause order while retaining literal signs and repeats."""

    normalized = [
        sorted(map(int, clause), key=lambda value: (abs(value), value < 0)) for clause in clauses
    ]
    return sorted(normalized, key=lambda clause: (len(clause), clause))


def formula_graph_invariants(cnf: Mapping[str, Any]) -> JsonDict:
    """Summarize the variable-clause graph with symbol-independent values."""

    n_vars = int(cnf["n_vars"])
    clauses = cnf["clauses"]
    positive = Counter()
    negative = Counter()
    adjacency = {index: set() for index in range(1, n_vars + 1)}
    for clause in clauses:
        variables = sorted({abs(int(literal)) for literal in clause})
        for literal in clause:
            (positive if literal > 0 else negative)[abs(int(literal))] += 1
        for left, right in combinations(variables, 2):
            adjacency[left].add(right)
            adjacency[right].add(left)
    components = []
    unseen = set(adjacency)
    while unseen:
        pending = [min(unseen)]
        unseen.remove(pending[0])
        size = 0
        while pending:
            node = pending.pop()
            size += 1
            fresh = adjacency[node] & unseen
            unseen -= fresh
            pending.extend(sorted(fresh))
        components.append(size)
    return {
        "variable_count": n_vars,
        "clause_count": len(clauses),
        "literal_count": sum(len(clause) for clause in clauses),
        "clause_widths": [
            [width, count] for width, count in sorted(Counter(map(len, clauses)).items())
        ],
        "signed_occurrence_pairs": sorted(
            [positive[index], negative[index]] for index in range(1, n_vars + 1)
        ),
        "cooccurrence_degrees": sorted(len(adjacency[index]) for index in adjacency),
        "component_sizes": sorted(components),
    }


def _mate_mapping(n_vars: int, pair_id: str) -> dict[int, int]:
    rng = random.Random(_stable_integer(pair_id + ":relabel"))
    targets = list(range(1, n_vars + 1))
    rng.shuffle(targets)
    if targets == list(range(1, n_vars + 1)) and n_vars > 1:
        targets = targets[1:] + targets[:1]
    return dict(zip(range(1, n_vars + 1), targets))


def _apply_relabel(cnf: Mapping[str, Any], mapping: Mapping[int, int]) -> JsonDict:
    return {
        "n_vars": int(cnf["n_vars"]),
        "clauses": [
            [mapping[abs(literal)] if literal > 0 else -mapping[abs(literal)] for literal in clause]
            for clause in cnf["clauses"]
        ],
    }


def _inverse_mapped_clauses(
    clauses: Sequence[Sequence[int]], old_to_new: Mapping[str, int]
) -> list[list[int]]:
    inverse = {int(new): int(old) for old, new in old_to_new.items()}
    return [
        [inverse[abs(literal)] if literal > 0 else -inverse[abs(literal)] for literal in clause]
        for clause in clauses
    ]


def _make_row(
    registration: Mapping[str, Any],
    role: str,
    cnf: JsonDict,
    solved: JsonDict,
    certificate: JsonDict,
    mapping: Mapping[int, int],
) -> JsonDict:
    row: JsonDict = {
        "schema": ROW_SCHEMA,
        "row_id": f"{registration['pair_id']}-{role}",
        "pair_id": registration["pair_id"],
        "pair_role": role,
        "family": registration["family"],
        "size_bin": registration["size_bin"],
        "seed": registration["seed"],
        "split": registration["split"],
        "label": solved["label"],
        "cnf": cnf,
        "formula_sha256": sha256_json(cnf),
        "certificate": certificate,
        "certificate_sha256": sha256_json(certificate),
        "graph_invariants": formula_graph_invariants(cnf),
        "relabel_mapping": {"old_to_new": {str(key): value for key, value in mapping.items()}},
    }
    row["row_sha256"] = _row_hash(row)
    return row


def _checker_receipt(row: Mapping[str, Any], solved: Mapping[str, Any], passed: bool) -> JsonDict:
    receipt: JsonDict = {
        "row_id": row["row_id"],
        "solver_identity": SOLVER_IDENTITY,
        "solver_version": SOLVER_VERSION,
        "solver_exit_code": 0,
        "solver_sha256": SOLVER_SHA256,
        "solver_result_sha256": sha256_json(
            {"label": solved["label"], "assignment": solved["assignment"]}
        ),
        "checker_identity": CHECKER_IDENTITY,
        "checker_version": CHECKER_VERSION,
        "checker_exit_code": 0 if passed else 1,
        "checker_sha256": CHECKER_SHA256,
        "formula_sha256": row["formula_sha256"],
        "certificate_sha256": row["certificate_sha256"],
        "observed_label": row["label"],
        "passed": passed,
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    return receipt


def verify_relabel_pair(base: Mapping[str, Any], mate: Mapping[str, Any]) -> JsonDict:
    """Recompute all proof-preserving pair checks from raw row content."""

    mapping = mate["relabel_mapping"]["old_to_new"]
    mapped_back = _inverse_mapped_clauses(mate["cnf"]["clauses"], mapping)
    values = list(mapping.values())
    checks = {
        "pair_identity": base["pair_id"] == mate["pair_id"],
        "label_invariant": base["label"] == mate["label"],
        "base_certificate_valid": check_certificate(
            base["cnf"], base["label"], base["certificate"]
        ),
        "mate_certificate_valid": check_certificate(
            mate["cnf"], mate["label"], mate["certificate"]
        ),
        "clause_multiset_invariant": canonical_clause_multiset(base["cnf"]["clauses"])
        == canonical_clause_multiset(mapped_back),
        "graph_invariant": base["graph_invariants"] == mate["graph_invariants"],
        "mapping_bijective": sorted(values) == list(range(1, mate["cnf"]["n_vars"] + 1)),
        "split_invariant": base["split"] == mate["split"],
    }
    receipt: JsonDict = {
        "pair_id": base["pair_id"],
        "base_row_id": base["row_id"],
        "mate_row_id": mate["row_id"],
        "checks": checks,
        "base_formula_sha256": base["formula_sha256"],
        "mate_formula_sha256": mate["formula_sha256"],
        "passed": all(checks.values()),
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    return receipt


def generate_stream(
    timer: Clock = time.perf_counter,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Generate, solve, certify, and independently check every frozen row."""

    rows = []
    checker_receipts = []
    pair_receipts = []
    work_rows = []
    manifest = build_preregistered_manifest()
    for registration in manifest["base_rows"]:
        base_cnf = generate_formula(
            registration["family"],
            registration["size_bin"],
            registration["label"],
            registration["seed"],
        )
        n_vars = base_cnf["n_vars"]
        identity = {index: index for index in range(1, n_vars + 1)}
        mapping = _mate_mapping(n_vars, registration["pair_id"])
        pair = []
        for role, cnf, relabel in (
            ("base", base_cnf, identity),
            ("relabel", _apply_relabel(base_cnf, mapping), mapping),
        ):
            started = timer()
            solved = solve_cnf_exact(cnf)
            elapsed = max(0.0, timer() - started)
            certificate = make_certificate(cnf, solved)
            passed = solved["label"] == registration["label"] and check_certificate(
                cnf, solved["label"], certificate
            )
            row = _make_row(registration, role, cnf, solved, certificate, relabel)
            rows.append(row)
            pair.append(row)
            checker_receipts.append(_checker_receipt(row, solved, passed))
            work_rows.append(
                {
                    "row_id": row["row_id"],
                    "conflicts": solved["stats"]["conflicts"],
                    "decisions": solved["stats"]["decisions"],
                    "propagations": solved["stats"]["propagations"],
                    "wall_time_s": elapsed,
                }
            )
        pair_receipts.append(verify_relabel_pair(pair[0], pair[1]))
    return rows, checker_receipts, pair_receipts, work_rows


def stream_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash only canonical mathematical rows, never timing diagnostics."""

    return sha256_json(list(rows))


def _observed_family_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        family: {
            **{
                label: sum(row["family"] == family and row["label"] == label for row in rows)
                for label in LABELS
            },
            "total": sum(row["family"] == family for row in rows),
        }
        for family in FAMILIES
    }


def build_split_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build train, dev, and test receipts from row-level assignments."""

    splits: JsonDict = {}
    for split in ("train", "dev", "test"):
        selected = [row for row in rows if row.get("split") == split]
        splits[split] = {
            "families": sorted({row["family"] for row in selected}),
            "row_count": len(selected),
            "pair_ids": sorted({row["pair_id"] for row in selected}),
        }
    family_sets = [set(payload["families"]) for payload in splits.values()]
    pair_splits: dict[str, set[str]] = {}
    for row in rows:
        pair_splits.setdefault(row["pair_id"], set()).add(row.get("split", "missing"))
    pair_leak_count = sum(len(values) != 1 for values in pair_splits.values())
    complete = len(rows) == 72 and all(
        row.get("split") == SPLIT_BY_FAMILY.get(row.get("family")) for row in rows
    )
    return {
        "schema": SCHEMA + ".split_manifest",
        "splits": splits,
        "family_disjoint": all(
            left.isdisjoint(right)
            for index, left in enumerate(family_sets)
            for right in family_sets[index + 1 :]
        ),
        "pair_leak_count": pair_leak_count,
        "row_assignment_complete": complete,
    }


def _path_is_writable(path: Path) -> bool:
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return (
        parent.is_dir()
        and os.access(parent, os.W_OK)
        and (not path.exists() or os.access(path, os.W_OK))
    )


def collect_preconditions(output_path: Path) -> list[JsonDict]:
    """Check two separate exact paths, fixed seeds, and writable output."""

    independent = (
        callable(solve_cnf_exact)
        and callable(check_certificate)
        and SOLVER_IDENTITY != CHECKER_IDENTITY
        and SOLVER_SHA256 != CHECKER_SHA256
    )
    seeds_ready = len(SEEDS) >= 3 and len(set(SEEDS)) == len(SEEDS)
    writable = _path_is_writable(Path(output_path))
    return [
        {
            "check": "independent_exact_paths",
            "expected_value": True,
            "observed_value": independent,
            "passed": independent,
        },
        {
            "check": "deterministic_seed_registry",
            "expected_value": True,
            "observed_value": seeds_ready,
            "passed": seeds_ready,
        },
        {
            "check": "writable_result_path",
            "expected_value": True,
            "observed_value": writable,
            "passed": writable,
        },
    ]


def _field_principles() -> JsonDict:
    return {field: FIELD_PRINCIPLE_TEXT[field] for field in ARTIFACT_FIELDS}


def _first_failed(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    for row in checks:
        if not row.get("passed"):
            return {
                "failed_check": row.get("check"),
                "expected_value": row.get("expected_value", True),
                "observed_value": row.get("observed_value"),
            }
    return {"failed_check": None, "expected_value": True, "observed_value": True}


def _blocked_artifact(
    preconditions: list[JsonDict], manifest: JsonDict, duration_s: float
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "title": "Hardness-controlled exact certificate stream",
        "status": "complete_blocked",
        "preconditions_checked": preconditions,
        "preregistered_manifest": manifest,
        "field_principles": _field_principles(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": list(SEEDS),
        "reproducibility_checksum": stream_checksum([]),
        "rows": [],
        "family_counts": manifest["family_counts"],
        "split_manifest": build_split_manifest([]),
        "certificate_checker_receipts": [],
        "relabel_pair_receipts": [],
        "solver_work_metadata": [],
        "deterministic_replay_receipt": {"matched": False, "reason": "precondition_block"},
        "verification_commands": [],
        "hardness_stream_ready": False,
        "gate_check_summary": _first_failed(preconditions),
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_exact_checker: an exact-path precondition failed",
    }
    return artifact


def build_artifact(output_path: Path, duration_s: float) -> JsonDict:
    """Build a complete artifact and recompute the exact downstream gate."""

    manifest = build_preregistered_manifest()
    preconditions = collect_preconditions(output_path)
    if not all(row["passed"] for row in preconditions):
        return _blocked_artifact(preconditions, manifest, duration_s)
    rows, checker_receipts, pair_receipts, work_rows = generate_stream()
    replay_rows, _, _, _ = generate_stream()
    checksum = stream_checksum(rows)
    replay_checksum = stream_checksum(replay_rows)
    family_counts = _observed_family_counts(rows)
    split_manifest = build_split_manifest(rows)
    verification_commands = [
        {
            "command": "internal:validate_all_certificates",
            "exit_code": 0 if all(row["passed"] for row in checker_receipts) else 1,
        },
        {
            "command": "internal:validate_all_relabel_pairs",
            "exit_code": 0 if all(row["passed"] for row in pair_receipts) else 1,
        },
        {
            "command": "internal:validate_family_disjoint_split",
            "exit_code": 0
            if split_manifest["family_disjoint"]
            and split_manifest["pair_leak_count"] == 0
            and split_manifest["row_assignment_complete"]
            else 1,
        },
        {
            "command": "internal:deterministic_stream_replay",
            "exit_code": 0 if checksum == replay_checksum else 1,
        },
    ]
    checks = [
        {
            "check": "preconditions",
            "expected_value": True,
            "observed_value": all(row["passed"] for row in preconditions),
            "passed": all(row["passed"] for row in preconditions),
        },
        {
            "check": "row_count",
            "expected_value": 72,
            "observed_value": len(rows),
            "passed": len(rows) == 72,
        },
        {
            "check": "family_counts",
            "expected_value": manifest["family_counts"],
            "observed_value": family_counts,
            "passed": family_counts == manifest["family_counts"],
        },
        {
            "check": "certificates",
            "expected_value": True,
            "observed_value": all(row["passed"] for row in checker_receipts),
            "passed": all(row["passed"] for row in checker_receipts),
        },
        {
            "check": "relabel_pairs",
            "expected_value": 36,
            "observed_value": sum(row["passed"] for row in pair_receipts),
            "passed": len(pair_receipts) == 36 and all(row["passed"] for row in pair_receipts),
        },
        {
            "check": "split_isolation",
            "expected_value": True,
            "observed_value": split_manifest["family_disjoint"]
            and split_manifest["pair_leak_count"] == 0
            and split_manifest["row_assignment_complete"],
            "passed": split_manifest["family_disjoint"]
            and split_manifest["pair_leak_count"] == 0
            and split_manifest["row_assignment_complete"],
        },
        {
            "check": "deterministic_replay",
            "expected_value": checksum,
            "observed_value": replay_checksum,
            "passed": checksum == replay_checksum,
        },
        {
            "check": "verification_commands",
            "expected_value": 0,
            "observed_value": max(row["exit_code"] for row in verification_commands),
            "passed": all(row["exit_code"] == 0 for row in verification_commands),
        },
    ]
    ready = all(row["passed"] for row in checks)
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "title": "Hardness-controlled exact certificate stream",
        "status": "complete_ready" if ready else "complete_partial",
        "preconditions_checked": preconditions,
        "preregistered_manifest": manifest,
        "field_principles": _field_principles(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "random_seed": list(SEEDS),
        "reproducibility_checksum": checksum,
        "rows": rows,
        "family_counts": family_counts,
        "split_manifest": split_manifest,
        "certificate_checker_receipts": checker_receipts,
        "relabel_pair_receipts": pair_receipts,
        "solver_work_metadata": work_rows,
        "deterministic_replay_receipt": {
            "first_stream_sha256": checksum,
            "replay_stream_sha256": replay_checksum,
            "matched": checksum == replay_checksum,
        },
        "verification_commands": verification_commands,
        "hardness_stream_ready": ready,
        "gate_check_summary": _first_failed(checks),
        "verdict_class": "positive" if ready else "partial",
        "honest_verdict": (
            "complete_positive: all 72 exact certificate rows and 36 relabel pairs are ready"
            if ready
            else "complete_partial: one or more exact stream gates failed"
        ),
    }
    return artifact


def _ready_recomputation(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("rows", [])
    if len(rows) != 72:
        return False
    manifest = artifact.get("preregistered_manifest", {})
    if artifact.get("family_counts") != manifest.get("family_counts"):
        return False
    if _observed_family_counts(rows) != artifact.get("family_counts"):
        return False
    if any(
        not receipt.get("passed") for receipt in artifact.get("certificate_checker_receipts", [])
    ):
        return False
    if len(artifact.get("certificate_checker_receipts", [])) != 72:
        return False
    if len(artifact.get("relabel_pair_receipts", [])) != 36 or any(
        not receipt.get("passed") for receipt in artifact.get("relabel_pair_receipts", [])
    ):
        return False
    split = build_split_manifest(rows)
    if split != artifact.get("split_manifest"):
        return False
    if (
        not split["family_disjoint"]
        or split["pair_leak_count"]
        or not split["row_assignment_complete"]
    ):
        return False
    replay = artifact.get("deterministic_replay_receipt", {})
    if not replay.get("matched") or replay.get("first_stream_sha256") != stream_checksum(rows):
        return False
    return all(row.get("exit_code") == 0 for row in artifact.get("verification_commands", []))


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Reject missing, corrupt, leaked, or falsely ready artifact content."""

    missing = sorted(set(ARTIFACT_FIELDS) - set(artifact))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    errors = []
    if set(artifact["field_principles"]) != set(artifact):
        errors.append("field_principles_missing")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["random_seed"] != list(SEEDS):
        errors.append("random_seed_mismatch")
    if artifact["duration_s"] <= 0:
        errors.append("duration_not_positive")
    rows = artifact["rows"]
    if artifact["reproducibility_checksum"] != stream_checksum(rows):
        errors.append("reproducibility_checksum_mismatch")
    if artifact["verdict_class"] == "blocked":
        if rows or artifact["hardness_stream_ready"]:
            errors.append("blocked_rows_or_readiness_present")
        if artifact["gate_check_summary"].get("failed_check") is None:
            errors.append("blocked_gate_summary_missing")
        if not artifact["honest_verdict"].startswith("complete_blocked_exact_checker"):
            errors.append("blocked_verdict_prefix_invalid")
        return errors
    row_by_id = {row.get("row_id"): row for row in rows}
    row_invalid = False
    for row in rows:
        valid = (
            row.get("row_sha256") == _row_hash(row)
            and row.get("formula_sha256") == sha256_json(row.get("cnf"))
            and row.get("certificate_sha256") == sha256_json(row.get("certificate"))
            and row.get("graph_invariants") == formula_graph_invariants(row.get("cnf"))
            and check_certificate(row.get("cnf"), row.get("label"), row.get("certificate"))
        )
        row_invalid = row_invalid or not valid
    if row_invalid:
        errors.append("row_certificate_invalid")
    receipts = artifact["certificate_checker_receipts"]
    receipt_invalid = len(receipts) != 72 or any(
        receipt.get("receipt_sha256") != _receipt_hash(receipt)
        or receipt.get("row_id") not in row_by_id
        or receipt.get("formula_sha256")
        != row_by_id.get(receipt.get("row_id"), {}).get("formula_sha256")
        or receipt.get("certificate_sha256")
        != row_by_id.get(receipt.get("row_id"), {}).get("certificate_sha256")
        or receipt.get("solver_exit_code") != 0
        or receipt.get("checker_exit_code") != 0
        or not receipt.get("passed")
        for receipt in receipts
    )
    if receipt_invalid:
        errors.append("checker_receipt_invalid")
    pair_invalid = len(artifact["relabel_pair_receipts"]) != 36
    for receipt in artifact["relabel_pair_receipts"]:
        base = row_by_id.get(receipt.get("base_row_id"))
        mate = row_by_id.get(receipt.get("mate_row_id"))
        if base is None or mate is None or receipt != verify_relabel_pair(base, mate):
            pair_invalid = True
    if pair_invalid:
        errors.append("pair_receipt_invalid")
    if build_split_manifest(rows) != artifact["split_manifest"]:
        errors.append("split_manifest_invalid")
    recomputed = _ready_recomputation(artifact)
    if artifact["hardness_stream_ready"] != recomputed:
        errors.append("readiness_recomputation_mismatch")
    if artifact["hardness_stream_ready"] and artifact["verdict_class"] != "positive":
        errors.append("ready_verdict_class_invalid")
    if artifact["hardness_stream_ready"] and not artifact["honest_verdict"].startswith(
        "complete_positive"
    ):
        errors.append("ready_verdict_prefix_invalid")
    return errors


def write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the result only after a complete JSON file reaches disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(
    output_path: Path = REPO_ROOT / RESULT_PATH,
    *,
    precondition_probe: PreconditionProbe = collect_preconditions,
    clock: Clock = time.monotonic,
) -> JsonDict:
    """Run once, measure with a monotonic clock, validate, and write."""

    output_path = Path(output_path)
    started = clock()
    preconditions = precondition_probe(output_path)
    manifest = build_preregistered_manifest()
    if not all(row.get("passed") for row in preconditions):
        duration = max(clock() - started, 1e-9)
        artifact = _blocked_artifact(preconditions, manifest, duration)
    else:
        artifact = build_artifact(output_path, 1e-9)
        artifact["preconditions_checked"] = preconditions
        artifact["duration_s"] = max(clock() - started, 1e-9)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid_exp6744_artifact:" + ",".join(errors))
    write_json_atomic(output_path, artifact)
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_PATH)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate:
        try:
            payload = json.loads(args.output.read_text(encoding="utf-8"))
            return int(bool(validate_artifact(payload)))
        except (OSError, json.JSONDecodeError, TypeError):
            return 1
    run(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - covered through the public main function.
    raise SystemExit(main())
