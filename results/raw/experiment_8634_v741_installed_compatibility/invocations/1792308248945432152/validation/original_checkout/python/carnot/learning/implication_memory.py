"""Retain source-checkable implications from public 2-CNF formulas.

The exact solver remains the truth source. Memory can only reuse a path made
from original clause edges, and it can only reject an assumption conflict.
It never certifies that a formula is satisfiable.

Spec refs: REQ-CL-7370 and SCENARIO-CL-7370-*.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


MAX_DISCOVERED_PATHS = 8
MAX_RETAINED_PATHS = 128
MAX_STATE_BYTES = 65_536


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _literal_key(literal: int) -> tuple[int, bool]:
    return abs(literal), literal < 0


@dataclass(frozen=True, order=True)
class SourceClause:
    """Bind one canonical public clause to a stable non-negative identifier."""

    clause_id: int
    literals: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {"clause_id": self.clause_id, "literals": list(self.literals)}


@dataclass(frozen=True, order=True)
class ImplicationEdge:
    """Name one implication and the original clause that creates it."""

    from_literal: int
    to_literal: int
    source_clause_id: int

    def to_dict(self) -> dict[str, int]:
        return {
            "from_literal": self.from_literal,
            "to_literal": self.to_literal,
            "source_clause_id": self.source_clause_id,
        }


@dataclass(frozen=True)
class FormulaVersion:
    """Keep canonical clauses, identity, and implication edges immutable."""

    version: str
    n_vars: int
    clauses: tuple[SourceClause, ...]
    source_hash: str
    implication_edges: frozenset[ImplicationEdge]

    @classmethod
    def from_clauses(
        cls,
        version: str,
        n_vars: int,
        clauses: Iterable[Sequence[int]],
    ) -> FormulaVersion:
        if not version:
            raise ValueError("formula_version_empty")
        if type(n_vars) is not int or n_vars < 1:
            raise ValueError("n_vars_invalid")
        normalized: list[tuple[int, int]] = []
        for raw in clauses:
            if len(raw) != 2:
                raise ValueError("clause_not_2cnf")
            left, right = raw
            if any(
                type(literal) is not int or literal == 0 or abs(literal) > n_vars
                for literal in (left, right)
            ):
                raise ValueError("clause_literal_invalid")
            normalized.append(tuple(sorted((left, right), key=_literal_key)))
        normalized.sort(key=lambda pair: (_literal_key(pair[0]), _literal_key(pair[1])))
        sources = tuple(SourceClause(index, pair) for index, pair in enumerate(normalized))
        identity = {
            "version": version,
            "n_vars": n_vars,
            "clauses": [clause.to_dict() for clause in sources],
        }
        edges = frozenset(
            edge
            for clause in sources
            for edge in (
                ImplicationEdge(-clause.literals[0], clause.literals[1], clause.clause_id),
                ImplicationEdge(-clause.literals[1], clause.literals[0], clause.clause_id),
            )
        )
        return cls(version, n_vars, sources, _sha256(identity), edges)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_vars": self.n_vars,
            "clauses": [clause.to_dict() for clause in self.clauses],
            "source_hash": self.source_hash,
        }

    def _validate_assumptions(self, assumptions: Sequence[int]) -> tuple[int, ...]:
        values = tuple(assumptions)
        if any(
            type(literal) is not int or literal == 0 or abs(literal) > self.n_vars
            for literal in values
        ):
            raise ValueError("assumption_literal_invalid")
        return values

    def _is_satisfiable(self, assumptions: Sequence[int]) -> bool:
        assumptions = self._validate_assumptions(assumptions)
        vertices = tuple(range(1, self.n_vars + 1)) + tuple(range(-1, -self.n_vars - 1, -1))
        graph = {literal: [] for literal in vertices}
        reverse = {literal: [] for literal in vertices}
        edges = [(edge.from_literal, edge.to_literal) for edge in self.implication_edges]
        edges.extend((-literal, literal) for literal in assumptions)
        for source, target in edges:
            graph[source].append(target)
            reverse[target].append(source)
        visited: set[int] = set()
        order: list[int] = []

        def visit(vertex: int) -> None:
            visited.add(vertex)
            for target in graph[vertex]:
                if target not in visited:
                    visit(target)
            order.append(vertex)

        for vertex in vertices:
            if vertex not in visited:
                visit(vertex)
        component: dict[int, int] = {}

        def assign(vertex: int, component_id: int) -> None:
            component[vertex] = component_id
            for target in reverse[vertex]:
                if target not in component:
                    assign(target, component_id)

        for vertex in reversed(order):
            if vertex not in component:
                assign(vertex, len(component))
        return all(
            component[variable] != component[-variable] for variable in range(1, self.n_vars + 1)
        )

    def solve(self, assumptions: Sequence[int]) -> tuple[bool, dict[int, bool] | None]:
        """Return an exact result and construct a source-verified assignment."""

        assumptions = self._validate_assumptions(assumptions)
        if not self._is_satisfiable(assumptions):
            return False, None
        chosen = list(assumptions)
        fixed = {abs(literal): literal > 0 for literal in assumptions}
        assignment: dict[int, bool] = {}
        for variable in range(1, self.n_vars + 1):
            if variable in fixed:
                assignment[variable] = fixed[variable]
                continue
            candidate = (*chosen, -variable)
            value = not self._is_satisfiable(candidate)
            assignment[variable] = value
            chosen.append(variable if value else -variable)
        if not self.verify_assignment(
            assignment, assumptions
        ):  # pragma: no cover - constructive invariant.
            raise RuntimeError("exact_solver_assignment_verification_failed")
        return True, assignment

    def verify_assignment(
        self, assignment: Mapping[int, bool], assumptions: Sequence[int] = ()
    ) -> bool:
        if set(assignment) != set(range(1, self.n_vars + 1)):
            return False

        def value(literal: int) -> bool:
            selected = bool(assignment[abs(literal)])
            return selected if literal > 0 else not selected

        return all(value(literal) for literal in assumptions) and all(
            value(left) or value(right)
            for left, right in (clause.literals for clause in self.clauses)
        )

    def find_path(self, antecedent: int, consequent: int) -> tuple[ImplicationEdge, ...] | None:
        """Find one short path that uses only original source edges."""

        self._validate_assumptions((antecedent, consequent))
        queue = deque([antecedent])
        prior: dict[int, ImplicationEdge | None] = {antecedent: None}
        outgoing: dict[int, list[ImplicationEdge]] = {}
        for edge in self.implication_edges:
            outgoing.setdefault(edge.from_literal, []).append(edge)
        for edges in outgoing.values():
            edges.sort(key=lambda edge: (edge.to_literal, edge.source_clause_id))
        while queue:
            current = queue.popleft()
            if current == consequent:
                break
            for edge in outgoing.get(current, []):
                if edge.to_literal not in prior:
                    prior[edge.to_literal] = edge
                    queue.append(edge.to_literal)
        if consequent not in prior or antecedent == consequent:
            return None
        path: list[ImplicationEdge] = []
        current = consequent
        while current != antecedent:
            edge = prior[current]
            if edge is None:  # pragma: no cover - only the seeded root has no predecessor.
                return None
            path.append(edge)
            current = edge.from_literal
        path.reverse()
        return tuple(path)


@dataclass(frozen=True)
class ProofPath:
    """Represent one implication with every source edge made explicit."""

    formula_version: str
    source_hash: str
    antecedent: int
    consequent: int
    edges: tuple[ImplicationEdge, ...]

    @property
    def path_id(self) -> str:
        return _sha256(self._content())

    def _content(self) -> dict[str, Any]:
        return {
            "formula_version": self.formula_version,
            "source_hash": self.source_hash,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "edges": [edge.to_dict() for edge in self.edges],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "path_id": self.path_id}


def validate_proof(formula: FormulaVersion, proof: ProofPath) -> list[str]:
    """Check a path from formula bytes without calling any learner."""

    errors: list[str] = []
    if proof.formula_version != formula.version:
        errors.append("formula_version_mismatch")
    if proof.source_hash != formula.source_hash:
        errors.append("source_hash_mismatch")
    if any(
        type(literal) is not int or literal == 0 or abs(literal) > formula.n_vars
        for literal in (proof.antecedent, proof.consequent)
    ):
        errors.append("endpoint_invalid")
    if not proof.edges:
        errors.append("path_empty")
    if len(proof.edges) > 2 * formula.n_vars:
        errors.append("path_edge_cap_exceeded")
    if proof.edges:
        if proof.edges[0].from_literal != proof.antecedent:
            errors.append("antecedent_endpoint_mismatch")
        if proof.edges[-1].to_literal != proof.consequent:
            errors.append("consequent_endpoint_mismatch")
    vertices = [proof.antecedent]
    for index, edge in enumerate(proof.edges):
        if edge.source_clause_id < 0:
            errors.append("negative_source_clause_id")
        if edge not in formula.implication_edges:
            errors.append("edge_not_in_source_formula")
        if index and proof.edges[index - 1].to_literal != edge.from_literal:
            errors.append("path_literal_omitted")
        vertices.append(edge.to_literal)
    if len(vertices) != len(set(vertices)):
        errors.append("path_cycle")
    return list(dict.fromkeys(errors))


def proof_from_dict(payload: Mapping[str, Any], formula: FormulaVersion) -> ProofPath:
    """Parse a closed proof schema and fail on extra label-like authority."""

    expected = {
        "formula_version",
        "source_hash",
        "antecedent",
        "consequent",
        "edges",
        "path_id",
    }
    if set(payload) != expected:
        raise ValueError("proof_schema_mismatch")
    raw_edges = payload.get("edges")
    if not isinstance(raw_edges, list):
        raise ValueError("proof_edges_not_list")
    edges: list[ImplicationEdge] = []
    for raw in raw_edges:
        if not isinstance(raw, Mapping) or set(raw) != {
            "from_literal",
            "to_literal",
            "source_clause_id",
        }:
            raise ValueError("proof_edge_schema_mismatch")
        values = (raw["from_literal"], raw["to_literal"], raw["source_clause_id"])
        if any(type(value) is not int for value in values):
            raise ValueError("proof_edge_type_invalid")
        edges.append(ImplicationEdge(*values))
    if type(payload.get("antecedent")) is not int or type(payload.get("consequent")) is not int:
        raise ValueError("proof_endpoint_type_invalid")
    proof = ProofPath(
        formula_version=str(payload.get("formula_version")),
        source_hash=str(payload.get("source_hash")),
        antecedent=payload["antecedent"],
        consequent=payload["consequent"],
        edges=tuple(edges),
    )
    errors = validate_proof(formula, proof)
    if payload.get("path_id") != proof.path_id:
        errors.append("path_id_mismatch")
    if errors:
        raise ValueError(",".join(errors))
    return proof


@dataclass(frozen=True)
class ProofMemory:
    """Hold a bounded committed snapshot for exactly one formula version."""

    formula: FormulaVersion
    paths: tuple[ProofPath, ...] = ()

    @classmethod
    def empty(cls, formula: FormulaVersion) -> ProofMemory:
        return cls(formula, ())

    @property
    def sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "carnot.implication_memory.v1",
            "formula_version": self.formula.version,
            "source_hash": self.formula.source_hash,
            "paths": [path.to_dict() for path in self.paths],
        }

    def to_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def commit(self, candidates: Sequence[ProofPath]) -> ProofMemory:
        """Return a new snapshot after validation and both storage caps."""

        retained = list(self.paths)
        known = {path.path_id for path in retained}
        for candidate in candidates[:MAX_DISCOVERED_PATHS]:
            errors = validate_proof(self.formula, candidate)
            if errors:
                continue
            if candidate.path_id in known or len(retained) >= MAX_RETAINED_PATHS:
                continue
            trial = ProofMemory(self.formula, tuple((*retained, candidate)))
            if len(trial.to_bytes()) > MAX_STATE_BYTES:
                break
            retained.append(candidate)
            known.add(candidate.path_id)
        return ProofMemory(self.formula, tuple(retained))

    def without(self, path_id: str) -> ProofMemory:
        return ProofMemory(
            self.formula,
            tuple(path for path in self.paths if path.path_id != path_id),
        )

    def save(self, path: Path) -> None:
        """Write one complete snapshot before replacing the prior file."""

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.to_bytes() + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: Path, formula: FormulaVersion) -> ProofMemory:
        """Restore exact committed bytes or return an invalidated empty state."""

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping) or set(payload) != {
                "schema",
                "formula_version",
                "source_hash",
                "paths",
            }:
                return cls.empty(formula)
            if (
                payload["schema"] != "carnot.implication_memory.v1"
                or payload["formula_version"] != formula.version
                or payload["source_hash"] != formula.source_hash
                or not isinstance(payload["paths"], list)
            ):
                return cls.empty(formula)
            paths = tuple(proof_from_dict(item, formula) for item in payload["paths"])
            restored = cls(formula, paths)
            if len(paths) > MAX_RETAINED_PATHS or len(restored.to_bytes()) > MAX_STATE_BYTES:
                return cls.empty(formula)
            return restored
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return cls.empty(formula)


@dataclass(frozen=True)
class QueryResult:
    """Separate the entry snapshot decision from its later committed update."""

    decision: str
    used_exact_solver: bool
    assignment: dict[int, bool] | None
    proof_path_id: str | None
    discovered_paths: tuple[ProofPath, ...]
    snapshot_sha256: str
    committed_memory: ProofMemory
    cost_ns: dict[str, int]


def _discover_paths(formula: FormulaVersion, assumptions: Sequence[int]) -> tuple[ProofPath, ...]:
    paths: list[ProofPath] = []
    known: set[str] = set()
    checked_conflicts: set[frozenset[int]] = set()
    for antecedent in assumptions:
        for conflicting in assumptions:
            conflict_key = frozenset((antecedent, conflicting))
            if conflict_key in checked_conflicts:
                continue
            checked_conflicts.add(conflict_key)
            consequent = -conflicting
            path_edges = formula.find_path(antecedent, consequent)
            if path_edges is None:
                continue
            proof = ProofPath(
                formula.version,
                formula.source_hash,
                antecedent,
                consequent,
                path_edges,
            )
            if proof.path_id not in known and not validate_proof(formula, proof):
                paths.append(proof)
                known.add(proof.path_id)
            if len(paths) >= MAX_DISCOVERED_PATHS:
                return tuple(paths)
    return tuple(paths)


def execute_query(memory: ProofMemory, assumptions: Sequence[int]) -> QueryResult:
    """Use prior proofs for rejection, otherwise call the exact formula solver."""

    assumptions = memory.formula._validate_assumptions(assumptions)
    assumption_set = set(assumptions)
    checked_started = time.perf_counter_ns()
    for proof in memory.paths:
        if validate_proof(memory.formula, proof):
            continue
        if proof.antecedent in assumption_set and -proof.consequent in assumption_set:
            proof_check_ns = time.perf_counter_ns() - checked_started
            storage_started = time.perf_counter_ns()
            memory.to_bytes()
            storage_ns = time.perf_counter_ns() - storage_started
            return QueryResult(
                "reject",
                False,
                None,
                proof.path_id,
                (),
                memory.sha256,
                memory,
                {
                    "proof_discovery": 0,
                    "proof_checking": proof_check_ns,
                    "updates": 0,
                    "storage": storage_ns,
                    "exact_solve": 0,
                },
            )
    proof_check_ns = time.perf_counter_ns() - checked_started
    solve_started = time.perf_counter_ns()
    satisfiable, assignment = memory.formula.solve(assumptions)
    exact_solve_ns = time.perf_counter_ns() - solve_started
    discovery_started = time.perf_counter_ns()
    discovered = () if satisfiable else _discover_paths(memory.formula, assumptions)
    proof_discovery_ns = time.perf_counter_ns() - discovery_started
    update_started = time.perf_counter_ns()
    committed = memory.commit(discovered)
    update_ns = time.perf_counter_ns() - update_started
    storage_started = time.perf_counter_ns()
    committed.to_bytes()
    storage_ns = time.perf_counter_ns() - storage_started
    return QueryResult(
        "satisfiable" if satisfiable else "unsatisfiable",
        True,
        assignment,
        None,
        discovered,
        memory.sha256,
        committed,
        {
            "proof_discovery": proof_discovery_ns,
            "proof_checking": proof_check_ns,
            "updates": update_ns,
            "storage": storage_ns,
            "exact_solve": exact_solve_ns,
        },
    )
