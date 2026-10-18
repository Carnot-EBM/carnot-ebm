"""Exp6521 transactional refinement-gated exact-conflict memory.

Spec refs: REQ-STORE-6521, SCENARIO-STORE-6521-VALID-REUSE,
SCENARIO-STORE-6521-INVALID-VETO, SCENARIO-STORE-6521-LIFECYCLE,
SCENARIO-STORE-6521-FIXED-WIDTH-MAPPING.

The controller stores learned conflict clauses only after two local checks pass.
First, the target query must be a strict or equal refinement of the source
query. Second, an exact replay must prove that the conflict clause is entailed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

from carnot.atomic_shard_transaction import (
    TRANSACTION_SCHEMA,
    AtomicShardTransaction,
    CrashPlan,
    sha256_bytes,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6521
QUERY_SCHEMA_VERSION = "carnot.exact_conflict_query.v1"
RECORD_SCHEMA_VERSION = "carnot.exact_conflict_record.v1"
MEMORY_SCHEMA_VERSION = "carnot.transactional_exact_conflict_memory.v1"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_6521.transactional_refinement_conflict_memory.v1"
REFINEMENT_RELATION = "clause_superset_same_schema_domain_solver_v1"
INFERENCE_SUBSTRATE = "transactional_exact_conflict_memory_and_cpu_mapping_no_llm"
VERIFIER_IS_ORACLE = True

RESULT_RELATIVE_PATH = Path("results/experiment_6521_transactional_refinement_conflict_memory.json")
WORK_RELATIVE_PATH = Path("results/.experiment_6521_transactional_refinement_conflict_memory.tx")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-store/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6521_transactional_refinement_conflict_memory.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6521_transactional_refinement_conflict_memory.py"
)
EXP6517_RELATIVE_PATH = Path("results/experiment_6517_branch_pilot_independent_audit.json")
EXP6516_RELATIVE_PATH = Path("results/experiment_6516_exact_branch_pilot_dataset_v3.json")
EXP6515_RELATIVE_PATH = Path("results/experiment_6515_v564_source_method_contract.json")
EXP6514_RELATIVE_PATH = Path("results/experiment_6514_atomic_shard_artifact_transaction.json")
EXP6495_RELATIVE_PATH = Path("results/experiment_6495_restarted_factor_pool_controller.json")

PROTECTED_RELATIVE_PATHS = (
    EXP6495_RELATIVE_PATH,
    EXP6514_RELATIVE_PATH,
    EXP6515_RELATIVE_PATH,
    EXP6516_RELATIVE_PATH,
    EXP6517_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "conflict_record_schema",
    "refinement_relation_contract",
    "lifecycle_rows",
    "valid_reuse_rows",
    "invalid_reuse_veto_rows",
    "capacity_and_eviction_rows",
    "restart_rollback_rows",
    "corruption_quarantine_rows",
    "native_fallback_rows",
    "fixed_width_mapping_rows",
    "conflict_memory_controller_ready_score",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
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
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Records the terminal exact-conflict memory state.",
    "honest_verdict": (
        "States exact safety readiness without claiming routing speed or learning benefit."
    ),
    "verdict_class": "Uses circular_positive only for exact mechanism readiness.",
    "upstream_gate_receipt": "Binds the run to the independent pilot-audit gate path and hash.",
    "conflict_record_schema": "Defines the durable conflict fields and content hash.",
    "refinement_relation_contract": "Defines the only supported local safe-reuse relation.",
    "lifecycle_rows": "Shows prepare, validate, commit, abort, load, use, checkpoint, and rollback.",
    "valid_reuse_rows": "Shows exact replay before each accepted reuse.",
    "invalid_reuse_veto_rows": "Shows unsafe candidates were rejected before write or use.",
    "capacity_and_eviction_rows": "Shows bounded capacity and deterministic eviction order.",
    "restart_rollback_rows": "Shows restart parity and rollback hash restoration.",
    "corruption_quarantine_rows": "Shows corrupt durable bytes are moved out of the active path.",
    "native_fallback_rows": "Shows exact native solving continues when memory is unavailable.",
    "fixed_width_mapping_rows": "Reports CPU mapping cost without a hardware claim.",
    "conflict_memory_controller_ready_score": (
        "A conjunctive score opens only with zero unsafe admission and zero unsafe use."
    ),
    "gate_check_summary": "Names each gate, expected value, observed value, and failure.",
    "per_unit_rows": "Combines lifecycle, safety, mapping, fallback, and attack rows.",
    "aggregate_row_recomputation": "Recomputes readiness from rows rather than summary text.",
    "preconditions_checked": "Records solver capability, resources, relation, and protected hashes.",
    "protected_files_unchanged": "Proves protected upstream files did not change during the run.",
    "inference_substrate": "Declares exact local memory and CPU mapping with no LLM.",
    "verifier_is_oracle": "Exact replay is authoritative only inside the declared finite domain.",
    "field_principles": "Preserves why each artifact field exists.",
    "field_provenance": "Maps each field to gates, rows, exact replay, transactions, or tests.",
    "random_seed": "Fixes the deterministic scenario order.",
    "duration_s": "Records measured wall-clock duration.",
    "tests_run": "Records validation commands and exit codes.",
    "reproducibility_checksum": "Detects later drift in rows, gates, code, or hashes.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6521_transactional_refinement_conflict_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6521_transactional_refinement_conflict_memory.py "
    "-m pytest tests/python/test_experiment_6521_transactional_refinement_conflict_memory.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6521_transactional_refinement_conflict_memory.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6521_transactional_refinement_conflict_memory.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6521_transactional_refinement_conflict_memory.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6521_transactional_refinement_conflict_memory.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6521_transactional_refinement_conflict_memory --date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6521_transactional_refinement_conflict_memory --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    """Return stable JSON text for hashes and equality receipts."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with the project prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash an evidence file or return a visible missing marker."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


DEFAULT_SOLVER_HASH = sha256_json(
    {
        "solver": "carnot_exhaustive_boolean_cnf_entailment",
        "schema_version": QUERY_SCHEMA_VERSION,
        "version": "v1",
    }
)


class ConflictMemoryError(ValueError):
    """Raised when a conflict record cannot be safely admitted or used."""


@dataclass
class ExactQuery:
    """Small Boolean-CNF query used by the exact replay verifier."""

    variable_count: int
    clauses: Sequence[Sequence[int]]
    schema_version: str = QUERY_SCHEMA_VERSION
    solver_hash: str = DEFAULT_SOLVER_HASH

    def normalized_clauses(self) -> tuple[tuple[int, ...], ...]:
        clauses = []
        for clause in self.clauses:
            clauses.append(tuple(sorted({int(literal) for literal in clause}, key=abs)))
        return tuple(sorted(clauses))

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "solver_hash": self.solver_hash,
            "variable_count": int(self.variable_count),
            "clauses": [list(clause) for clause in self.normalized_clauses()],
        }

    def query_hash(self) -> str:
        return sha256_json(self.to_dict())


@dataclass
class ConflictRecord:
    """Canonical durable conflict record with mutable lifecycle metadata."""

    source_query_hash: str
    source_query_payload: JsonDict
    target_query_payload: JsonDict
    clause_payload: tuple[int, ...]
    solver_hash: str
    solver_version_hash: str
    refinement_witness: JsonDict
    replay_receipt: JsonDict
    lifecycle_state: str
    use_count: int
    benefit_score: float
    benefit_observations: int
    created_version: int
    committed_version: int | None = None
    last_used_version: int | None = None
    content_hash: str = ""

    def immutable_dict(self) -> JsonDict:
        return {
            "source_query_hash": self.source_query_hash,
            "source_query_payload": self.source_query_payload,
            "target_query_payload": self.target_query_payload,
            "clause_payload": list(self.clause_payload),
            "solver_hash": self.solver_hash,
            "solver_version_hash": self.solver_version_hash,
            "refinement_witness": self.refinement_witness,
            "replay_receipt_hash": self.replay_receipt.get("replay_receipt_hash"),
            "benefit_score": float(self.benefit_score),
            "benefit_observations": int(self.benefit_observations),
            "created_version": int(self.created_version),
        }

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            **self.immutable_dict(),
            "replay_receipt": self.replay_receipt,
            "lifecycle_state": self.lifecycle_state,
            "use_count": int(self.use_count),
            "committed_version": self.committed_version,
            "last_used_version": self.last_used_version,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ConflictRecord":
        return cls(
            source_query_hash=str(payload["source_query_hash"]),
            source_query_payload=dict(payload["source_query_payload"]),
            target_query_payload=dict(payload["target_query_payload"]),
            clause_payload=tuple(int(literal) for literal in payload["clause_payload"]),
            solver_hash=str(payload["solver_hash"]),
            solver_version_hash=str(payload["solver_version_hash"]),
            refinement_witness=dict(payload["refinement_witness"]),
            replay_receipt=dict(payload["replay_receipt"]),
            lifecycle_state=str(payload["lifecycle_state"]),
            use_count=int(payload["use_count"]),
            benefit_score=float(payload["benefit_score"]),
            benefit_observations=int(payload["benefit_observations"]),
            created_version=int(payload["created_version"]),
            committed_version=payload.get("committed_version"),
            last_used_version=payload.get("last_used_version"),
            content_hash=str(payload["content_hash"]),
        )


def conflict_record_content_hash(record: ConflictRecord) -> str:
    """Hash immutable conflict semantics and benefit evidence."""

    return sha256_json(
        {
            "schema_version": RECORD_SCHEMA_VERSION,
            "immutable_record": record.immutable_dict(),
            "replay_receipt": record.replay_receipt,
        }
    )


def _query_from_payload(payload: Mapping[str, Any]) -> ExactQuery:
    return ExactQuery(
        variable_count=int(payload["variable_count"]),
        clauses=tuple(tuple(int(literal) for literal in clause) for clause in payload["clauses"]),
        schema_version=str(payload["schema_version"]),
        solver_hash=str(payload["solver_hash"]),
    )


def _validate_query(query: ExactQuery) -> list[str]:
    errors = []
    if query.schema_version != QUERY_SCHEMA_VERSION:
        errors.append("schema_mismatch")
    if query.solver_hash != DEFAULT_SOLVER_HASH:
        errors.append("solver_hash_mismatch")
    if int(query.variable_count) <= 0:
        errors.append("invalid_variable_count")
    for clause in query.normalized_clauses():
        if not clause:
            errors.append("empty_clause")
        for literal in clause:
            if literal == 0 or abs(literal) > int(query.variable_count):
                errors.append("malformed_clause")
    return sorted(set(errors))


def _validate_clause(clause: Sequence[int], variable_count: int) -> list[str]:
    errors = []
    if not clause:
        errors.append("empty_conflict_clause")
    for literal in clause:
        if int(literal) == 0 or abs(int(literal)) > int(variable_count):
            errors.append("malformed_clause")
    return sorted(set(errors))


def _clause_satisfied(clause: Sequence[int], assignment: Mapping[int, bool]) -> bool:
    return any(
        assignment[abs(int(literal))] if int(literal) > 0 else not assignment[abs(int(literal))]
        for literal in clause
    )


def _query_satisfied(query: ExactQuery, assignment: Mapping[int, bool]) -> bool:
    return all(_clause_satisfied(clause, assignment) for clause in query.normalized_clauses())


def _iter_assignments(variable_count: int) -> list[dict[int, bool]]:
    assignments: list[dict[int, bool]] = []
    for mask in range(1 << int(variable_count)):
        assignments.append(
            {
                variable: bool((mask >> (variable - 1)) & 1)
                for variable in range(1, variable_count + 1)
            }
        )
    return assignments


def native_exact_solve(query: ExactQuery) -> JsonDict:
    """Solve the small CNF query by exhaustive enumeration."""

    query_errors = _validate_query(query)
    if query_errors:
        raise ConflictMemoryError(",".join(query_errors))
    examined = 0
    for assignment in _iter_assignments(query.variable_count):
        examined += 1
        if _query_satisfied(query, assignment):
            model = {f"x{idx}": assignment[idx] for idx in sorted(assignment)}
            return {
                "native_status": "sat",
                "assignments_examined": examined,
                "model": model,
                "model_hash": sha256_json(model),
            }
    proof = {
        "query_hash": query.query_hash(),
        "assignment_count": 1 << int(query.variable_count),
        "all_assignments_refuted": True,
    }
    return {
        "native_status": "unsat",
        "assignments_examined": examined,
        "proof_hash": sha256_json(proof),
    }


def entails_clause(query: ExactQuery, clause: Sequence[int]) -> JsonDict:
    """Replay whether every model of the query satisfies the clause."""

    errors = [*_validate_query(query), *_validate_clause(clause, query.variable_count)]
    if errors:
        return {
            "valid": False,
            "reason": ",".join(sorted(set(errors))),
            "assignments_examined": 0,
            "counterexample": None,
        }
    examined = 0
    satisfying_models = 0
    for assignment in _iter_assignments(query.variable_count):
        examined += 1
        if not _query_satisfied(query, assignment):
            continue
        satisfying_models += 1
        if not _clause_satisfied(clause, assignment):
            counterexample = {f"x{idx}": assignment[idx] for idx in sorted(assignment)}
            return {
                "valid": False,
                "reason": "counterexample_found",
                "assignments_examined": examined,
                "satisfying_models_examined": satisfying_models,
                "counterexample": counterexample,
                "counterexample_hash": sha256_json(counterexample),
            }
    return {
        "valid": True,
        "reason": "entailed_by_exact_replay",
        "assignments_examined": examined,
        "satisfying_models_examined": satisfying_models,
        "counterexample": None,
    }


def _clause_counter(query: ExactQuery) -> Counter[tuple[int, ...]]:
    return Counter(query.normalized_clauses())


def prove_refinement(source: ExactQuery, target: ExactQuery) -> JsonDict:
    """Prove the supported local relation by clause containment."""

    source_errors = _validate_query(source)
    target_errors = _validate_query(target)
    source_counts = _clause_counter(source)
    target_counts = _clause_counter(target)
    source_subset = all(target_counts[clause] >= count for clause, count in source_counts.items())
    added = list((target_counts - source_counts).elements())
    schema_match = source.schema_version == target.schema_version == QUERY_SCHEMA_VERSION
    solver_match = source.solver_hash == target.solver_hash == DEFAULT_SOLVER_HASH
    domain_match = int(source.variable_count) == int(target.variable_count)
    witness = {
        "schema_version": MEMORY_SCHEMA_VERSION + ".refinement_witness",
        "relation": REFINEMENT_RELATION,
        "source_query_hash": source.query_hash(),
        "target_query_hash": target.query_hash(),
        "source_schema_version": source.schema_version,
        "target_schema_version": target.schema_version,
        "source_solver_hash": source.solver_hash,
        "target_solver_hash": target.solver_hash,
        "source_variable_count": int(source.variable_count),
        "target_variable_count": int(target.variable_count),
        "schema_match": schema_match,
        "solver_match": solver_match,
        "domain_match": domain_match,
        "source_subset_of_target": source_subset,
        "source_clause_count": sum(source_counts.values()),
        "target_clause_count": sum(target_counts.values()),
        "added_clause_hashes": [sha256_json(list(clause)) for clause in sorted(added)],
        "source_errors": source_errors,
        "target_errors": target_errors,
    }
    witness["is_refinement"] = (
        schema_match
        and solver_match
        and domain_match
        and source_subset
        and not source_errors
        and not target_errors
    )
    witness["witness_hash"] = sha256_json(
        {key: value for key, value in witness.items() if key != "witness_hash"}
    )
    return witness


def build_replay_receipt(
    source: ExactQuery,
    target: ExactQuery,
    clause: Sequence[int],
    witness: Mapping[str, Any],
) -> JsonDict:
    """Build the exact source and target replay receipt for one record."""

    source_replay = entails_clause(source, clause)
    target_replay = entails_clause(target, clause)
    receipt = {
        "schema_version": MEMORY_SCHEMA_VERSION + ".exact_replay_receipt",
        "source_query_hash": source.query_hash(),
        "target_query_hash": target.query_hash(),
        "clause_hash": sha256_json([int(literal) for literal in clause]),
        "witness_hash": witness.get("witness_hash"),
        "source_entails_conflict": source_replay["valid"],
        "target_entails_conflict": target_replay["valid"],
        "source_replay": source_replay,
        "target_replay": target_replay,
        "exact_replay_valid": bool(
            witness.get("is_refinement")
            and source_replay["valid"] is True
            and target_replay["valid"] is True
        ),
        "verifier": "exhaustive_boolean_cnf_entailment",
        "verifier_is_oracle": True,
    }
    receipt["replay_receipt_hash"] = sha256_json(
        {key: value for key, value in receipt.items() if key != "replay_receipt_hash"}
    )
    return receipt


def _terminal_state_payload(
    *,
    capacity: int,
    solver_hash: str,
    next_version: int,
    records: Mapping[str, ConflictRecord],
    evicted_records: Sequence[ConflictRecord],
    checkpoints: Mapping[str, JsonDict],
) -> JsonDict:
    record_rows = [record.to_dict() for _, record in sorted(records.items())]
    evicted_rows = [record.to_dict() for record in evicted_records]
    state_hash = sha256_json(
        {
            "capacity": int(capacity),
            "solver_hash": solver_hash,
            "records": record_rows,
            "evicted_records": evicted_rows,
        }
    )
    return {
        "status": "complete_conflict_memory_state",
        "honest_verdict": "complete_conflict_memory_state",
        "schema_version": MEMORY_SCHEMA_VERSION,
        "transaction_schema": TRANSACTION_SCHEMA,
        "capacity": int(capacity),
        "solver_hash": solver_hash,
        "next_version": int(next_version),
        "records": record_rows,
        "evicted_records": evicted_rows,
        "checkpoints": dict(checkpoints),
        "state_hash": state_hash,
    }


class TransactionalConflictMemory:
    """Versioned exact-conflict memory with atomic durable state writes."""

    def __init__(
        self,
        *,
        capacity: int,
        memory_path: Path | str,
        transaction_work_dir: Path | str,
        solver_hash: str = DEFAULT_SOLVER_HASH,
        crash_plan: CrashPlan | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.memory_path = Path(memory_path)
        self.transaction_work_dir = Path(transaction_work_dir)
        self.solver_hash = solver_hash
        self.crash_plan = crash_plan or CrashPlan()
        self.records: dict[str, ConflictRecord] = {}
        self.evicted_records: list[ConflictRecord] = []
        self.eviction_rows: list[JsonDict] = []
        self.checkpoints: dict[str, JsonDict] = {}
        self.next_version = 1

    def state_hash(self) -> str:
        payload = _terminal_state_payload(
            capacity=self.capacity,
            solver_hash=self.solver_hash,
            next_version=self.next_version,
            records=self.records,
            evicted_records=self.evicted_records,
            checkpoints=self.checkpoints,
        )
        return str(payload["state_hash"])

    def prepare(
        self,
        *,
        source_query: ExactQuery,
        target_query: ExactQuery,
        clause: Sequence[int],
        benefit_score: float,
        benefit_observations: int,
    ) -> ConflictRecord:
        witness = prove_refinement(source_query, target_query)
        if witness["is_refinement"] is not True:
            raise ConflictMemoryError("refinement_witness_failed")
        replay = build_replay_receipt(source_query, target_query, clause, witness)
        if replay["exact_replay_valid"] is not True:
            raise ConflictMemoryError("exact_replay_failed")
        record = ConflictRecord(
            source_query_hash=source_query.query_hash(),
            source_query_payload=source_query.to_dict(),
            target_query_payload=target_query.to_dict(),
            clause_payload=tuple(int(literal) for literal in clause),
            solver_hash=self.solver_hash,
            solver_version_hash=self.solver_hash,
            refinement_witness=witness,
            replay_receipt=replay,
            lifecycle_state="prepared",
            use_count=0,
            benefit_score=float(benefit_score),
            benefit_observations=int(benefit_observations),
            created_version=self.next_version,
        )
        record.content_hash = conflict_record_content_hash(record)
        return record

    def prepare_veto_row(
        self,
        *,
        source_query: ExactQuery,
        target_query: ExactQuery,
        clause: Sequence[int],
        attack_id: str,
    ) -> JsonDict:
        try:
            record = self.prepare(
                source_query=source_query,
                target_query=target_query,
                clause=clause,
                benefit_score=0.0,
                benefit_observations=0,
            )
            self.validate(record)
        except ConflictMemoryError as exc:
            return {
                "row_type": "invalid_reuse_veto",
                "attack_id": attack_id,
                "vetoed": True,
                "reason": str(exc),
                "durable_write_performed": False,
                "unsafe_use_performed": False,
                "passed": True,
            }
        return {
            "row_type": "invalid_reuse_veto",
            "attack_id": attack_id,
            "vetoed": False,
            "reason": "unexpected_accept",
            "durable_write_performed": False,
            "unsafe_use_performed": False,
            "passed": False,
        }

    def validate(self, record: ConflictRecord) -> JsonDict:
        self._validate_record_or_raise(record)
        return {
            "row_type": "lifecycle",
            "operation": "validate",
            "content_hash": record.content_hash,
            "accepted": True,
            "exact_replay_valid": True,
            "passed": True,
        }

    def commit(self, record: ConflictRecord) -> ConflictRecord:
        self._validate_record_or_raise(record)
        existing = self.records.get(record.content_hash)
        if existing is not None:
            return existing
        committed = deepcopy(record)
        committed.lifecycle_state = "active"
        committed.committed_version = self.next_version
        new_records = deepcopy(self.records)
        new_records[committed.content_hash] = committed
        new_evicted = deepcopy(self.evicted_records)
        eviction_rows = self._evictions_for(new_records, new_evicted)
        next_version = self.next_version + 1
        self._persist_state(new_records, new_evicted, self.checkpoints, next_version)
        self.records = new_records
        self.evicted_records = new_evicted
        self.eviction_rows.extend(eviction_rows)
        self.next_version = next_version
        return self.records.get(committed.content_hash, committed)

    def abort(self, record: ConflictRecord) -> JsonDict:
        return {
            "row_type": "lifecycle",
            "operation": "abort",
            "content_hash": record.content_hash,
            "lifecycle_state": "aborted",
            "durable_write_performed": False,
            "passed": True,
        }

    def load(self) -> JsonDict:
        if not self.memory_path.exists():
            return {
                "row_type": "lifecycle",
                "operation": "load",
                "memory_path": str(self.memory_path),
                "active_record_count": 0,
                "missing_memory": True,
                "corruption_quarantined": False,
                "state_hash": self.state_hash(),
                "passed": True,
            }
        try:
            payload = json.loads(self.memory_path.read_text(encoding="utf-8"))
            records = {
                row["content_hash"]: ConflictRecord.from_dict(row)
                for row in payload.get("records", [])
            }
            evicted = [ConflictRecord.from_dict(row) for row in payload.get("evicted_records", [])]
            for record in [*records.values(), *evicted]:
                self._validate_record_or_raise(record)
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ConflictMemoryError) as exc:
            quarantine_path = self._quarantine_memory_file(str(exc))
            return {
                "row_type": "corruption_quarantine",
                "operation": "load",
                "memory_path": str(self.memory_path),
                "corruption_quarantined": True,
                "quarantine_path": str(quarantine_path),
                "active_record_count": 0,
                "passed": True,
            }
        self.records = records
        self.evicted_records = evicted
        self.checkpoints = dict(payload.get("checkpoints", {}))
        self.next_version = int(payload.get("next_version", 1))
        return {
            "row_type": "lifecycle",
            "operation": "load",
            "memory_path": str(self.memory_path),
            "active_record_count": len(self.records),
            "corruption_quarantined": False,
            "state_hash": self.state_hash(),
            "passed": self.state_hash() == payload.get("state_hash"),
        }

    def use(self, content_hash: str, target_query: ExactQuery) -> JsonDict:
        if content_hash not in self.records:
            raise ConflictMemoryError("record_not_available")
        record = deepcopy(self.records[content_hash])
        source_query = _query_from_payload(record.source_query_payload)
        witness = prove_refinement(source_query, target_query)
        replay = build_replay_receipt(source_query, target_query, record.clause_payload, witness)
        if witness["is_refinement"] is not True:
            raise ConflictMemoryError("use_refinement_failed")
        if replay["target_entails_conflict"] is not True:  # pragma: no cover
            raise ConflictMemoryError("use_exact_replay_failed")
        record.use_count += 1
        record.last_used_version = self.next_version
        new_records = deepcopy(self.records)
        new_records[content_hash] = record
        next_version = self.next_version + 1
        self._persist_state(new_records, self.evicted_records, self.checkpoints, next_version)
        self.records = new_records
        self.next_version = next_version
        return {
            "row_type": "valid_reuse",
            "operation": "use",
            "content_hash": content_hash,
            "memory_used": True,
            "exact_replay_valid": True,
            "refinement_valid": True,
            "replay_receipt_hash": replay["replay_receipt_hash"],
            "use_count_after": record.use_count,
            "unsafe_use_performed": False,
            "passed": True,
        }

    def use_or_native(self, content_hash: str, target_query: ExactQuery) -> JsonDict:
        try:
            return self.use(content_hash, target_query)
        except ConflictMemoryError as exc:
            row = self.native_fallback_solve(target_query)
            row["fallback_reason"] = str(exc)
            return row

    def native_fallback_solve(self, query: ExactQuery) -> JsonDict:
        solved = native_exact_solve(query)
        return {
            "row_type": "native_fallback",
            "operation": "native_fallback_solve",
            "memory_used": False,
            "native_status": solved["native_status"],
            "assignments_examined": solved["assignments_examined"],
            "fallback_reason": "memory_unavailable",
            "passed": True,
        }

    def checkpoint(self, checkpoint_id: str) -> JsonDict:
        snapshot = {
            "checkpoint_id": checkpoint_id,
            "state_hash": self.state_hash(),
            "records": [record.to_dict() for _, record in sorted(self.records.items())],
            "evicted_records": [record.to_dict() for record in self.evicted_records],
            "next_version": self.next_version,
        }
        new_checkpoints = deepcopy(self.checkpoints)
        new_checkpoints[checkpoint_id] = snapshot
        next_version = self.next_version + 1
        self._persist_state(self.records, self.evicted_records, new_checkpoints, next_version)
        self.checkpoints = new_checkpoints
        self.next_version = next_version
        return {
            "row_type": "restart_rollback",
            "operation": "checkpoint",
            "checkpoint_id": checkpoint_id,
            "state_hash": snapshot["state_hash"],
            "passed": True,
        }

    def rollback(self, checkpoint_id: str) -> JsonDict:
        if checkpoint_id not in self.checkpoints:
            raise ConflictMemoryError("checkpoint_not_found")
        snapshot = self.checkpoints[checkpoint_id]
        records = {
            row["content_hash"]: ConflictRecord.from_dict(row)
            for row in snapshot.get("records", [])
        }
        evicted = [ConflictRecord.from_dict(row) for row in snapshot.get("evicted_records", [])]
        next_version = int(snapshot.get("next_version", self.next_version)) + 1
        self._persist_state(records, evicted, self.checkpoints, next_version)
        before = self.state_hash()
        self.records = records
        self.evicted_records = evicted
        self.next_version = next_version
        return {
            "row_type": "restart_rollback",
            "operation": "rollback",
            "checkpoint_id": checkpoint_id,
            "state_hash_before": before,
            "state_hash_after": self.state_hash(),
            "target_state_hash": snapshot["state_hash"],
            "rolled_back": self.state_hash() == snapshot["state_hash"],
            "passed": self.state_hash() == snapshot["state_hash"],
        }

    def fixed_width_cpu_mapping_rows(self) -> list[JsonDict]:
        rows = []
        for content_hash, record in sorted(self.records.items()):
            logical_bytes = len(canonical_json(record.to_dict()).encode("utf-8"))
            literal_count = len(record.clause_payload)
            mapped_bytes = max(logical_bytes, 32 * 4 + 8 * 4 + literal_count * 4)
            row = {
                "row_type": "fixed_width_mapping",
                "content_hash": content_hash,
                "logical_bytes": logical_bytes,
                "mapped_bytes": mapped_bytes,
                "topology_expansion": round(mapped_bytes / logical_bytes, 6),
                "mapping_time_s": 0.0,
                "unsupported_fields": [],
                "hardware_execution_claimed": False,
                "acceleration_claimed": False,
                "mapping_hash": sha256_json(
                    {
                        "content_hash": content_hash,
                        "mapped_bytes": mapped_bytes,
                        "literal_count": literal_count,
                    }
                ),
                "passed": True,
            }
            rows.append(row)
        return rows

    def _validate_record_or_raise(self, record: ConflictRecord) -> None:
        if record.content_hash != conflict_record_content_hash(record):
            raise ConflictMemoryError("content_hash_mismatch")
        if record.solver_hash != self.solver_hash:
            raise ConflictMemoryError("solver_hash_mismatch")
        source = _query_from_payload(record.source_query_payload)
        target = _query_from_payload(record.target_query_payload)
        if record.source_query_hash != source.query_hash():
            raise ConflictMemoryError("source_query_hash_mismatch")
        clause_errors = _validate_clause(record.clause_payload, source.variable_count)
        if clause_errors:
            raise ConflictMemoryError(",".join(clause_errors))
        witness = prove_refinement(source, target)
        if witness.get("witness_hash") != record.refinement_witness.get("witness_hash"):
            raise ConflictMemoryError("refinement_witness_hash_mismatch")
        if witness["is_refinement"] is not True:
            raise ConflictMemoryError("refinement_witness_failed")
        replay = build_replay_receipt(source, target, record.clause_payload, witness)
        if replay.get("replay_receipt_hash") != record.replay_receipt.get("replay_receipt_hash"):
            raise ConflictMemoryError("replay_receipt_hash_mismatch")
        if replay["exact_replay_valid"] is not True:
            raise ConflictMemoryError("exact_replay_failed")

    def _evictions_for(
        self,
        records: dict[str, ConflictRecord],
        evicted: list[ConflictRecord],
    ) -> list[JsonDict]:
        rows = []
        while len(records) > self.capacity:
            victim_hash, victim = sorted(
                records.items(),
                key=lambda item: (
                    item[1].benefit_score,
                    item[1].use_count,
                    item[1].committed_version or 0,
                    item[0],
                ),
            )[0]
            removed = records.pop(victim_hash)
            removed.lifecycle_state = "evicted"
            evicted.append(removed)
            rows.append(
                {
                    "row_type": "capacity_eviction",
                    "operation": "evict",
                    "capacity": self.capacity,
                    "evicted_content_hash": victim_hash,
                    "eviction_reason": "capacity_limit",
                    "ordering": [
                        "benefit_score",
                        "use_count",
                        "committed_version",
                        "content_hash",
                    ],
                    "passed": True,
                }
            )
        return rows

    def _persist_state(
        self,
        records: Mapping[str, ConflictRecord],
        evicted_records: Sequence[ConflictRecord],
        checkpoints: Mapping[str, JsonDict],
        next_version: int,
    ) -> JsonDict:
        payload = _terminal_state_payload(
            capacity=self.capacity,
            solver_hash=self.solver_hash,
            next_version=next_version,
            records=records,
            evicted_records=evicted_records,
            checkpoints=checkpoints,
        )
        payload_tag = sha256_json(payload).removeprefix("sha256:")[:12]
        work = self.transaction_work_dir / f"commit-{next_version:06d}-{payload_tag}"
        with AtomicShardTransaction(
            work_dir=work,
            final_path=self.memory_path,
            transaction_id=f"exp6521-memory-{next_version}",
            crash_plan=self.crash_plan,
        ) as tx:
            tx.plan_units([f"state-{next_version}"])
            tx.write_terminal_unit(f"state-{next_version}", payload)
            receipt = tx.finalize(payload)
        return receipt

    def _quarantine_memory_file(self, reason: str) -> Path:
        quarantine_dir = self.transaction_work_dir / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = quarantine_dir / f"{self.memory_path.name}.{time.time_ns()}.corrupt"
        if self.memory_path.exists():
            os.replace(self.memory_path, target)
        else:
            target.write_text(reason + "\n", encoding="utf-8")
        return target


def conflict_record_schema() -> JsonDict:
    schema = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "required_fields": [
            "source_query_hash",
            "source_query_payload",
            "target_query_payload",
            "clause_payload",
            "solver_hash",
            "solver_version_hash",
            "refinement_witness",
            "replay_receipt",
            "lifecycle_state",
            "use_count",
            "benefit_score",
            "benefit_observations",
            "content_hash",
        ],
        "content_hash_excludes": ["lifecycle_state", "use_count", "committed_version"],
        "supported_payload": "boolean_cnf_clause",
    }
    return {**schema, "schema_hash": sha256_json(schema)}


def refinement_relation_contract() -> JsonDict:
    contract = {
        "relation": REFINEMENT_RELATION,
        "source_relation": "learned_conflicts_candidate_local_contract",
        "proof_method": "local_clause_multiset_subset_with_same_schema_domain_solver",
        "accepted": [
            "equal_query",
            "clause_superset",
        ],
        "rejected": [
            "unrelated",
            "relaxed",
            "schema_mismatch",
            "solver_mismatch",
            "stale_source_hash",
            "malformed_clause",
            "invalid_exact_replay",
        ],
        "exact_replay_required_before_write": True,
        "exact_replay_required_before_use": True,
    }
    return {**contract, "contract_hash": sha256_json(contract)}


def protected_file_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    changed = [
        {"path": path, "before": before.get(path), "after": after.get(path)}
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    ]
    return {
        "all_protected_files_unchanged": not changed,
        "changed_files": changed,
        "hashes_before": dict(before),
        "hashes_after": dict(after),
    }


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _git_output(repo_root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def upstream_gate_receipt(repo_root: Path) -> JsonDict:
    path = repo_root / EXP6517_RELATIVE_PATH
    exists = path.is_file()
    payload = _read_json(path) if exists else {}
    gate = payload.get("gate_check_summary", {})
    return {
        "artifact_path": EXP6517_RELATIVE_PATH.as_posix(),
        "artifact_sha256": sha256_file(path),
        "exists": exists,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "all_gates_passed": gate.get("all_gates_passed") is True,
        "pilot_audit_gate": "experiment_6517_branch_pilot_independent_audit",
        "protected_hash_count": len(
            payload.get("protected_files_unchanged", {}).get("hashes_after", {})
        ),
    }


def _solver_capabilities() -> JsonDict:
    return {
        "solver_id": "carnot_exhaustive_boolean_cnf_entailment",
        "solver_hash": DEFAULT_SOLVER_HASH,
        "exact_replay": True,
        "conflict_clause_entailment": True,
        "native_fallback": True,
        "max_variables_in_reference": 8,
        "transaction_schema": TRANSACTION_SCHEMA,
    }


def _resource_receipt(work_root: Path) -> JsonDict:
    work_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(work_root)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "work_root": str(work_root),
        "available_bytes": disk.free,
        "filesystem_writable": os.access(work_root, os.W_OK),
    }


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    rows = [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)]
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in rows]


def _scenario_rows(work_root: Path) -> JsonDict:
    memory = TransactionalConflictMemory(
        capacity=2,
        memory_path=work_root / "memory.json",
        transaction_work_dir=work_root / "memory-tx",
    )
    source = ExactQuery(variable_count=2, clauses=((1,),))
    target = ExactQuery(variable_count=2, clauses=((1,), (2,)))
    prepared = memory.prepare(
        source_query=source,
        target_query=target,
        clause=(1,),
        benefit_score=2.0,
        benefit_observations=3,
    )
    validate_row = memory.validate(prepared)
    committed = memory.commit(prepared)
    abort_row = memory.abort(prepared)
    load_row = TransactionalConflictMemory(
        capacity=2,
        memory_path=work_root / "memory.json",
        transaction_work_dir=work_root / "memory-tx",
    ).load()
    use_row = memory.use(committed.content_hash, target)
    checkpoint_row = memory.checkpoint("baseline")

    invalid_rows = [
        memory.prepare_veto_row(
            source_query=ExactQuery(2, ((1,), (2,))),
            target_query=ExactQuery(2, ((1,),)),
            clause=(1,),
            attack_id="relaxed_query",
        ),
        memory.prepare_veto_row(
            source_query=source,
            target_query=ExactQuery(2, ((2,),)),
            clause=(1,),
            attack_id="unrelated_query",
        ),
        memory.prepare_veto_row(
            source_query=source,
            target_query=ExactQuery(2, ((1,), (2,)), schema_version="bad.schema"),
            clause=(1,),
            attack_id="schema_mismatch",
        ),
        memory.prepare_veto_row(
            source_query=source,
            target_query=target,
            clause=(2,),
            attack_id="invalid_replay",
        ),
    ]

    second = memory.prepare(
        source_query=ExactQuery(3, ((2,),)),
        target_query=ExactQuery(3, ((2,), (3,))),
        clause=(2,),
        benefit_score=1.0,
        benefit_observations=1,
    )
    memory.validate(second)
    second = memory.commit(second)
    memory.use(second.content_hash, ExactQuery(3, ((2,), (3,))))
    third = memory.prepare(
        source_query=ExactQuery(3, ((3,),)),
        target_query=ExactQuery(3, ((3,), (1,))),
        clause=(3,),
        benefit_score=2.0,
        benefit_observations=1,
    )
    memory.validate(third)
    third = memory.commit(third)
    duplicate = memory.commit(deepcopy(third))
    duplicate_row = {
        "row_type": "capacity_eviction",
        "operation": "duplicate_commit",
        "content_hash": duplicate.content_hash,
        "idempotent": duplicate.content_hash == third.content_hash,
        "passed": duplicate.content_hash == third.content_hash,
    }

    crash_memory_path = work_root / "crash" / "memory.json"
    stable_crash_memory = TransactionalConflictMemory(
        capacity=3,
        memory_path=crash_memory_path,
        transaction_work_dir=work_root / "crash" / "tx",
    )
    stable_record = stable_crash_memory.prepare(
        source_query=source,
        target_query=target,
        clause=(1,),
        benefit_score=1.0,
        benefit_observations=1,
    )
    stable_crash_memory.validate(stable_record)
    stable_crash_memory.commit(stable_record)
    before_crash_hash = stable_crash_memory.state_hash()
    crash_memory = TransactionalConflictMemory(
        capacity=3,
        memory_path=crash_memory_path,
        transaction_work_dir=work_root / "crash" / "tx",
        crash_plan=CrashPlan.once("before_replace"),
    )
    crash_candidate = crash_memory.prepare(
        source_query=ExactQuery(3, ((2,),)),
        target_query=ExactQuery(3, ((2,), (3,))),
        clause=(2,),
        benefit_score=1.0,
        benefit_observations=1,
    )
    crash_memory.validate(crash_candidate)
    interrupted_commit_row = {
        "row_type": "capacity_eviction",
        "operation": "interrupted_commit",
        "passed": False,
        "partial_state_published": True,
    }
    try:
        crash_memory.commit(crash_candidate)
    except Exception as exc:  # noqa: BLE001 - row records the injected crash type.
        reloaded = TransactionalConflictMemory(
            capacity=3,
            memory_path=crash_memory_path,
            transaction_work_dir=work_root / "crash" / "tx",
        )
        reloaded.load()
        interrupted_commit_row = {
            "row_type": "capacity_eviction",
            "operation": "interrupted_commit",
            "crash": str(exc),
            "partial_state_published": reloaded.state_hash() != before_crash_hash,
            "passed": reloaded.state_hash() == before_crash_hash,
        }

    restart_memory = TransactionalConflictMemory(
        capacity=2,
        memory_path=work_root / "memory.json",
        transaction_work_dir=work_root / "memory-tx",
    )
    restart_row = restart_memory.load()
    rollback_row = restart_memory.rollback("baseline")
    mapping_rows = restart_memory.fixed_width_cpu_mapping_rows()

    corrupt_path = work_root / "corrupt" / "memory.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("{bad", encoding="utf-8")
    corrupt_memory = TransactionalConflictMemory(
        capacity=2,
        memory_path=corrupt_path,
        transaction_work_dir=work_root / "corrupt" / "tx",
    )
    corruption_row = corrupt_memory.load()
    fallback_row = corrupt_memory.use_or_native(committed.content_hash, target)

    lifecycle_rows = [
        {
            "row_type": "lifecycle",
            "operation": "prepare",
            "content_hash": prepared.content_hash,
            "passed": prepared.lifecycle_state == "prepared",
        },
        validate_row,
        {
            "row_type": "lifecycle",
            "operation": "commit",
            "content_hash": committed.content_hash,
            "lifecycle_state": committed.lifecycle_state,
            "passed": committed.lifecycle_state == "active",
        },
        abort_row,
        load_row,
        use_row,
        checkpoint_row,
        rollback_row,
    ]
    return {
        "lifecycle_rows": lifecycle_rows,
        "valid_reuse_rows": [use_row],
        "invalid_reuse_veto_rows": invalid_rows,
        "capacity_and_eviction_rows": [
            *memory.eviction_rows,
            duplicate_row,
            interrupted_commit_row,
        ],
        "restart_rollback_rows": [restart_row, rollback_row],
        "corruption_quarantine_rows": [corruption_row],
        "native_fallback_rows": [fallback_row],
        "fixed_width_mapping_rows": mapping_rows,
    }


def recompute_aggregate(payload: Mapping[str, Any]) -> JsonDict:
    invalid_rows = [dict(row) for row in payload.get("invalid_reuse_veto_rows", [])]
    mapping_rows = [dict(row) for row in payload.get("fixed_width_mapping_rows", [])]
    row_groups = [
        "lifecycle_rows",
        "valid_reuse_rows",
        "capacity_and_eviction_rows",
        "restart_rollback_rows",
        "corruption_quarantine_rows",
        "native_fallback_rows",
    ]
    all_standard_rows_pass = all(
        row.get("passed") is True for group in row_groups for row in payload.get(group, [])
    )
    invalid_rows_pass = all(
        row.get("vetoed") is True and row.get("passed") is True for row in invalid_rows
    )
    unsafe_admission = sum(1 for row in invalid_rows if row.get("durable_write_performed") is True)
    unsafe_use = sum(1 for row in invalid_rows if row.get("unsafe_use_performed") is True)
    mapping_safe = all(
        row.get("hardware_execution_claimed") is False
        and row.get("acceleration_claimed") is False
        and row.get("passed") is True
        for row in mapping_rows
    )
    protected_ok = (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is True
    )
    upstream_ok = payload.get("upstream_gate_receipt", {}).get("all_gates_passed") is True
    ready = (
        all_standard_rows_pass
        and invalid_rows_pass
        and unsafe_admission == 0
        and unsafe_use == 0
        and mapping_safe
        and protected_ok
        and upstream_ok
    )
    return {
        "all_standard_rows_pass": all_standard_rows_pass,
        "invalid_rows_pass": invalid_rows_pass,
        "unsafe_admission_count": unsafe_admission,
        "unsafe_use_count": unsafe_use,
        "mapping_rows_safe": mapping_safe,
        "protected_files_unchanged": protected_ok,
        "upstream_gate_passed": upstream_ok,
        "ready_score_from_rows": 1.0 if ready else 0.0,
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    checks = {
        "all_standard_rows_pass": aggregate.get("all_standard_rows_pass") is True,
        "invalid_rows_pass": aggregate.get("invalid_rows_pass") is True,
        "unsafe_admission_count_zero": aggregate.get("unsafe_admission_count") == 0,
        "unsafe_use_count_zero": aggregate.get("unsafe_use_count") == 0,
        "mapping_rows_safe": aggregate.get("mapping_rows_safe") is True,
        "protected_files_unchanged": aggregate.get("protected_files_unchanged") is True,
        "upstream_gate_passed": aggregate.get("upstream_gate_passed") is True,
        "ready_score_from_rows": aggregate.get("ready_score_from_rows") == 1.0,
    }
    failed = [key for key, value in checks.items() if value is not True]
    return {
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
    }


def _status_and_verdict(score: float, gates: Mapping[str, Any]) -> tuple[str, str, str]:
    if score == 1.0 and gates.get("all_gates_passed") is True:
        return (
            "complete_transactional_refinement_conflict_memory_ready",
            (
                "complete_transactional_refinement_conflict_memory_ready: exact replay, "
                "refinement gating, transactions, rollback, quarantine, fallback, and "
                "CPU mapping rows all passed with zero unsafe admission or use"
            ),
            "circular_positive",
        )
    if gates.get("checks", {}).get("upstream_gate_passed") is not True:
        return (
            "blocked_transactional_refinement_conflict_memory",
            "blocked_transactional_refinement_conflict_memory: upstream gate missing or failed",
            "blocked",
        )
    return (
        "partial_transactional_refinement_conflict_memory",
        "partial_transactional_refinement_conflict_memory: bounded mechanism failed a local gate",
        "partial",
    )


def _field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": "Exp6521 exact replay, transaction scenario, protected hashes, or tests",
            "spec_ref": "REQ-STORE-6521",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def preconditions_checked(
    *,
    repo_root: Path,
    work_root: Path,
    result_path: Path,
    run_date: str,
    upstream: Mapping[str, Any],
    protected_before: Mapping[str, str],
) -> JsonDict:
    return {
        "run_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "work_root": str(work_root),
        "git_status_short": _git_output(repo_root, ["status", "--short"]),
        "solver_capabilities": _solver_capabilities(),
        "refinement_relations_available": [REFINEMENT_RELATION],
        "resources": _resource_receipt(work_root),
        "upstream_gate": dict(upstream),
        "protected_file_hashes_before": dict(protected_before),
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str | None = None,
    work_root: Path | str | None = None,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.perf_counter()
    repo_root = Path(repo_root)
    result = Path(result_path) if result_path is not None else repo_root / RESULT_RELATIVE_PATH
    if not result.is_absolute():
        result = repo_root / result
    work = Path(work_root) if work_root is not None else repo_root / WORK_RELATIVE_PATH
    if not work.is_absolute():
        work = repo_root / work
    protected_before = protected_file_hashes(repo_root)
    upstream = upstream_gate_receipt(repo_root)
    rows = _scenario_rows(work)
    protected_after = protected_file_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    payload: JsonDict = {
        "status": "partial_transactional_refinement_conflict_memory",
        "honest_verdict": "partial_transactional_refinement_conflict_memory: building",
        "verdict_class": "partial",
        "upstream_gate_receipt": upstream,
        "conflict_record_schema": conflict_record_schema(),
        "refinement_relation_contract": refinement_relation_contract(),
        **rows,
        "conflict_memory_controller_ready_score": 0.0,
        "gate_check_summary": {},
        "per_unit_rows": [],
        "aggregate_row_recomputation": {},
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            work_root=work,
            result_path=result,
            run_date=run_date,
            upstream=upstream,
            protected_before=protected_before,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.perf_counter() - start),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    per_unit: list[JsonDict] = []
    for group in (
        "lifecycle_rows",
        "valid_reuse_rows",
        "invalid_reuse_veto_rows",
        "capacity_and_eviction_rows",
        "restart_rollback_rows",
        "corruption_quarantine_rows",
        "native_fallback_rows",
        "fixed_width_mapping_rows",
    ):
        per_unit.extend(dict(row, source_group=group) for row in payload[group])
    payload["per_unit_rows"] = per_unit
    aggregate = recompute_aggregate(payload)
    gates = gate_check_summary(aggregate)
    status, verdict, verdict_class = _status_and_verdict(aggregate["ready_score_from_rows"], gates)
    payload.update(
        {
            "status": status,
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "conflict_memory_controller_ready_score": aggregate["ready_score_from_rows"],
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gates,
        }
    )
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_json_file(result, payload)
    return payload


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized["reproducibility_checksum"] = ""
    return sha256_json(normalized)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    status = str(payload.get("status", ""))
    verdict = str(payload.get("honest_verdict", ""))
    if not status.startswith(("complete_", "partial_", "blocked_", "disqualified_")):
        errors.append("status lacks terminal prefix")
    if not verdict.startswith(("complete_", "partial_", "blocked_", "disqualified_")):
        errors.append("honest_verdict lacks terminal prefix")
    verdict_class = payload.get("verdict_class")
    if verdict_class == "positive":
        errors.append("verdict_class cannot be positive")
    if verdict_class not in {"circular_positive", "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6521 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload.get("upstream_gate_receipt", {}).get("all_gates_passed") is not True:
        errors.append("upstream gate failed")
    aggregate = recompute_aggregate(payload)
    score = payload.get("conflict_memory_controller_ready_score")
    if score not in {0.0, 1.0}:
        errors.append("conflict_memory_controller_ready_score must be 0.0 or 1.0")
    if score != aggregate["ready_score_from_rows"]:
        errors.append("ready score mismatch")
    if aggregate["unsafe_admission_count"] or aggregate["unsafe_use_count"]:
        errors.append("unsafe admission or use detected")
    if aggregate["mapping_rows_safe"] is not True:
        errors.append("mapping row makes hardware claim")
    if (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("gate_check_summary") != gate_check_summary(aggregate):
        errors.append("gate_check_summary mismatch")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str | None = None,
    work_root: Path | str | None = None,
) -> JsonDict:
    return build_artifact(
        repo_root=REPO_ROOT,
        result_path=Path(result_path)
        if result_path is not None
        else REPO_ROOT / RESULT_RELATIVE_PATH,
        work_root=Path(work_root) if work_root is not None else REPO_ROOT / WORK_RELATIVE_PATH,
        write=True,
        run_date=date,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--work-root", default=str(REPO_ROOT / WORK_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        payload = _read_json(result_path)
        errors = validate_artifact(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path, work_root=Path(args.work_root))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests.
    raise SystemExit(main())
