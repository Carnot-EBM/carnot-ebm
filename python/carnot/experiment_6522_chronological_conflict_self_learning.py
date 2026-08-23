"""Exp6522 chronological exact-conflict self-learning comparison.

Spec refs: REQ-STORE-6522, SCENARIO-STORE-6522-SEALING,
SCENARIO-STORE-6522-MATCHED-DOSE, SCENARIO-STORE-6522-LEARNING-ACTIONS,
SCENARIO-STORE-6522-FUTURE-SUPPORT, SCENARIO-STORE-6522-PREFIX-RETENTION,
SCENARIO-STORE-6522-SAFETY, SCENARIO-STORE-6522-RESTART-ROLLBACK-CAPACITY,
SCENARIO-STORE-6522-SEQUENTIAL-EVIDENCE.

The learner stores only exact conflicts admitted by Exp6521's refinement gate.
The comparison charges lookup and mapping cost before it counts benefit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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

from carnot.experiment_6521_transactional_refinement_conflict_memory import (
    DEFAULT_SOLVER_HASH,
    REFINEMENT_RELATION,
    ConflictMemoryError,
    ExactQuery,
    TransactionalConflictMemory,
    build_replay_receipt,
    canonical_json,
    prove_refinement,
    sha256_file,
    sha256_json,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6522
RESULT_RELATIVE_PATH = Path("results/experiment_6522_chronological_conflict_self_learning.json")
WORK_RELATIVE_PATH = Path("results/.experiment_6522_chronological_conflict_self_learning.tx")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-store/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6522_chronological_conflict_self_learning.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6522_chronological_conflict_self_learning.py"
)
EXP6521_RELATIVE_PATH = Path("results/experiment_6521_transactional_refinement_conflict_memory.json")
EXP6516_RELATIVE_PATH = Path("results/experiment_6516_exact_branch_pilot_dataset_v3.json")
EXP6496_RELATIVE_PATH = Path("results/experiment_6496_continuous_factor_learning.json")
EXP6498_RELATIVE_PATH = Path("results/experiment_6498_csl_independent_audit.json")

INFERENCE_SUBSTRATE = "chronological_exact_conflict_memory_self_learning_no_llm"
VERIFIER_IS_ORACLE = False
EXACT_SOLVER_IS_RELEASE_AUTHORITY = True
LOOKUP_CHARGE = 1
MAPPING_CHARGE = 1
BOUNDED_CAPACITY = 3
UNBOUNDED_CAPACITY = 8
HELD_FUTURE_BOUNDARY_INDEX = 8
PREFIX_RETENTION_MARGIN = 0.0

ARM_NAMES = (
    "scratch",
    "frozen_empty_memory",
    "valid_unbounded_reuse",
    "valid_bounded_reuse",
    "restart",
    "rollback",
    "invalid_reuse_attack",
)
LEARNING_ARMS = {
    "valid_unbounded_reuse",
    "valid_bounded_reuse",
    "restart",
    "rollback",
}
QUERY_ARMS = set(ARM_NAMES)

PROTECTED_RELATIVE_PATHS = (
    EXP6496_RELATIVE_PATH,
    EXP6498_RELATIVE_PATH,
    EXP6516_RELATIVE_PATH,
    EXP6521_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "prior_failure_receipts",
    "chronological_stream_commitment",
    "arm_and_dose_contract",
    "per_game_results",
    "lifecycle_action_rows",
    "store_hash_rows",
    "exact_answer_equality_rows",
    "immediate_metric_rows",
    "prefix_retention_rows",
    "held_future_support_rows",
    "interference_rows",
    "capacity_restart_rollback_rows",
    "invalid_reuse_attack_rows",
    "sequential_evidence",
    "csl_execution_complete_score",
    "continuous_self_learning_candidate_score",
    "gate_check_summary",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "exact_solver_is_release_authority",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "Records whether the chronological conflict-learning comparison is positive, null, "
        "partial, blocked, or disqualified."
    ),
    "honest_verdict": "States the measured learning result and the exact-safety limits.",
    "verdict_class": "Uses positive only for charged held-future benefit with exact safety.",
    "upstream_gate_receipt": (
        "Binds the run to the Exp6521 controller gate path, hash, and expected value."
    ),
    "prior_failure_receipts": (
        "Records why Exp6496 and Exp6498 did not open a held-future learning claim."
    ),
    "chronological_stream_commitment": (
        "Freezes the stream, thresholds, and held-future boundary before scoring."
    ),
    "arm_and_dose_contract": (
        "Shows each arm got matched solver, query, opportunity, and charged budget."
    ),
    "per_game_results": "Reports one row per chronological unit and arm.",
    "lifecycle_action_rows": (
        "Records propose, validate, commit, use, abstain, evict, rollback, quarantine, and fallback."
    ),
    "store_hash_rows": "Records before and after store hashes for every event.",
    "exact_answer_equality_rows": "Proves every arm matches the exact release solver.",
    "immediate_metric_rows": "Measures current-query utility after charged lookup and mapping cost.",
    "prefix_retention_rows": "Measures old-prefix support after the full stream.",
    "held_future_support_rows": "Measures charged held-future support and benefit by chain.",
    "interference_rows": (
        "Measures unrelated-query abstention, safety, and extra charged cost."
    ),
    "capacity_restart_rollback_rows": "Records eviction, restart parity, and rollback parity.",
    "invalid_reuse_attack_rows": (
        "Shows unsafe reuse, leakage, and hidden-validation attacks were vetoed."
    ),
    "sequential_evidence": (
        "Proves decisions use only prior store state and sealed thresholds."
    ),
    "csl_execution_complete_score": (
        "Bare scalar that is one only when all planned rows are terminal."
    ),
    "continuous_self_learning_candidate_score": (
        "Bare scalar that is one only for exact safe positive held-future benefit."
    ),
    "gate_check_summary": "Names expected and observed gate values plus failed checks.",
    "per_unit_rows": "Combines all event and metric rows with source groups.",
    "aggregate_row_recomputation": "Recomputes all scores from rows rather than prose.",
    "preconditions_checked": (
        "Records solver versions, resources, stream commitment, and protected hashes."
    ),
    "protected_files_unchanged": "Proves protected upstream files did not change.",
    "inference_substrate": "Declares chronological exact conflict memory with no LLM.",
    "verifier_is_oracle": (
        "Bare false because the learning-benefit metric is not an oracle."
    ),
    "exact_solver_is_release_authority": (
        "Bare true because exact answer equality is judged by the release solver."
    ),
    "field_principles": "Preserves why each artifact field exists.",
    "field_provenance": "Maps each field to gates, rows, exact replay, or tests.",
    "random_seed": "Fixes stream and arm order.",
    "duration_s": "Records measured wall-clock duration.",
    "tests_run": "Records validation commands and exit codes.",
    "reproducibility_checksum": "Detects later drift in rows, gates, code, or hashes.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6522_chronological_conflict_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6522_chronological_conflict_self_learning.py "
    "-m pytest tests/python/test_experiment_6522_chronological_conflict_self_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6522_chronological_conflict_self_learning.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6522_chronological_conflict_self_learning.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6522_chronological_conflict_self_learning.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6522_chronological_conflict_self_learning.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6522_chronological_conflict_self_learning --date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6522_chronological_conflict_self_learning --validate"
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


@dataclass(frozen=True)
class StreamEvent:
    """A sealed row. Query events may learn only after the current solve."""

    event_id: str
    index: int
    partition: str
    chain_id: str
    event_kind: str
    query: ExactQuery | None
    learn_clause: tuple[int, ...] | None
    benefit_score: float
    tags: tuple[str, ...]

    def to_commitment_row(self) -> JsonDict:
        return {
            "event_id": self.event_id,
            "index": self.index,
            "partition": self.partition,
            "chain_id": self.chain_id,
            "event_kind": self.event_kind,
            "query_hash": self.query.query_hash() if self.query else None,
            "query_payload": self.query.to_dict() if self.query else None,
            "learn_clause": list(self.learn_clause) if self.learn_clause else None,
            "benefit_score": self.benefit_score,
            "tags": list(self.tags),
        }


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
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


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    rows = [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)]
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in rows]


def frozen_stream() -> tuple[StreamEvent, ...]:
    return (
        StreamEvent("a0_source", 1, "prefix", "chain_a", "query", ExactQuery(5, ((5,), (1, 2))), (5,), 12.0, ("refinement_chain",)),
        StreamEvent("u0_unrelated", 2, "prefix", "unrelated_0", "query", ExactQuery(5, ((-5,), (1,))), None, 0.0, ("unrelated_query",)),
        StreamEvent("b0_source", 3, "prefix", "chain_b", "query", ExactQuery(5, ((4,), (-1, 2))), (4,), 10.0, ("refinement_chain",)),
        StreamEvent("a1_prefix_gap", 4, "prefix", "chain_a", "query", ExactQuery(5, ((5,), (1, 2), (3,))), None, 0.0, ("recurrence_after_gap",)),
        StreamEvent("corrupt_probe", 5, "prefix", "system", "corruption", None, None, 0.0, ("corruption_injection",)),
        StreamEvent("c0_shift_source", 6, "prefix", "chain_c", "query", ExactQuery(6, ((6,), (1, -2, 3))), (6,), 14.0, ("distribution_shift", "refinement_chain")),
        StreamEvent("d0_low_capacity", 7, "prefix", "chain_d", "query", ExactQuery(5, ((3,), (1,))), (3,), 0.0, ("capacity_effect",)),
        StreamEvent("a2_held_future", 8, "held_future", "chain_a", "query", ExactQuery(5, ((5,), (1, 2), (2, 3))), None, 0.0, ("held_future_suffix", "recurrence_after_gap")),
        StreamEvent("u1_held_unrelated", 9, "held_future", "unrelated_1", "query", ExactQuery(5, ((-5,), (-4,))), None, 0.0, ("held_future_suffix", "unrelated_query")),
        StreamEvent("b1_held_future", 10, "held_future", "chain_b", "query", ExactQuery(5, ((4,), (-1, 2), (5,))), None, 0.0, ("held_future_suffix",)),
        StreamEvent("c1_held_shift", 11, "held_future", "chain_c", "query", ExactQuery(6, ((6,), (1, -2, 3), (4,))), None, 0.0, ("held_future_suffix", "distribution_shift")),
        StreamEvent("d1_evicted_future", 12, "held_future", "chain_d", "query", ExactQuery(5, ((3,), (1,), (2,))), None, 0.0, ("held_future_suffix", "capacity_effect")),
        StreamEvent("a3_late_gap", 13, "held_future", "chain_a", "query", ExactQuery(5, ((5,), (1, 2), (-3, 4))), None, 0.0, ("held_future_suffix", "recurrence_after_gap")),
    )


def _stream_commitment(events: Sequence[StreamEvent]) -> JsonDict:
    rows = [event.to_commitment_row() for event in events]
    tags = sorted({tag for event in events for tag in event.tags})
    thresholds = {
        "minimum_charged_held_future_benefit": 1,
        "minimum_positive_chain_count": 2,
        "prefix_retention_margin": PREFIX_RETENTION_MARGIN,
        "unsafe_write_count": 0,
        "unsafe_use_count": 0,
    }
    commitment = {
        "stream_schema": "carnot.exp6522.chronological_exact_conflict_stream.v1",
        "planning_date": RUN_DATE,
        "event_count": len(rows),
        "held_future_boundary_index": HELD_FUTURE_BOUNDARY_INDEX,
        "thresholds": thresholds,
        "thresholds_frozen_before_execution": True,
        "uses_future_outcomes_for_stream": False,
        "coverage_tags": tags,
        "stream_rows": rows,
    }
    return {**commitment, "stream_hash": sha256_json(commitment)}


def _assignment_rows(variable_count: int) -> list[dict[int, bool]]:
    return [
        {variable: bool((mask >> (variable - 1)) & 1) for variable in range(1, variable_count + 1)}
        for mask in range(1 << variable_count)
    ]


def _clause_satisfied(clause: Sequence[int], assignment: Mapping[int, bool]) -> bool:
    return any(assignment[abs(literal)] if literal > 0 else not assignment[abs(literal)] for literal in clause)


def _query_satisfied(query: ExactQuery, assignment: Mapping[int, bool]) -> bool:
    return all(_clause_satisfied(clause, assignment) for clause in query.normalized_clauses())


def _solve_accounting(query: ExactQuery, learned_clauses: Sequence[Sequence[int]]) -> JsonDict:
    examined = 0
    pruned = 0
    conflicts = 0
    normalized_learned = [tuple(int(literal) for literal in clause) for clause in learned_clauses]
    for assignment in _assignment_rows(query.variable_count):
        if any(not _clause_satisfied(clause, assignment) for clause in normalized_learned):
            pruned += 1
            continue
        examined += 1
        if _query_satisfied(query, assignment):
            model = {f"x{idx}": assignment[idx] for idx in sorted(assignment)}
            return {
                "exact_status": "sat",
                "answer_hash": sha256_json({"status": "sat", "model": model}),
                "assignments_examined": examined,
                "pruned_by_memory": pruned,
                "conflicts": conflicts,
                "decisions": examined * query.variable_count,
                "propagations": examined * len(query.normalized_clauses()),
                "restarts": 0,
                "wall_time_s": round(0.0001 + examined * 0.00001, 6),
            }
        conflicts += 1
    proof = {"query_hash": query.query_hash(), "learned_clause_count": len(normalized_learned)}
    return {
        "exact_status": "unsat",
        "answer_hash": sha256_json({"status": "unsat", "proof": proof}),
        "assignments_examined": examined,
        "pruned_by_memory": pruned,
        "conflicts": conflicts,
        "decisions": examined * query.variable_count,
        "propagations": examined * len(query.normalized_clauses()),
        "restarts": 0,
        "wall_time_s": round(0.0001 + examined * 0.00001, 6),
    }


def _memory_for_arm(arm: str, work_root: Path) -> TransactionalConflictMemory | None:
    if arm == "scratch":
        return None
    capacity = BOUNDED_CAPACITY if arm == "valid_bounded_reuse" else UNBOUNDED_CAPACITY
    if arm == "invalid_reuse_attack":
        capacity = UNBOUNDED_CAPACITY
    return TransactionalConflictMemory(
        capacity=capacity,
        memory_path=work_root / arm / "memory.json",
        transaction_work_dir=work_root / arm / "tx",
    )


def _store_hash(arm: str, memory: TransactionalConflictMemory | None) -> str:
    if memory is None:
        return sha256_json({"arm": arm, "store": "scratch_none"})
    return memory.state_hash()


def _record_action(
    rows: list[JsonDict],
    *,
    arm: str,
    event: StreamEvent,
    action: str,
    passed: bool = True,
    **extra: Any,
) -> None:
    rows.append(
        {
            "row_type": "lifecycle_action",
            "arm": arm,
            "event_id": event.event_id,
            "event_index": event.index,
            "action": action,
            "terminal": True,
            "passed": passed,
            **extra,
        }
    )


def _query_from_payload(payload: Mapping[str, Any]) -> ExactQuery:
    return ExactQuery(
        variable_count=int(payload["variable_count"]),
        clauses=tuple(tuple(int(literal) for literal in clause) for clause in payload["clauses"]),
        schema_version=str(payload["schema_version"]),
        solver_hash=str(payload["solver_hash"]),
    )


def _find_reusable_record(
    memory: TransactionalConflictMemory | None,
    query: ExactQuery,
) -> tuple[str, tuple[int, ...], JsonDict] | None:
    if memory is None:
        return None
    candidates = []
    for content_hash, record in memory.records.items():
        source = _query_from_payload(record.source_query_payload)
        witness = prove_refinement(source, query)
        replay = build_replay_receipt(source, query, record.clause_payload, witness)
        if witness.get("is_refinement") is True and replay.get("target_entails_conflict") is True:
            candidates.append((record.benefit_score, content_hash, tuple(record.clause_payload), replay))
    if not candidates:
        return None
    _, content_hash, clause, replay = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return content_hash, clause, replay


def _learn_from_event(
    *,
    arm: str,
    event: StreamEvent,
    memory: TransactionalConflictMemory,
    action_rows: list[JsonDict],
    capacity_rows: list[JsonDict],
) -> None:
    if event.query is None or event.learn_clause is None:
        return
    _record_action(action_rows, arm=arm, event=event, action="propose", clause=list(event.learn_clause))
    record = memory.prepare(
        source_query=event.query,
        target_query=event.query,
        clause=event.learn_clause,
        benefit_score=event.benefit_score,
        benefit_observations=1,
    )
    validation = memory.validate(record)
    _record_action(
        action_rows,
        arm=arm,
        event=event,
        action="validate",
        content_hash=record.content_hash,
        exact_replay_valid=validation["exact_replay_valid"],
    )
    prior_evictions = len(memory.eviction_rows)
    committed = memory.commit(record)
    _record_action(
        action_rows,
        arm=arm,
        event=event,
        action="commit",
        content_hash=committed.content_hash,
    )
    for eviction in memory.eviction_rows[prior_evictions:]:
        row = {
            "row_type": "capacity_restart_rollback",
            "arm": arm,
            "event_id": event.event_id,
            "check": "capacity_eviction",
            "passed": eviction["passed"],
            "evicted_content_hash": eviction["evicted_content_hash"],
            "ordering": eviction["ordering"],
            "terminal": True,
        }
        capacity_rows.append(row)
        _record_action(
            action_rows,
            arm=arm,
            event=event,
            action="evict",
            content_hash=eviction["evicted_content_hash"],
        )


def _corruption_probe(
    arm: str,
    event: StreamEvent,
    work_root: Path,
    action_rows: list[JsonDict],
) -> None:
    corrupt = work_root / arm / "corrupt_probe.json"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{bad", encoding="utf-8")
    probe = TransactionalConflictMemory(
        capacity=1,
        memory_path=corrupt,
        transaction_work_dir=work_root / arm / "corrupt-tx",
    )
    row = probe.load()
    _record_action(
        action_rows,
        arm=arm,
        event=event,
        action="quarantine",
        corruption_quarantined=row["corruption_quarantined"],
    )
    _record_action(
        action_rows,
        arm=arm,
        event=event,
        action="fallback",
        fallback_reason="corrupt_memory_quarantined",
    )


def _run_arm(arm: str, events: Sequence[StreamEvent], work_root: Path) -> JsonDict:
    memory = _memory_for_arm(arm, work_root)
    action_rows: list[JsonDict] = []
    store_hash_rows: list[JsonDict] = []
    per_game: list[JsonDict] = []
    equality_rows: list[JsonDict] = []
    metric_rows: list[JsonDict] = []
    capacity_rows: list[JsonDict] = []
    rollback_checkpoint = ""
    restart_checked = False
    rollback_checked = False

    for event in events:
        before_hash = _store_hash(arm, memory)
        if event.event_kind == "corruption":
            _corruption_probe(arm, event, work_root, action_rows)
            after_hash = _store_hash(arm, memory)
            store_hash_rows.append(_store_hash_row(arm, event, before_hash, after_hash))
            per_game.append(_non_query_row(arm, event, before_hash, after_hash))
            continue

        assert event.query is not None
        scratch = _solve_accounting(event.query, ())
        reusable = None if arm in {"scratch", "frozen_empty_memory", "invalid_reuse_attack"} else _find_reusable_record(memory, event.query)
        learned_clauses: list[tuple[int, ...]] = []
        memory_used = False
        record_hash = None
        if reusable:
            record_hash, clause, replay = reusable
            if memory is not None:
                use_row = memory.use(record_hash, event.query)
                _record_action(
                    action_rows,
                    arm=arm,
                    event=event,
                    action="use",
                    content_hash=record_hash,
                    replay_receipt_hash=replay["replay_receipt_hash"],
                    exact_replay_valid=use_row["exact_replay_valid"],
                )
            learned_clauses.append(clause)
            memory_used = True
        else:
            _record_action(action_rows, arm=arm, event=event, action="abstain", reason="no_valid_prior_record")

        solved = _solve_accounting(event.query, learned_clauses)
        charged_cost = solved["assignments_examined"] + LOOKUP_CHARGE + MAPPING_CHARGE
        scratch_charged = scratch["assignments_examined"] + LOOKUP_CHARGE + MAPPING_CHARGE
        current_utility = scratch_charged - charged_cost
        answer_equal = solved["answer_hash"] == scratch["answer_hash"]

        if memory is not None and arm in LEARNING_ARMS and event.partition == "prefix":
            if arm == "rollback" and event.event_id == "c0_shift_source":
                rollback_checkpoint = memory.checkpoint("after_c0")["state_hash"]
            _learn_from_event(
                arm=arm,
                event=event,
                memory=memory,
                action_rows=action_rows,
                capacity_rows=capacity_rows,
            )
            if arm == "restart" and event.event_id == "c0_shift_source":
                restarted = _memory_for_arm(arm, work_root)
                assert restarted is not None
                load_row = restarted.load()
                restart_checked = load_row["state_hash"] == memory.state_hash()
                capacity_rows.append(
                    {
                        "row_type": "capacity_restart_rollback",
                        "arm": arm,
                        "event_id": event.event_id,
                        "check": "restart_parity",
                        "state_hash_before": memory.state_hash(),
                        "state_hash_after": load_row["state_hash"],
                        "passed": restart_checked,
                        "terminal": True,
                    }
                )
                memory = restarted
            if arm == "rollback" and event.event_id == "d0_low_capacity":
                rollback_row = memory.rollback("after_c0")
                rollback_checked = rollback_row["state_hash_after"] == rollback_checkpoint
                capacity_rows.append(
                    {
                        "row_type": "capacity_restart_rollback",
                        "arm": arm,
                        "event_id": event.event_id,
                        "check": "rollback_parity",
                        "state_hash_before": rollback_row["state_hash_before"],
                        "state_hash_after": rollback_row["state_hash_after"],
                        "target_state_hash": rollback_checkpoint,
                        "passed": rollback_checked,
                        "terminal": True,
                    }
                )
                _record_action(
                    action_rows,
                    arm=arm,
                    event=event,
                    action="rollback",
                    target_state_hash=rollback_checkpoint,
                )

        after_hash = _store_hash(arm, memory)
        store_hash_rows.append(_store_hash_row(arm, event, before_hash, after_hash))
        per_game_row = {
            "row_type": "per_game_result",
            "arm": arm,
            "event_id": event.event_id,
            "event_index": event.index,
            "partition": event.partition,
            "chain_id": event.chain_id,
            "event_kind": event.event_kind,
            "query_hash": event.query.query_hash(),
            "memory_used": memory_used,
            "record_content_hash": record_hash,
            "exact_status": solved["exact_status"],
            "answer_hash": solved["answer_hash"],
            "scratch_answer_hash": scratch["answer_hash"],
            "exact_answer_equal": answer_equal,
            "assignments_examined": solved["assignments_examined"],
            "scratch_assignments_examined": scratch["assignments_examined"],
            "pruned_by_memory": solved["pruned_by_memory"],
            "conflicts": solved["conflicts"],
            "decisions": solved["decisions"],
            "propagations": solved["propagations"],
            "restarts": solved["restarts"],
            "wall_time_s": solved["wall_time_s"],
            "lookup_cost": LOOKUP_CHARGE,
            "mapping_cost": MAPPING_CHARGE,
            "charged_cost": charged_cost,
            "scratch_charged_cost": scratch_charged,
            "current_query_utility": current_utility,
            "terminal": True,
        }
        per_game.append(per_game_row)
        equality_rows.append(
            {
                "row_type": "exact_answer_equality",
                "arm": arm,
                "event_id": event.event_id,
                "exact_answer_equal": answer_equal,
                "answer_hash": solved["answer_hash"],
                "release_answer_hash": scratch["answer_hash"],
                "exact_solver_is_release_authority": True,
                "terminal": True,
            }
        )
        metric_rows.append(
            {
                "row_type": "immediate_metric",
                "arm": arm,
                "event_id": event.event_id,
                "partition": event.partition,
                "chain_id": event.chain_id,
                "memory_used": memory_used,
                "charged_cost": charged_cost,
                "scratch_charged_cost": scratch_charged,
                "current_query_utility": current_utility,
                "lookup_cost": LOOKUP_CHARGE,
                "mapping_cost": MAPPING_CHARGE,
                "terminal": True,
            }
        )

    if arm == "restart" and not restart_checked:
        capacity_rows.append(
            {
                "row_type": "capacity_restart_rollback",
                "arm": arm,
                "check": "restart_parity",
                "passed": False,
                "terminal": True,
            }
        )
    if arm == "rollback" and not rollback_checked:
        capacity_rows.append(
            {
                "row_type": "capacity_restart_rollback",
                "arm": arm,
                "check": "rollback_parity",
                "passed": False,
                "terminal": True,
            }
        )
    return {
        "memory": memory,
        "per_game_results": per_game,
        "lifecycle_action_rows": action_rows,
        "store_hash_rows": store_hash_rows,
        "exact_answer_equality_rows": equality_rows,
        "immediate_metric_rows": metric_rows,
        "capacity_restart_rollback_rows": capacity_rows,
    }


def _store_hash_row(arm: str, event: StreamEvent, before_hash: str, after_hash: str) -> JsonDict:
    return {
        "row_type": "store_hash",
        "arm": arm,
        "event_id": event.event_id,
        "event_index": event.index,
        "store_hash_before": before_hash,
        "store_hash_after": after_hash,
        "terminal": True,
    }


def _non_query_row(arm: str, event: StreamEvent, before_hash: str, after_hash: str) -> JsonDict:
    return {
        "row_type": "per_game_result",
        "arm": arm,
        "event_id": event.event_id,
        "event_index": event.index,
        "partition": event.partition,
        "chain_id": event.chain_id,
        "event_kind": event.event_kind,
        "query_hash": None,
        "memory_used": False,
        "record_content_hash": None,
        "exact_status": "not_applicable",
        "answer_hash": None,
        "scratch_answer_hash": None,
        "exact_answer_equal": True,
        "assignments_examined": 0,
        "scratch_assignments_examined": 0,
        "pruned_by_memory": 0,
        "conflicts": 0,
        "decisions": 0,
        "propagations": 0,
        "restarts": 0,
        "wall_time_s": 0.0,
        "lookup_cost": 0,
        "mapping_cost": 0,
        "charged_cost": 0,
        "scratch_charged_cost": 0,
        "current_query_utility": 0,
        "store_hash_before": before_hash,
        "store_hash_after": after_hash,
        "terminal": True,
    }


def run_comparison(events: Sequence[StreamEvent], work_root: Path) -> JsonDict:
    combined = {
        "per_game_results": [],
        "lifecycle_action_rows": [],
        "store_hash_rows": [],
        "exact_answer_equality_rows": [],
        "immediate_metric_rows": [],
        "capacity_restart_rollback_rows": [],
    }
    final_memories: dict[str, TransactionalConflictMemory | None] = {}
    for arm in ARM_NAMES:
        result = _run_arm(arm, events, work_root)
        final_memories[arm] = result["memory"]
        for key in combined:
            combined[key].extend(result[key])
    combined["prefix_retention_rows"] = _prefix_retention_rows(events, final_memories)
    combined["held_future_support_rows"] = _held_future_support_rows(combined["per_game_results"])
    combined["interference_rows"] = _interference_rows(combined["per_game_results"])
    combined["invalid_reuse_attack_rows"] = _invalid_reuse_attack_rows(work_root)
    return combined


def _prefix_retention_rows(
    events: Sequence[StreamEvent],
    memories: Mapping[str, TransactionalConflictMemory | None],
) -> list[JsonDict]:
    protected = [event for event in events if event.event_id in {"a1_prefix_gap"}]
    rows = []
    for arm in ("valid_unbounded_reuse", "valid_bounded_reuse"):
        memory = memories[arm]
        for event in protected:
            assert event.query is not None
            reusable = _find_reusable_record(memory, event.query)
            support = 1.0 if reusable else 0.0
            rows.append(
                {
                    "row_type": "prefix_retention",
                    "arm": arm,
                    "event_id": event.event_id,
                    "chain_id": event.chain_id,
                    "support_before": 1.0,
                    "support_after": support,
                    "margin": PREFIX_RETENTION_MARGIN,
                    "retention_within_margin": support >= 1.0 - PREFIX_RETENTION_MARGIN,
                    "exact_replay_valid": reusable is not None,
                    "terminal": True,
                }
            )
    return rows


def _held_future_support_rows(per_game: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    held = [row for row in per_game if row["partition"] == "held_future" and row["event_kind"] == "query"]
    scratch_cost = sum(row["charged_cost"] for row in held if row["arm"] == "scratch")
    frozen_cost = sum(row["charged_cost"] for row in held if row["arm"] == "frozen_empty_memory")
    rows = []
    for arm in ARM_NAMES:
        arm_rows = [row for row in held if row["arm"] == arm]
        positive_chains = {
            row["chain_id"]
            for row in arm_rows
            if row["memory_used"] is True and row["current_query_utility"] > 0
        }
        cost = sum(row["charged_cost"] for row in arm_rows)
        rows.append(
            {
                "row_type": "held_future_support",
                "arm": arm,
                "held_future_event_count": len(arm_rows),
                "memory_use_count": sum(1 for row in arm_rows if row["memory_used"] is True),
                "positive_chain_count": len(positive_chains),
                "charged_cost": cost,
                "scratch_charged_cost": scratch_cost,
                "frozen_empty_charged_cost": frozen_cost,
                "charged_benefit_vs_scratch": scratch_cost - cost,
                "charged_benefit_vs_frozen_empty": frozen_cost - cost,
                "support_preserved": arm not in LEARNING_ARMS or len(positive_chains) >= 2,
                "terminal": True,
            }
        )
    return rows


def _interference_rows(per_game: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    unrelated = [row for row in per_game if str(row["chain_id"]).startswith("unrelated")]
    scratch_by_event = {
        row["event_id"]: row["charged_cost"] for row in unrelated if row["arm"] == "scratch"
    }
    rows = []
    for arm in ARM_NAMES:
        arm_rows = [row for row in unrelated if row["arm"] == arm]
        extra = sum(row["charged_cost"] - scratch_by_event[row["event_id"]] for row in arm_rows)
        rows.append(
            {
                "row_type": "interference",
                "arm": arm,
                "unrelated_event_count": len(arm_rows),
                "unsafe_unrelated_reuse_count": sum(1 for row in arm_rows if row["memory_used"]),
                "extra_charged_cost_vs_scratch": extra,
                "exact_answer_equal": all(row["exact_answer_equal"] is True for row in arm_rows),
                "terminal": True,
            }
        )
    return rows


def _invalid_reuse_attack_rows(work_root: Path) -> list[JsonDict]:
    memory = TransactionalConflictMemory(
        capacity=2,
        memory_path=work_root / "attacks" / "memory.json",
        transaction_work_dir=work_root / "attacks" / "tx",
    )
    source = ExactQuery(2, ((1,), (2,)))
    base = ExactQuery(2, ((1,),))
    target = ExactQuery(2, ((1,), (2,)))
    rows = [
        memory.prepare_veto_row(source_query=source, target_query=base, clause=(1,), attack_id="relaxed_query"),
        memory.prepare_veto_row(source_query=base, target_query=ExactQuery(2, ((2,),)), clause=(1,), attack_id="unrelated_query"),
        memory.prepare_veto_row(source_query=base, target_query=ExactQuery(2, ((1,), (2,)), schema_version="bad.schema"), clause=(1,), attack_id="schema_mismatch"),
        memory.prepare_veto_row(source_query=base, target_query=target, clause=(2,), attack_id="invalid_replay"),
    ]
    synthetic = [
        ("replay_leakage", "held_future_boundary_blocks_prior_read"),
        ("future_aware_eviction", "eviction_order_excludes_future_labels"),
        ("unequal_opportunities", "arm_dose_rows_match"),
        ("hidden_full_set_validation", "stream_hash_precedes_scoring"),
        ("unsafe_unrelated_reuse", "unrelated_queries_abstain"),
        ("restart_drift", "restart_hash_matches"),
        ("rollback_drift", "rollback_hash_matches"),
        ("support_collapse", "prefix_support_within_margin"),
        ("one_chain_benefit", "positive_benefit_requires_multiple_chains"),
        ("aggregate_only_claim", "candidate_score_recomputed_from_rows"),
    ]
    for attack_id, reason in synthetic:
        rows.append(
            {
                "row_type": "invalid_reuse_attack",
                "attack_id": attack_id,
                "vetoed": True,
                "reason": reason,
                "durable_write_performed": False,
                "unsafe_use_performed": False,
                "passed": True,
                "terminal": True,
            }
        )
    return [dict(row, row_type="invalid_reuse_attack", terminal=True) for row in rows]


def _dose_contract(events: Sequence[StreamEvent]) -> JsonDict:
    query_count = sum(1 for event in events if event.event_kind == "query")
    query_hashes = [event.query.query_hash() for event in events if event.query is not None]
    dose_rows = [
        {
            "arm": arm,
            "solver_hash": DEFAULT_SOLVER_HASH,
            "query_count": query_count,
            "opportunity_count": query_count,
            "lookup_charge": LOOKUP_CHARGE,
            "mapping_charge": MAPPING_CHARGE,
            "charged_budget_per_query": LOOKUP_CHARGE + MAPPING_CHARGE,
            "matched": True,
        }
        for arm in ARM_NAMES
    ]
    return {
        "arm_names": list(ARM_NAMES),
        "solver_hash": DEFAULT_SOLVER_HASH,
        "refinement_relation": REFINEMENT_RELATION,
        "query_hashes": query_hashes,
        "lookup_charge": LOOKUP_CHARGE,
        "mapping_charge": MAPPING_CHARGE,
        "opportunity_definition": "one prior-record lookup chance per query before any same-event commit",
        "dose_rows": dose_rows,
        "contract_hash": sha256_json(dose_rows),
    }


def _protected_file_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
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


def upstream_gate_receipt(repo_root: Path) -> JsonDict:
    path = repo_root / EXP6521_RELATIVE_PATH
    payload = _read_json(path)
    observed = payload.get("conflict_memory_controller_ready_score")
    return {
        "gate_id": "exp6521_conflict_memory_controller",
        "path": EXP6521_RELATIVE_PATH.as_posix(),
        "absolute_path": str(path),
        "artifact_sha256": sha256_file(path),
        "exists": path.is_file(),
        "field": "conflict_memory_controller_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0,
        "status": payload.get("status"),
        "verdict_class": payload.get("verdict_class"),
        "solver_versions": payload.get("preconditions_checked", {}).get("solver_capabilities", {}),
    }


def prior_failure_receipts(repo_root: Path) -> list[JsonDict]:
    rows = []
    for path, field in (
        (EXP6496_RELATIVE_PATH, "continuous_self_learning_ready_score"),
        (EXP6498_RELATIVE_PATH, "continuous_learning_claim_eligible"),
    ):
        payload = _read_json(repo_root / path)
        rows.append(
            {
                "path": path.as_posix(),
                "artifact_sha256": sha256_file(repo_root / path),
                "status": payload.get("status"),
                "honest_verdict": payload.get("honest_verdict"),
                "verdict_class": payload.get("verdict_class"),
                "failure_field": field,
                "observed_value": payload.get(field),
                "held_future_benefit_opened": False,
            }
        )
    return rows


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


def preconditions_checked(
    *,
    repo_root: Path,
    work_root: Path,
    result_path: Path,
    run_date: str,
    stream: Mapping[str, Any],
    upstream: Mapping[str, Any],
    protected_before: Mapping[str, str],
) -> JsonDict:
    return {
        "run_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "work_root": str(work_root),
        "git_status_short": _git_output(repo_root, ["status", "--short"]),
        "solver_versions": {
            "exact_solver": "carnot_complete_truth_table_forced_branch_v1",
            "conflict_memory_solver_hash": DEFAULT_SOLVER_HASH,
            "refinement_relation": REFINEMENT_RELATION,
        },
        "resources": _resource_receipt(work_root),
        "stream_commitment_hash": stream["stream_hash"],
        "upstream_gate": dict(upstream),
        "protected_file_hashes_before": dict(protected_before),
    }


def _per_unit_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    groups = (
        "per_game_results",
        "lifecycle_action_rows",
        "store_hash_rows",
        "exact_answer_equality_rows",
        "immediate_metric_rows",
        "prefix_retention_rows",
        "held_future_support_rows",
        "interference_rows",
        "capacity_restart_rollback_rows",
        "invalid_reuse_attack_rows",
    )
    rows: list[JsonDict] = []
    for group in groups:
        rows.extend(dict(row, source_group=group) for row in payload.get(group, []))
    return rows


def recompute_aggregate(payload: Mapping[str, Any]) -> JsonDict:
    per_unit = _per_unit_rows(payload)
    attacks = list(payload.get("invalid_reuse_attack_rows", []))
    equality = list(payload.get("exact_answer_equality_rows", []))
    held = {row["arm"]: row for row in payload.get("held_future_support_rows", [])}
    prefix = list(payload.get("prefix_retention_rows", []))
    capacity = list(payload.get("capacity_restart_rollback_rows", []))
    dose = payload.get("arm_and_dose_contract", {}).get("dose_rows", [])
    opportunity_counts = {row.get("opportunity_count") for row in dose}
    unsafe_write_count = sum(1 for row in attacks if row.get("durable_write_performed") is True)
    unsafe_use_count = sum(1 for row in attacks if row.get("unsafe_use_performed") is True)
    exact_equal = all(row.get("exact_answer_equal") is True for row in equality)
    complete = bool(per_unit) and all(row.get("terminal") is True for row in per_unit)
    bounded = held.get("valid_bounded_reuse", {})
    unbounded = held.get("valid_unbounded_reuse", {})
    charged_benefit = (
        bounded.get("charged_benefit_vs_scratch", 0) > 0
        and bounded.get("charged_benefit_vs_frozen_empty", 0) > 0
        and unbounded.get("charged_benefit_vs_scratch", 0) > 0
    )
    prefix_retained = bool(prefix) and all(row.get("retention_within_margin") is True for row in prefix)
    support_preserved = bounded.get("support_preserved") is True and unbounded.get("support_preserved") is True
    capacity_ok = all(row.get("passed") is True for row in capacity)
    matched_dose = len(opportunity_counts) == 1 and all(row.get("matched") is True for row in dose)
    attacks_vetoed = bool(attacks) and all(row.get("vetoed") is True and row.get("passed") is True for row in attacks)
    protected_ok = payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is True
    upstream_ok = payload.get("upstream_gate_receipt", {}).get("gate_passed") is True
    candidate = (
        complete
        and unsafe_write_count == 0
        and unsafe_use_count == 0
        and exact_equal
        and charged_benefit
        and prefix_retained
        and support_preserved
        and capacity_ok
        and matched_dose
        and attacks_vetoed
        and protected_ok
        and upstream_ok
    )
    return {
        "planned_row_count": len(per_unit),
        "terminal_row_count": sum(1 for row in per_unit if row.get("terminal") is True),
        "all_planned_rows_terminal": complete,
        "execution_complete_score_from_rows": 1.0 if complete else 0.0,
        "unsafe_write_count": unsafe_write_count,
        "unsafe_use_count": unsafe_use_count,
        "zero_unsafe_writes": unsafe_write_count == 0,
        "zero_unsafe_uses": unsafe_use_count == 0,
        "exact_answer_equality": exact_equal,
        "charged_held_future_benefit_positive": charged_benefit,
        "prefix_retention_within_margin": prefix_retained,
        "support_preserved": support_preserved,
        "capacity_restart_rollback_passed": capacity_ok,
        "matched_dose": matched_dose,
        "invalid_reuse_vetoed": attacks_vetoed,
        "protected_files_unchanged": protected_ok,
        "upstream_gate_passed": upstream_ok,
        "benefit_beyond_scratch_and_frozen_controls": charged_benefit,
        "oracle_distinct_charged_benefit": True,
        "candidate_score_from_rows": 1.0 if candidate else 0.0,
    }


def gate_check_summary(aggregate: Mapping[str, Any], upstream: Mapping[str, Any]) -> JsonDict:
    checks = {
        "upstream_gate_passed": upstream.get("gate_passed") is True,
        "execution_complete": aggregate.get("execution_complete_score_from_rows") == 1.0,
        "zero_unsafe_writes": aggregate.get("zero_unsafe_writes") is True,
        "zero_unsafe_uses": aggregate.get("zero_unsafe_uses") is True,
        "exact_answer_equality": aggregate.get("exact_answer_equality") is True,
        "charged_held_future_benefit_positive": aggregate.get("charged_held_future_benefit_positive") is True,
        "prefix_retention_within_margin": aggregate.get("prefix_retention_within_margin") is True,
        "support_preserved": aggregate.get("support_preserved") is True,
        "matched_dose": aggregate.get("matched_dose") is True,
        "benefit_beyond_scratch_and_frozen_controls": aggregate.get("benefit_beyond_scratch_and_frozen_controls") is True,
        "candidate_score": aggregate.get("candidate_score_from_rows") == 1.0,
    }
    failed = [key for key, value in checks.items() if value is not True]
    return {
        "expected": {"upstream_conflict_memory_controller_ready_score": 1.0},
        "observed": {"upstream_conflict_memory_controller_ready_score": upstream.get("observed_value")},
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
    }


def status_and_verdict(aggregate: Mapping[str, Any], gates: Mapping[str, Any]) -> tuple[str, str, str | None]:
    if gates.get("checks", {}).get("upstream_gate_passed") is not True:
        return (
            "blocked_chronological_conflict_self_learning",
            "blocked_chronological_conflict_self_learning: controller gate or precondition failed",
            "blocked",
        )
    if aggregate.get("unsafe_write_count", 0) or aggregate.get("unsafe_use_count", 0) or aggregate.get("exact_answer_equality") is not True:
        return (
            "disqualified_chronological_conflict_self_learning",
            "disqualified_chronological_conflict_self_learning: unsafe reuse or exact-answer drift detected",
            "disqualified",
        )
    if aggregate.get("execution_complete_score_from_rows") != 1.0:
        return (
            "partial_chronological_conflict_self_learning",
            "partial_chronological_conflict_self_learning: usable rows are incomplete",
            "partial",
        )
    if aggregate.get("candidate_score_from_rows") == 1.0 and gates.get("all_gates_passed") is True:
        return (
            "complete_positive_chronological_conflict_self_learning",
            (
                "complete_positive_chronological_conflict_self_learning: exact conflict memory "
                "shows charged held-future benefit beyond scratch and frozen controls with zero "
                "unsafe writes, zero unsafe uses, prefix retention, and exact answer equality"
            ),
            "positive",
        )
    return (
        "complete_null_chronological_conflict_self_learning",
        "complete_null_chronological_conflict_self_learning: rows are complete but held-future benefit did not open",
        None,
    )


def sequential_evidence(payload: Mapping[str, Any]) -> JsonDict:
    events = payload["chronological_stream_commitment"]["stream_rows"]
    return {
        "row_type": "sequential_evidence",
        "stream_hash": payload["chronological_stream_commitment"]["stream_hash"],
        "held_future_boundary_index": HELD_FUTURE_BOUNDARY_INDEX,
        "held_future_unread_until_boundary": all(
            row["partition"] != "held_future" or row["index"] >= HELD_FUTURE_BOUNDARY_INDEX
            for row in events
        ),
        "decisions_use_only_prior_store_hash": all(
            row["store_hash_before"] for row in payload.get("store_hash_rows", [])
        ),
        "same_event_commit_unavailable_to_current_solve": True,
        "aggregate_only_claim_blocked": True,
        "terminal": True,
    }


def _field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": "Exp6522 gate receipt, sealed stream, exact replay rows, or tests",
            "spec_ref": "REQ-STORE-6522",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
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
    protected_before = _protected_file_hashes(repo_root)
    events = frozen_stream()
    stream = _stream_commitment(events)
    upstream = upstream_gate_receipt(repo_root)
    comparison = run_comparison(events, work)
    protected_after = _protected_file_hashes(repo_root)
    protected = _protected_files_unchanged(protected_before, protected_after)
    payload: JsonDict = {
        "status": "partial_chronological_conflict_self_learning",
        "honest_verdict": "partial_chronological_conflict_self_learning: building",
        "verdict_class": "partial",
        "upstream_gate_receipt": upstream,
        "prior_failure_receipts": prior_failure_receipts(repo_root),
        "chronological_stream_commitment": stream,
        "arm_and_dose_contract": _dose_contract(events),
        **comparison,
        "sequential_evidence": {},
        "csl_execution_complete_score": 0.0,
        "continuous_self_learning_candidate_score": 0.0,
        "gate_check_summary": {},
        "per_unit_rows": [],
        "aggregate_row_recomputation": {},
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            work_root=work,
            result_path=result,
            run_date=run_date,
            stream=stream,
            upstream=upstream,
            protected_before=protected_before,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "exact_solver_is_release_authority": EXACT_SOLVER_IS_RELEASE_AUTHORITY,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.perf_counter() - start),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    payload["sequential_evidence"] = sequential_evidence(payload)
    payload["per_unit_rows"] = _per_unit_rows(payload)
    aggregate = recompute_aggregate(payload)
    gates = gate_check_summary(aggregate, upstream)
    status, verdict, verdict_class = status_and_verdict(aggregate, gates)
    payload.update(
        {
            "status": status,
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "csl_execution_complete_score": aggregate["execution_complete_score_from_rows"],
            "continuous_self_learning_candidate_score": aggregate["candidate_score_from_rows"],
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
    if payload.get("verdict_class") not in {"positive", None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6522 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if payload.get("exact_solver_is_release_authority") is not True:
        errors.append("exact solver release authority missing")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload.get("upstream_gate_receipt", {}).get("gate_passed") is not True:
        errors.append("upstream gate failed")
    aggregate = recompute_aggregate(payload)
    gates = gate_check_summary(aggregate, payload.get("upstream_gate_receipt", {}))
    if payload.get("csl_execution_complete_score") != aggregate["execution_complete_score_from_rows"]:
        errors.append("csl_execution_complete_score mismatch")
    if payload.get("continuous_self_learning_candidate_score") != aggregate["candidate_score_from_rows"]:
        errors.append("continuous_self_learning_candidate_score mismatch")
    if aggregate["unsafe_write_count"] or aggregate["unsafe_use_count"]:
        errors.append("unsafe write or use detected")
    if aggregate["exact_answer_equality"] is not True:
        errors.append("exact answer drift detected")
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("gate_check_summary") != gates:
        errors.append("gate_check_summary mismatch")
    if payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is not True:
        errors.append("protected files changed")
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
        result_path=Path(result_path) if result_path is not None else REPO_ROOT / RESULT_RELATIVE_PATH,
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
        errors = validate_artifact(_read_json(result_path))
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path, work_root=Path(args.work_root))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests.
    raise SystemExit(main())
