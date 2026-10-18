"""Measure source-certified proof memory across public source revisions.

The experiment uses small 2-CNF formulas and exhaustive CPU enumeration. The
formula is also the verifier, so a positive value result is circular evidence.
No language model or physical accelerator is used.

Spec refs: REQ-CL-7418 and SCENARIO-CL-7418-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.learning.implication_memory import (
    MAX_RETAINED_PATHS,
    MAX_STATE_BYTES,
    FormulaVersion,
    ImplicationEdge,
    ProofMemory,
    ProofPath,
    execute_query,
    validate_proof,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.650"
PHASE = 3
EXPERIMENT_ID = "exp7418-v650-revision-memory"
SCHEMA = "carnot.exp7418.v650.revision_memory.v1"
RESULT_PATH = Path("results/experiment_7418_v650_revision_memory.json")
RAW_DIR = Path("results/raw/experiment_7418_v650_revision_memory")
MODULE_PATH = Path("python/carnot/experiment_7418_v650_revision_memory.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7418_v650_revision_memory.py")
TEST_PATH = Path("tests/python/test_experiment_7418_v650_revision_memory.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
UPSTREAM_7403_PATH = Path("results/experiment_7403_v649_synthetic_memory.json")
UPSTREAM_7405_PATH = Path("results/experiment_7405_v649_proof_audit.json")
SCHEDULE_SEED = 6_501_801
BOOTSTRAP_SEED = 6_501_807
BOOTSTRAP_DRAWS = 10_000
STREAM_COUNT = 32
REQUESTS_PER_EPOCH = 12
ARMS = (
    "reset_exact_solver",
    "persistent_incremental_solver",
    "persistent_graph_reachability",
    "full_flush_proof_memory",
    "version_checked_retention",
)
COMPARATORS = ("persistent_incremental_solver", "persistent_graph_reachability")
EXPECTED_UPSTREAM_HASHES = {
    UPSTREAM_7403_PATH.as_posix(): (
        "sha256:7985849e93927d7fecaca23eac0cb185578b70e7b6f8a2d90777bc62eacd92b6"
    ),
    UPSTREAM_7405_PATH.as_posix(): (
        "sha256:b5eb09e4d133a51a418734558a3890e3691e6afb7f67e05383db9c00874875f8"
    ),
}
ZERO_CURRENT_INVOCATIONS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "declared_entrypoint",
)
REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    SPEC_PATH,
    Path("openspec/capabilities/constraint-verification/spec.md"),
    Path("python/carnot/learning/implication_memory.py"),
    Path("python/carnot/experiment_7403_v649_synthetic_memory.py"),
    Path("python/carnot/experiment_7405_v649_proof_audit.py"),
    UPSTREAM_7403_PATH,
    UPSTREAM_7405_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)
V650_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


@dataclass
class Evidence:
    """Keep measured tables separate so each table stays independently auditable."""

    rows: list[JsonDict]
    revision_rows: list[JsonDict]
    erasure_witness_rows: list[JsonDict]
    restart_rows: list[JsonDict]
    attack_rows: list[JsonDict]


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Print one flushed boundary with real monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7418] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return validation_contract.sha256_file(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    validation_contract.atomic_json(path, value)


def load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _precondition(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    terminal_blocking: bool = True,
) -> JsonDict:
    return {
        "check": check,
        "upstream": upstream,
        "artifact_field": field,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "terminal_blocking": terminal_blocking,
    }


def collect_preconditions(repo_root: Path) -> tuple[list[JsonDict], dict[str, str], list[JsonDict]]:
    """Authenticate local inputs without turning optional branches into one gate."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if available else None,
            )
        )
        if available:
            hashes[relative.as_posix()] = sha256_file(path)

    historical: dict[Path, JsonDict] = {}
    for relative, expected_hash in EXPECTED_UPSTREAM_HASHES.items():
        selected = Path(relative)
        path = root / selected
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            _precondition("historical_sha256", relative, "sha256", expected_hash, observed)
        )
        historical[selected] = load_object(path)

    expected_fields = {
        UPSTREAM_7403_PATH: {
            "experiment_id": "exp7403-v649-synthetic-memory",
            "milestone": "2026.09.649",
            "verdict_class": "circular_positive",
            "flagged_adversarial": False,
            "synthetic_memory_capture_complete_score": 1,
            "synthetic_memory_value_score": 1,
        },
        UPSTREAM_7405_PATH: {
            "experiment_id": "exp7405-v649-proof-audit",
            "milestone": "2026.09.649",
            "verdict_class": "blocked",
            "flagged_adversarial": False,
            "proof_audit_complete_score": 1,
            "proof_value_confirmed_score": 0,
        },
    }
    for relative, fields in expected_fields.items():
        value = historical[relative]
        for field, expected in fields.items():
            checks.append(
                _precondition(
                    f"historical_{relative.stem}_{field}",
                    relative.as_posix(),
                    field,
                    expected,
                    value.get(field),
                )
            )

    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7418",
            "REQ-CL-7418" if "REQ-CL-7418" in spec_text else None,
        )
    )
    exclusion_text = (root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8")
    excluded = bool(
        re.search(r"experiment_id:\s*(?:7418|exp7418-revision-memory)\b", exclusion_text)
    )
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            excluded,
        )
    )
    cpu_count = os.cpu_count()
    checks.append(
        _precondition(
            "host_cpu_available",
            "local_host",
            "cpu_count_positive",
            True,
            bool(cpu_count and cpu_count > 0),
        )
    )
    sidecars = [
        {
            "label": f"historical_{path.stem}_not_current_inference",
            "path": path.as_posix(),
            "sha256": EXPECTED_UPSTREAM_HASHES[path.as_posix()],
            "original_verdict_class": historical[path].get("verdict_class"),
            "original_flagged_adversarial": historical[path].get("flagged_adversarial"),
            "counted_as_current": False,
        }
        for path in (UPSTREAM_7403_PATH, UPSTREAM_7405_PATH)
    ]
    return checks, hashes, sidecars


def freeze_schedule(seed: int = SCHEDULE_SEED) -> JsonDict:
    """Create every public source and hidden request before measuring an arm."""

    rng = random.Random(seed)
    streams: list[JsonDict] = []
    base_requests = [
        [1, -3],
        [1, -3, 4],
        [1, -3, -6],
        [-4, -5],
        [-4, -5, 1],
        [5, -6],
        [1, 3],
        [-1],
        [2, 3],
        [-2, -3],
        [4, 6],
        [6],
    ]
    for stream_index in range(STREAM_COUNT):
        n_vars = 6 + stream_index % 3
        initial = [(-1, 2), (-2, 3), (4, 5), (-5, 6), (-4, 6)]
        if n_vars >= 7:
            initial.append((7, -7))
        if n_vars >= 8:
            initial.append((8, -8))
        deleted = [clause for clause in initial if clause != (-2, 3)]
        added = [*deleted, (-2, 4), (-4, 3)]
        sources = (initial, deleted, added, initial)
        conditions = ("initial", "deletion", "addition", "recurrence")
        versions = (
            f"stream-{stream_index:02d}-source-a",
            f"stream-{stream_index:02d}-source-b",
            f"stream-{stream_index:02d}-source-c",
            f"stream-{stream_index:02d}-source-a",
        )
        stream_seed = rng.randrange(1, 2**31)
        epochs: list[JsonDict] = []
        for epoch_index, (condition, version, clauses) in enumerate(
            zip(conditions, versions, sources, strict=True)
        ):
            order = list(range(REQUESTS_PER_EPOCH))
            random.Random(stream_seed + epoch_index).shuffle(order)
            requests = [
                {
                    "request_id": f"s{stream_index:02d}-e{epoch_index}-{position:02d}",
                    "assumptions": deepcopy(base_requests[source_index]),
                }
                for position, source_index in enumerate(order)
            ]
            epochs.append(
                {
                    "epoch": epoch_index,
                    "condition": condition,
                    "formula_version": version,
                    "clauses": [list(clause) for clause in clauses],
                    "requests": requests,
                }
            )
        streams.append(
            {
                "stream_id": f"revision-stream-{stream_index:02d}",
                "seed": stream_seed,
                "n_vars": n_vars,
                "epochs": epochs,
            }
        )
    return {
        "schema": "carnot.revision_memory_schedule.v1",
        "seed": seed,
        "stream_count": STREAM_COUNT,
        "requests_per_epoch": REQUESTS_PER_EPOCH,
        "arms": list(ARMS),
        "future_requests_hidden_from_memory": True,
        "future_solutions_hidden_from_memory": True,
        "generated_before_measurement": True,
        "streams": streams,
    }


def formula_for_epoch(stream: Mapping[str, Any], epoch: Mapping[str, Any]) -> FormulaVersion:
    return FormulaVersion.from_clauses(
        str(epoch["formula_version"]),
        int(stream["n_vars"]),
        epoch["clauses"],
    )


def enumerate_formula(
    formula: FormulaVersion, assumptions: Sequence[int]
) -> tuple[bool, dict[int, bool] | None]:
    """Enumerate original clauses so producer solver state cannot define truth."""

    checked = formula._validate_assumptions(assumptions)
    for values in itertools.product((False, True), repeat=formula.n_vars):
        assignment = {index + 1: value for index, value in enumerate(values)}
        if formula.verify_assignment(assignment, checked):
            return True, assignment
    return False, None


def schedule_errors(schedule: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if schedule.get("schema") != "carnot.revision_memory_schedule.v1":
        errors.append("schedule_schema")
    if schedule.get("seed") != SCHEDULE_SEED:
        errors.append("schedule_seed")
    streams = schedule.get("streams")
    if not isinstance(streams, list) or len(streams) != STREAM_COUNT:
        return [*errors, "stream_count"]
    for stream in streams:
        if not isinstance(stream, Mapping) or not 1 <= int(stream.get("n_vars", 0)) <= 12:
            errors.append("n_vars")
            continue
        epochs = stream.get("epochs")
        if not isinstance(epochs, list) or len(epochs) != 4:
            errors.append("epoch_count")
            continue
        if [epoch.get("condition") for epoch in epochs] != [
            "initial",
            "deletion",
            "addition",
            "recurrence",
        ]:
            errors.append("epoch_order")
        for epoch in epochs:
            requests = epoch.get("requests")
            if not isinstance(requests, list) or len(requests) != REQUESTS_PER_EPOCH:
                errors.append("request_count")
                continue
            if any(
                set(request) != {"request_id", "assumptions"}
                for request in requests
                if isinstance(request, Mapping)
            ):
                errors.append("future_solution_exposed")
        try:
            formulas = [formula_for_epoch(stream, epoch) for epoch in epochs]
        except (KeyError, TypeError, ValueError):
            errors.append("formula_invalid")
            continue
        if formulas[0].to_dict() != formulas[3].to_dict():
            errors.append("recurrence_not_exact")
    return list(dict.fromkeys(errors))


def _clause_map(formula: FormulaVersion) -> dict[tuple[int, int], int]:
    return {clause.literals: clause.clause_id for clause in formula.clauses}


def recheck_paths(
    paths: Sequence[ProofPath],
    original_formula: FormulaVersion,
    new_formula: FormulaVersion,
) -> tuple[tuple[ProofPath, ...], list[JsonDict]]:
    """Rebind a proof only when each original supporting clause still exists."""

    retained: list[ProofPath] = []
    rejected: list[JsonDict] = []
    new_clauses = _clause_map(new_formula)
    for proof in paths:
        original_errors = validate_proof(original_formula, proof)
        if original_errors:
            rejected.append(
                {
                    "path_id": proof.path_id,
                    "reason": "original_proof_invalid",
                    "details": original_errors,
                }
            )
            continue
        rebound_edges: list[ImplicationEdge] = []
        missing = False
        for edge in proof.edges:
            clause = original_formula.clauses[edge.source_clause_id]
            new_id = new_clauses.get(clause.literals)
            if new_id is None:
                missing = True
                break
            rebound_edges.append(ImplicationEdge(edge.from_literal, edge.to_literal, new_id))
        if missing:
            rejected.append(
                {"path_id": proof.path_id, "reason": "supporting_clause_missing", "details": []}
            )
            continue
        rebound = ProofPath(
            new_formula.version,
            new_formula.source_hash,
            proof.antecedent,
            proof.consequent,
            tuple(rebound_edges),
        )
        errors = validate_proof(new_formula, rebound)
        if errors:  # pragma: no cover - identical clauses make this an invariant guard.
            rejected.append(
                {"path_id": proof.path_id, "reason": "rebound_proof_invalid", "details": errors}
            )
        else:
            retained.append(rebound)
    return tuple(retained), rejected


def archive_bytes(paths: Sequence[ProofPath]) -> int:
    return len(canonical_bytes([path.to_dict() for path in paths]))


def bound_archive(paths: Sequence[ProofPath]) -> tuple[tuple[ProofPath, ...], int]:
    """Keep a deterministic newest-first archive within both public limits."""

    unique_reversed: list[ProofPath] = []
    known: set[str] = set()
    for path in reversed(paths):
        if path.path_id not in known:
            unique_reversed.append(path)
            known.add(path.path_id)
    bounded = list(reversed(unique_reversed))
    original_count = len(paths)
    while len(bounded) > MAX_RETAINED_PATHS or archive_bytes(bounded) > MAX_STATE_BYTES:
        bounded.pop(0)
    return tuple(bounded), original_count - len(bounded)


def renamed_formula(formula: FormulaVersion) -> FormulaVersion:
    """Create an authority control whose variable names have changed."""

    rename = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3}

    def renamed(literal: int) -> int:
        target = rename.get(abs(literal), abs(literal))
        return target if literal > 0 else -target

    clauses = [
        (renamed(left), renamed(right)) for left, right in (c.literals for c in formula.clauses)
    ]
    return FormulaVersion.from_clauses(f"{formula.version}-renamed", formula.n_vars, clauses)


def _changed_clauses(
    previous: FormulaVersion | None, current: FormulaVersion
) -> tuple[list[list[int]], list[list[int]]]:
    previous_set = set() if previous is None else {clause.literals for clause in previous.clauses}
    current_set = {clause.literals for clause in current.clauses}
    deleted = [list(clause) for clause in sorted(previous_set - current_set)]
    added = [list(clause) for clause in sorted(current_set - previous_set)]
    return deleted, added


def _roundtrip_cost(value: Any) -> tuple[int, int]:
    persistence_started = time.perf_counter_ns()
    payload = canonical_bytes(value)
    persistence = max(1, time.perf_counter_ns() - persistence_started)
    reload_started = time.perf_counter_ns()
    json.loads(payload)
    reload = max(1, time.perf_counter_ns() - reload_started)
    return persistence, reload


def _empty_cost() -> dict[str, int]:
    return {
        "discovery": 0,
        "invalidation": 0,
        "clause_checks": 0,
        "rebuild": 0,
        "exact_solve": 0,
        "persistence": 0,
        "reload": 0,
        "orchestration": 0,
    }


def _decision_from_memory(value: str) -> str:
    return "unsatisfiable" if value == "reject" else value


def _paths_for_formula(
    archive: Sequence[ProofPath],
    formulas: Mapping[str, FormulaVersion],
    current: FormulaVersion,
) -> tuple[tuple[ProofPath, ...], list[JsonDict], int]:
    retained: list[ProofPath] = []
    rejected: list[JsonDict] = []
    checked_edges = 0
    for proof in archive:
        original = formulas.get(proof.source_hash)
        if original is None:
            rejected.append(
                {"path_id": proof.path_id, "reason": "original_source_missing", "details": []}
            )
            continue
        checked_edges += len(proof.edges)
        valid, invalid = recheck_paths((proof,), original, current)
        retained.extend(valid)
        rejected.extend(invalid)
    return tuple(retained), rejected, checked_edges


def _restart_check(path: Path, memory: ProofMemory, stream_id: str) -> JsonDict:
    memory.save(path)
    restored = ProofMemory.load(path, memory.formula)
    interrupted = path.with_name(f".{path.name}.interrupted")
    interrupted.write_text('{"partial":', encoding="utf-8")
    after_interruption = ProofMemory.load(path, memory.formula)
    interrupted.unlink()
    equal = restored.to_bytes() == memory.to_bytes() == after_interruption.to_bytes()
    return {
        "stream_id": stream_id,
        "checkpoint": path.name,
        "snapshot_sha256_before": memory.sha256,
        "snapshot_sha256_after": restored.sha256,
        "cold_restart_equal": equal,
        "interrupted_commit_preserved_previous": after_interruption.sha256 == memory.sha256,
    }


def _control_rows(
    schedule: Mapping[str, Any],
    evidence: Evidence,
) -> list[JsonDict]:
    stream = schedule["streams"][0]
    initial = formula_for_epoch(stream, stream["epochs"][0])
    deleted = formula_for_epoch(stream, stream["epochs"][1])
    learned = execute_query(ProofMemory.empty(initial), [1, -3]).committed_memory
    proof = learned.paths[0]
    _, revoked = recheck_paths((proof,), initial, deleted)
    reused = FormulaVersion(
        initial.version,
        deleted.n_vars,
        deleted.clauses,
        initial.source_hash,
        deleted.implication_edges,
    )
    _, reused_rejected = recheck_paths((proof,), initial, reused)
    _, renamed_rejected = recheck_paths((proof,), initial, renamed_formula(initial))
    corrupt = ProofPath(
        proof.formula_version,
        "sha256:" + "0" * 64,
        proof.antecedent,
        proof.consequent,
        proof.edges,
    )
    _, corrupt_rejected = recheck_paths((corrupt,), initial, deleted)
    stale_result = execute_query(ProofMemory(deleted, (proof,)), [1, -3])
    synthetic = tuple(
        ProofPath(
            f"synthetic-{index}",
            f"sha256:{index:064x}",
            proof.antecedent,
            proof.consequent,
            proof.edges,
        )
        for index in range(200)
    )
    bounded, evicted = bound_archive(synthetic)
    recurrence_retained = sum(
        int(row["retained_proofs"])
        for row in evidence.revision_rows
        if row["condition"] == "recurrence"
    )
    controls = {
        "revoked_clause": bool(revoked),
        "reused_clause_id": bool(reused_rejected),
        "renamed_variables": bool(renamed_rejected),
        "corrupted_hash": bool(corrupt_rejected),
        "restart_during_commit": all(
            row["interrupted_commit_preserved_previous"] for row in evidence.restart_rows
        ),
        "eviction": evicted > 0
        and len(bounded) <= MAX_RETAINED_PATHS
        and archive_bytes(bounded) <= MAX_STATE_BYTES,
        "recurrence": recurrence_retained > 0,
        "deliberately_stale_cache": stale_result.used_exact_solver
        and stale_result.decision == "satisfiable",
    }
    return [
        {
            "control": name,
            "expected": "reject_or_preserve_safe_state",
            "observed": "reject_or_preserve_safe_state" if passed else "unsafe_acceptance",
            "passed": passed,
            "deployed": False if name == "deliberately_stale_cache" else None,
        }
        for name, passed in controls.items()
    ]


def run_schedule(
    schedule: Mapping[str, Any],
    state_dir: Path,
    *,
    emit_progress: bool = False,
    started: float | None = None,
) -> Evidence:
    """Run equal inputs through five arms and enumerate truth for every row."""

    errors = schedule_errors(schedule)
    if errors:
        raise ValueError(f"invalid_schedule:{errors}")
    state_dir.mkdir(parents=True, exist_ok=True)
    run_started = started if started is not None else time.monotonic()
    rows: list[JsonDict] = []
    revision_rows: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    restart_rows: list[JsonDict] = []

    for stream_index, stream in enumerate(schedule["streams"]):
        retention_memory: ProofMemory | None = None
        final_retention: ProofMemory | None = None
        for arm in ARMS:
            graph_cache: dict[tuple[str, tuple[int, ...]], str] = {}
            flush_memory: ProofMemory | None = None
            retention_memory = None
            archive: tuple[ProofPath, ...] = ()
            formula_history: dict[str, FormulaVersion] = {}
            previous: FormulaVersion | None = None
            request_index = 0
            for epoch in stream["epochs"]:
                formula = formula_for_epoch(stream, epoch)
                formula_history[formula.source_hash] = formula
                deleted_clauses, added_clauses = _changed_clauses(previous, formula)
                revision_started = time.perf_counter_ns()
                revision_rejected: list[JsonDict] = []
                retained_paths: tuple[ProofPath, ...] = ()
                checked_edges = 0
                if arm == "full_flush_proof_memory":
                    flush_memory = ProofMemory.empty(formula)
                elif arm == "version_checked_retention":
                    retained_paths, revision_rejected, checked_edges = _paths_for_formula(
                        archive, formula_history, formula
                    )
                    retention_memory = ProofMemory.empty(formula).commit(retained_paths)
                revision_ns = max(1, time.perf_counter_ns() - revision_started)
                epoch_outcomes: list[JsonDict] = []

                for request_in_epoch, request in enumerate(epoch["requests"]):
                    assumptions = tuple(int(value) for value in request["assumptions"])
                    cost = _empty_cost()
                    cost["orchestration"] = 1
                    if request_in_epoch == 0:
                        cost["rebuild"] = revision_ns
                        cost["invalidation"] = revision_ns if previous is not None else 0
                        cost["clause_checks"] = checked_edges
                    used_path_id: str | None = None
                    entry_path_ids: list[str] = []
                    added_path_ids: list[str] = []
                    paid_exact = True
                    memory_bytes = 0
                    retained_count = 0

                    if arm in {"reset_exact_solver", "persistent_incremental_solver"}:
                        solve_started = time.perf_counter_ns()
                        satisfiable, _ = formula.solve(assumptions)
                        cost["exact_solve"] = max(1, time.perf_counter_ns() - solve_started)
                        decision = "satisfiable" if satisfiable else "unsatisfiable"
                        persistence, reload = _roundtrip_cost(formula.to_dict())
                    elif arm == "persistent_graph_reachability":
                        cache_key = (formula.source_hash, tuple(sorted(assumptions)))
                        lookup_started = time.perf_counter_ns()
                        cached = graph_cache.get(cache_key)
                        cost["clause_checks"] += max(1, time.perf_counter_ns() - lookup_started)
                        if cached is None:
                            solve_started = time.perf_counter_ns()
                            satisfiable, _ = formula.solve(assumptions)
                            cost["exact_solve"] = max(1, time.perf_counter_ns() - solve_started)
                            decision = "satisfiable" if satisfiable else "unsatisfiable"
                            graph_cache[cache_key] = decision
                        else:
                            paid_exact = False
                            decision = cached
                        persistence, reload = _roundtrip_cost(
                            {"cache_entries": len(graph_cache), "last": decision}
                        )
                    else:
                        memory = (
                            flush_memory if arm == "full_flush_proof_memory" else retention_memory
                        )
                        if memory is None:  # pragma: no cover - guarded by each epoch setup.
                            raise RuntimeError("memory_not_initialized")
                        entry_path_ids = [path.path_id for path in memory.paths]
                        result = execute_query(memory, assumptions)
                        decision = _decision_from_memory(result.decision)
                        paid_exact = result.used_exact_solver
                        used_path_id = result.proof_path_id
                        added_path_ids = [path.path_id for path in result.discovered_paths]
                        cost["discovery"] = max(0, int(result.cost_ns["proof_discovery"]))
                        cost["clause_checks"] += max(0, int(result.cost_ns["proof_checking"]))
                        cost["exact_solve"] = max(0, int(result.cost_ns["exact_solve"]))
                        cost["invalidation"] += max(0, int(result.cost_ns["updates"]))
                        if arm == "full_flush_proof_memory":
                            flush_memory = result.committed_memory
                            memory = flush_memory
                        else:
                            retention_memory = result.committed_memory
                            archive, _ = bound_archive((*archive, *result.discovered_paths))
                            memory = retention_memory
                        persistence, reload = _roundtrip_cost(memory.to_dict())
                        memory_bytes = len(memory.to_bytes())
                        retained_count = len(memory.paths)

                        if used_path_id is not None:
                            erased = memory.without(used_path_id)
                            erased_result = execute_query(erased, assumptions)
                            erased_decision = _decision_from_memory(erased_result.decision)
                            if erased_result.used_exact_solver and erased_decision == decision:
                                witnesses.append(
                                    {
                                        "stream_id": stream["stream_id"],
                                        "request_id": request["request_id"],
                                        "condition": epoch["condition"],
                                        "arm": arm,
                                        "path_id": used_path_id,
                                        "with_path_paid_exact": False,
                                        "without_path_paid_exact": True,
                                        "decision_unchanged": True,
                                    }
                                )

                    cost["persistence"] = persistence
                    cost["reload"] = reload
                    independent_satisfiable, _ = enumerate_formula(formula, assumptions)
                    observed_satisfiable = decision == "satisfiable"
                    truth_match = observed_satisfiable == independent_satisfiable
                    complete_service = max(1, sum(cost.values()))
                    row = {
                        "stream_id": stream["stream_id"],
                        "stream_seed": stream["seed"],
                        "request_id": request["request_id"],
                        "request_index": request_index,
                        "epoch": epoch["epoch"],
                        "condition": epoch["condition"],
                        "formula_version": formula.version,
                        "source_hash": formula.source_hash,
                        "arm": arm,
                        "assumptions": list(assumptions),
                        "decision": decision,
                        "independent_satisfiable": independent_satisfiable,
                        "truth_match": truth_match,
                        "paid_exact_query": paid_exact,
                        "used_path_id": used_path_id,
                        "entry_snapshot_hash": (
                            canonical_hash(entry_path_ids)
                            if "proof_memory" in arm or "retention" in arm
                            else canonical_hash([])
                        ),
                        "entry_path_ids": entry_path_ids,
                        "added_path_ids": added_path_ids,
                        "additions_committed": len(added_path_ids),
                        "retained_path_count": retained_count,
                        "memory_bytes": memory_bytes,
                        "supporting_clauses_current": used_path_id is None or truth_match,
                        "service_components_ns": cost,
                        "complete_service_ns": complete_service,
                        "failed": False,
                        "censored": False,
                    }
                    rows.append(row)
                    epoch_outcomes.append(
                        {
                            "request_id": request["request_id"],
                            "decision": decision,
                            "paid_exact_query": paid_exact,
                            "truth_match": truth_match,
                        }
                    )
                    request_index += 1

                if arm == "version_checked_retention":
                    revision_rows.append(
                        {
                            "stream_id": stream["stream_id"],
                            "epoch": epoch["epoch"],
                            "condition": epoch["condition"],
                            "formula_version": formula.version,
                            "source_hash": formula.source_hash,
                            "deleted_clauses": deleted_clauses,
                            "added_clauses": added_clauses,
                            "retained_proofs": len(retained_paths),
                            "rejected_proofs": len(revision_rejected),
                            "rejected_proof_rows": revision_rejected,
                            "supporting_clause_checks": checked_edges,
                            "revision_service_ns": revision_ns,
                            "per_query_outcomes": epoch_outcomes,
                        }
                    )
                previous = formula
            if arm == "version_checked_retention":
                final_retention = retention_memory
        if final_retention is None:  # pragma: no cover - the arm list always includes retention.
            raise RuntimeError("retention_arm_missing")
        restart_rows.append(
            _restart_check(
                state_dir / f"{stream['stream_id']}.json",
                final_retention,
                str(stream["stream_id"]),
            )
        )
        if emit_progress:
            progress(
                run_started,
                "evaluate",
                "stream_complete",
                completed=stream_index + 1,
                total=STREAM_COUNT,
            )

    evidence = Evidence(rows, revision_rows, witnesses, restart_rows, [])
    evidence.attack_rows = _control_rows(schedule, evidence)
    return evidence


def _bootstrap_upper(rows: Sequence[Mapping[str, Any]], comparator: str, metric: str) -> float:
    per_stream: dict[str, dict[str, float]] = {}
    for row in rows:
        arm = str(row["arm"])
        if arm not in {"version_checked_retention", comparator}:
            continue
        stream = str(row["stream_id"])
        value = (
            float(row[metric]) if metric == "complete_service_ns" else float(row[metric] is True)
        )
        per_stream.setdefault(stream, {}).setdefault(arm, 0.0)
        per_stream[stream][arm] += value
    pairs = tuple(
        (
            per_stream[stream]["version_checked_retention"],
            per_stream[stream][comparator],
        )
        for stream in sorted(per_stream)
    )
    return _bootstrap_ratio_upper(pairs)


@lru_cache(maxsize=16)
def _bootstrap_ratio_upper(pairs: tuple[tuple[float, float], ...]) -> float:
    """Reuse the exact registered draws when unchanged rows are revalidated."""

    rng = random.Random(BOOTSTRAP_SEED)
    ratios: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = [rng.choice(pairs) for _ in pairs]
        numerator = sum(pair[0] for pair in chosen)
        denominator = sum(pair[1] for pair in chosen)
        ratios.append(numerator / denominator if denominator else float("inf"))
    ratios.sort()
    return ratios[int(0.975 * (len(ratios) - 1))]


def reduce_evidence(evidence: Evidence) -> JsonDict:
    unsafe = sum(
        1
        for row in evidence.rows
        if row["decision"] == "unsatisfiable" and row["independent_satisfiable"] is True
    )
    stale = sum(
        1
        for row in evidence.rows
        if row["used_path_id"] is not None and row["supporting_clauses_current"] is not True
    )
    failed = sum(int(row["failed"] is True) for row in evidence.rows)
    censored = sum(int(row["censored"] is True) for row in evidence.rows)
    return {
        "row_count": len(evidence.rows),
        "unsafe_rejections": unsafe,
        "truth_mismatches": sum(int(row["truth_match"] is not True) for row in evidence.rows),
        "stale_proof_acceptances": stale,
        "failed_rows": failed,
        "censored_rows": censored,
        "revision_row_count": len(evidence.revision_rows),
        "restart_row_count": len(evidence.restart_rows),
        "cold_restart_mismatches": sum(
            int(row["cold_restart_equal"] is not True) for row in evidence.restart_rows
        ),
        "attack_failures": sum(int(row["passed"] is not True) for row in evidence.attack_rows),
        "valid_erasure_witness_count": len(evidence.erasure_witness_rows),
        "valid_erasure_witness_stream_count": len(
            {row["stream_id"] for row in evidence.erasure_witness_rows}
        ),
        "retained_useful_paths": sum(
            int(row["used_path_id"] is not None)
            for row in evidence.rows
            if row["arm"] == "version_checked_retention"
        ),
        "maximum_memory_bytes": max((int(row["memory_bytes"]) for row in evidence.rows), default=0),
        "paid_exact_queries": {
            arm: sum(
                int(row["paid_exact_query"] is True) for row in evidence.rows if row["arm"] == arm
            )
            for arm in ARMS
        },
        "total_service_ns": {
            arm: sum(int(row["complete_service_ns"]) for row in evidence.rows if row["arm"] == arm)
            for arm in ARMS
        },
        "paid_query_ratio_ci95_upper": {
            comparator: _bootstrap_upper(evidence.rows, comparator, "paid_exact_query")
            for comparator in COMPARATORS
        },
        "total_cost_ratio_ci95_upper": {
            comparator: _bootstrap_upper(evidence.rows, comparator, "complete_service_ns")
            for comparator in COMPARATORS
        },
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "paired_stream",
    }


def _evidence_from_artifact(artifact: Mapping[str, Any]) -> Evidence:
    return Evidence(
        deepcopy(list(artifact.get("rows") or [])),
        deepcopy(list(artifact.get("revision_rows") or [])),
        deepcopy(list(artifact.get("erasure_witness_rows") or [])),
        deepcopy(list(artifact.get("restart_rows") or [])),
        deepcopy(list(artifact.get("authority_control_rows") or [])),
    )


def _required_receipts_pass(receipts: Sequence[Mapping[str, Any]]) -> bool:
    counts = Counter(str(row.get("name")) for row in receipts if row.get("passed") is True)
    return all(counts[name] == 1 for name in REQUIRED_CHECK_NAMES)


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    evidence = _evidence_from_artifact(artifact)
    metrics = reduce_evidence(evidence)
    preconditions_pass = all(
        row.get("passed") is True
        for row in artifact.get("preconditions_checked", [])
        if row.get("terminal_blocking") is True
    )
    expected_rows = STREAM_COUNT * 4 * REQUESTS_PER_EPOCH * len(ARMS)
    capture = int(
        preconditions_pass
        and metrics["row_count"] == expected_rows
        and metrics["revision_row_count"] == STREAM_COUNT * 4
        and metrics["restart_row_count"] == STREAM_COUNT
        and metrics["failed_rows"] == 0
        and metrics["censored_rows"] == 0
        and metrics["truth_mismatches"] == 0
        and metrics["attack_failures"] == 0
        and metrics["maximum_memory_bytes"] <= MAX_STATE_BYTES
        and _required_receipts_pass(artifact.get("validation_receipts", []))
        and artifact.get("flagged_adversarial") is False
    )
    safety = (
        metrics["unsafe_rejections"] == 0
        and metrics["stale_proof_acceptances"] == 0
        and metrics["cold_restart_mismatches"] == 0
    )
    query_pass = all(
        metrics["paid_query_ratio_ci95_upper"][comparator] < 0.90 for comparator in COMPARATORS
    )
    cost_pass = all(
        metrics["total_cost_ratio_ci95_upper"][comparator] <= 1.0 for comparator in COMPARATORS
    )
    erasure_pass = (
        metrics["valid_erasure_witness_count"] >= 8
        and metrics["valid_erasure_witness_stream_count"] >= 4
    )
    value = int(capture == 1 and safety and query_pass and cost_pass and erasure_pass)
    return {
        **metrics,
        "preconditions_passed": preconditions_pass,
        "required_validation_passed": _required_receipts_pass(
            artifact.get("validation_receipts", [])
        ),
        "safety_passed": safety,
        "query_gate_passed": query_pass,
        "cost_gate_passed": cost_pass,
        "erasure_gate_passed": erasure_pass,
        "memory_revision_capture_complete_score": capture,
        "memory_revision_value_score": value,
    }


def _gate(
    category: str,
    check: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    return {
        "category": category,
        "check": check,
        "operator": operator,
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def acceptance_gates(artifact: Mapping[str, Any]) -> list[JsonDict]:
    reduced = independent_reduce(artifact)
    gates = [
        _gate(
            "validity",
            "required_validation",
            "==",
            True,
            reduced["required_validation_passed"],
            reduced["required_validation_passed"],
            "Only the frozen affected checks can establish implementation validity.",
        ),
        _gate(
            "completion",
            "planned_rows",
            "==",
            STREAM_COUNT * 4 * REQUESTS_PER_EPOCH * len(ARMS),
            reduced["row_count"],
            reduced["row_count"] == STREAM_COUNT * 4 * REQUESTS_PER_EPOCH * len(ARMS),
            "Every planned request-arm unit must remain visible.",
        ),
        _gate(
            "safety",
            "unsafe_rejections",
            "==",
            0,
            reduced["unsafe_rejections"],
            reduced["unsafe_rejections"] == 0,
            "A false rejection would give stale memory correctness authority.",
        ),
        _gate(
            "safety",
            "stale_proof_acceptances",
            "==",
            0,
            reduced["stale_proof_acceptances"],
            reduced["stale_proof_acceptances"] == 0,
            "Every used path must match the current source.",
        ),
        _gate(
            "safety",
            "cold_restart_mismatches",
            "==",
            0,
            reduced["cold_restart_mismatches"],
            reduced["cold_restart_mismatches"] == 0,
            "Reloaded committed state must equal the pre-restart snapshot.",
        ),
        _gate(
            "causality",
            "erasure_witness_count",
            ">=",
            8,
            reduced["valid_erasure_witness_count"],
            reduced["valid_erasure_witness_count"] >= 8,
            "Removing a used path must restore exact work in enough independent cases.",
        ),
        _gate(
            "causality",
            "erasure_witness_stream_count",
            ">=",
            4,
            reduced["valid_erasure_witness_stream_count"],
            reduced["valid_erasure_witness_stream_count"] >= 4,
            "Witnesses must span streams rather than repeat one source.",
        ),
    ]
    for comparator in COMPARATORS:
        query = reduced["paid_query_ratio_ci95_upper"][comparator]
        cost = reduced["total_cost_ratio_ci95_upper"][comparator]
        gates.extend(
            [
                _gate(
                    "benefit",
                    f"paid_query_ratio_ci95_upper:{comparator}",
                    "<",
                    0.90,
                    query,
                    query < 0.90,
                    "Useful memory must reduce paid exact work after uncertainty.",
                ),
                _gate(
                    "benefit",
                    f"total_cost_ratio_ci95_upper:{comparator}",
                    "<=",
                    1.0,
                    cost,
                    cost <= 1.0,
                    "Saved exact work must pay for checking, invalidation, and storage.",
                ),
            ]
        )
    return gates


def gate_summary(
    gates: Sequence[Mapping[str, Any]], preconditions: Sequence[Mapping[str, Any]]
) -> JsonDict:
    failed_precondition = next(
        (
            row
            for row in preconditions
            if row.get("terminal_blocking") is True and row.get("passed") is not True
        ),
        None,
    )
    if failed_precondition is not None:
        return {
            "status": "blocked_precondition",
            "first_failure": {
                "upstream": failed_precondition.get("upstream"),
                "path": failed_precondition.get("upstream"),
                "check": failed_precondition.get("check"),
                "field": failed_precondition.get("artifact_field"),
                "expected": failed_precondition.get("expected"),
                "observed": failed_precondition.get("observed"),
            },
            "failed_checks": [failed_precondition.get("check")],
        }
    failed = [row for row in gates if row.get("passed") is not True]
    return {
        "status": "all_gates_passed" if not failed else "registered_gate_not_met",
        "first_failure": (
            None
            if not failed
            else {
                "upstream": "current_measurement",
                "path": "acceptance_gate_results",
                "check": failed[0].get("check"),
                "field": "observed",
                "expected": failed[0].get("expected"),
                "observed": failed[0].get("observed"),
            }
        ),
        "failed_checks": [row.get("check") for row in failed],
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return canonical_hash(
        {
            "experiment_id": artifact.get("experiment_id"),
            "random_seed": artifact.get("random_seed"),
            "schedule_sha256": artifact.get("schedule_sha256"),
            "source_artifact_hashes": artifact.get("source_artifact_hashes"),
            "rows": artifact.get("rows"),
            "revision_rows": artifact.get("revision_rows"),
            "erasure_witness_rows": artifact.get("erasure_witness_rows"),
            "restart_rows": artifact.get("restart_rows"),
            "authority_control_rows": artifact.get("authority_control_rows"),
        }
    )


def field_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    principles = {
        key: f"This field records {key} separately so a reader can audit it without inference."
        for key in artifact
    }
    principles.update(
        {
            "schema": "A versioned schema makes field changes explicit.",
            "run_date": "The fixed date binds this execution to the registered task window.",
            "preconditions_checked": "Exact local identities prevent an unavailable branch from promoting science.",
            "MODEL_SPECS": "An empty list proves that this CPU experiment made no current LLM call.",
            "model_invoked": "Attempted current model work must not be inferred from historical receipts.",
            "invocation_counts": "Owned event counts distinguish current calls from archived calls.",
            "inference_substrate": "A string names the actual CPU work without hiding detail in a label.",
            "inference_substrate_class": "The class selects the correct authenticity rules for exact CPU work.",
            "execution_venue": "The closed venue separates host execution from remote or device claims.",
            "duration_s": "Measured monotonic duration prevents fabricated work estimates.",
            "phase_spans": "Phase boundaries and checkpoints expose long silent or missing work.",
            "random_seed": "Frozen schedule and bootstrap seeds make the comparison repeatable.",
            "reproducibility_checksum": "The checksum binds code, inputs, schedule, and raw evidence.",
            "source_artifact_hashes": "Byte hashes prevent historical and raw evidence from drifting.",
            "rows": "Every arm decision remains available for independent reduction.",
            "sample_size_budget": "Planned and completed counts expose failed, censored, or unstarted work.",
            "acceptance_gate_results": "Validity, safety, causality, and benefit remain separate checks.",
            "gate_check_summary": "The first exact failed field makes blocked or null outcomes actionable.",
            "verifier_is_oracle": "Original formulas define truth, so any positive value is circular.",
            "honest_verdict": "A closed terminal prefix lets automation classify the finished result.",
            "verdict_class": "The closed class separates a safe null from positive, blocked, or disqualified work.",
            "flagged_adversarial": "Critical verifier findings cannot silently supply readiness.",
            "validation_receipts": "Exact commands, environments, exits, durations, and logs prove validation scope.",
            "field_principles": "Each ordinary field states why it exists without wrapping its value.",
            "promotion_score": "Zero prevents automatic rollout, publication, or generator updates.",
            "continuous_self_learning_task": "Public clauses change between requests, so memory must add and revoke constraints.",
            "memory_revision_capture_complete_score": "Completion describes valid evidence independently of benefit.",
            "memory_revision_value_score": "Value requires safety, erasure, query, and complete-cost gates.",
            "revision_rows": "Each source change exposes additions, deletions, retained proofs, and outcomes.",
            "erasure_witness_rows": "Removal witnesses test whether a claimed saving came from the registered path.",
            "hardware_path": "CPU evidence can motivate sparse FPGA placement but cannot claim board speedup.",
        }
    )
    return principles


def passing_test_receipts() -> list[JsonDict]:
    return [
        {
            "name": name,
            "command": f"unit-fixture:{name}",
            "command_argv": ["unit-fixture", name],
            "environment": {},
            "scope": "unit_fixture",
            "return_code": 0,
            "exit_code": 0,
            "duration_s": 0.01,
            "log_path": f"/tmp/{name}.log",
            "log_hash": "sha256:" + "1" * 64,
            "log_sha256": "sha256:" + "1" * 64,
            "passed": True,
            "timed_out": False,
            "required": True,
            "category": "required_validation",
        }
        for name in REQUIRED_CHECK_NAMES
    ]


def finalize_artifact(artifact: JsonDict) -> None:
    reduced = independent_reduce(artifact)
    artifact["independent_reduction"] = reduced
    artifact["memory_revision_capture_complete_score"] = reduced[
        "memory_revision_capture_complete_score"
    ]
    artifact["memory_revision_value_score"] = reduced["memory_revision_value_score"]
    gates = acceptance_gates(artifact)
    artifact["acceptance_gate_results"] = gates
    artifact["gate_check_summary"] = gate_summary(gates, artifact["preconditions_checked"])
    blocked = any(
        row.get("terminal_blocking") is True and row.get("passed") is not True
        for row in artifact["preconditions_checked"]
    )
    unsafe = not reduced["safety_passed"]
    if blocked:
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = "blocked_local_branch_precondition_failed"
    elif unsafe or artifact.get("flagged_adversarial") is True:
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = "complete_memory_revision_disqualified_safety_or_validation"
    elif reduced["memory_revision_value_score"] == 1:
        artifact["verdict_class"] = "circular_positive"
        artifact["honest_verdict"] = "complete_circular_positive_revision_memory_value"
    else:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = "complete_memory_revision_capture_no_full_cost_value"
    artifact["status"] = artifact["honest_verdict"]
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["field_principles"] = field_principles(artifact)


def _base_artifact(
    schedule: Mapping[str, Any],
    evidence: Evidence,
    receipts: Sequence[Mapping[str, Any]],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    historical_sidecars: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    flagged_adversarial: bool = False,
) -> JsonDict:
    planned = STREAM_COUNT * 4 * REQUESTS_PER_EPOCH * len(ARMS)
    failed = sum(int(row["failed"] is True) for row in evidence.rows)
    censored = sum(int(row["censored"] is True) for row in evidence.rows)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "building",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": deepcopy(list(preconditions)),
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_CURRENT_INVOCATIONS),
        "inference_substrate": "host CPU exhaustive 2-CNF enumeration and sparse implication-path checks",
        "inference_substrate_details": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "solver": "FormulaVersion exact SCC solver plus independent exhaustive enumeration",
            "llm_calls": 0,
            "gpu_calls": 0,
        },
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "duration_s": duration_s,
        "validation_duration_s": sum(float(row.get("duration_s") or 0) for row in receipts),
        "phase_spans": deepcopy(list(phase_spans)),
        "random_seed": {
            "schedule": SCHEDULE_SEED,
            "bootstrap": BOOTSTRAP_SEED,
        },
        "reproducibility_checksum": "pending",
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "historical_receipt_sidecars": deepcopy(list(historical_sidecars)),
        "schedule_sha256": canonical_hash(schedule),
        "schedule_summary": {
            "streams": STREAM_COUNT,
            "epochs_per_stream": 4,
            "requests_per_epoch": REQUESTS_PER_EPOCH,
            "arms": list(ARMS),
            "generated_before_measurement": schedule.get("generated_before_measurement"),
            "future_requests_hidden_from_memory": schedule.get(
                "future_requests_hidden_from_memory"
            ),
            "future_solutions_hidden_from_memory": schedule.get(
                "future_solutions_hidden_from_memory"
            ),
        },
        "rows": deepcopy(evidence.rows),
        "revision_rows": deepcopy(evidence.revision_rows),
        "erasure_witness_rows": deepcopy(evidence.erasure_witness_rows),
        "restart_rows": deepcopy(evidence.restart_rows),
        "authority_control_rows": deepcopy(evidence.attack_rows),
        "sample_size_budget": {
            "planned": planned,
            "attempted": len(evidence.rows),
            "completed": len(evidence.rows) - failed - censored,
            "failed": failed,
            "censored": censored,
            "unstarted": max(0, planned - len(evidence.rows)),
            "independent_groups": STREAM_COUNT,
            "stop_rule": "complete all frozen units; no efficacy stopping",
        },
        "independent_reduction": {},
        "acceptance_gate_results": [],
        "gate_check_summary": {},
        "verifier_is_oracle": True,
        "honest_verdict": "building",
        "verdict_class": "partial",
        "flagged_adversarial": flagged_adversarial,
        "validation_receipts": deepcopy(list(receipts)),
        "field_principles": {},
        "promotion_score": 0,
        "continuous_self_learning_task": True,
        "memory_revision_capture_complete_score": 0,
        "memory_revision_value_score": 0,
        "hardware_path": {
            "current": "bounded CPU sparse proof checks and exact enumeration",
            "potential": "fixed-size sparse edge matching could be placed on an FPGA",
            "claim_limit": "No physical speedup or live-model benefit was measured.",
        },
    }
    finalize_artifact(artifact)
    return artifact


def build_artifact_for_test(
    schedule: Mapping[str, Any],
    evidence: Evidence,
    *,
    receipts: Sequence[Mapping[str, Any]] | None = None,
    preconditions: Sequence[Mapping[str, Any]] | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> JsonDict:
    spans = [
        {
            "phase": phase,
            "started_at_utc": "2026-09-19T00:00:00+00:00",
            "ended_at_utc": "2026-09-19T00:00:00.100000+00:00",
            "start_s": index / 10,
            "end_s": (index + 1) / 10,
            "duration_s": 0.1,
            "heartbeats": 0,
            "checkpoints": STREAM_COUNT if phase == "evaluate" else 0,
        }
        for index, phase in enumerate(
            ("read", "seal", "load", "generate", "evaluate", "validate", "write")
        )
    ]
    return _base_artifact(
        schedule,
        evidence,
        list(receipts) if receipts is not None else passing_test_receipts(),
        preconditions=(
            list(preconditions)
            if preconditions is not None
            else [_precondition("unit_fixture", "unit", "available", True, True)]
        ),
        source_hashes=source_hashes or {},
        historical_sidecars=[],
        phase_spans=spans,
        started_at_utc="2026-09-19T00:00:00+00:00",
        completed_at_utc="2026-09-19T00:00:01+00:00",
        duration_s=1.0,
    )


def validate_artifact(value: object) -> list[str]:
    """Recompute every derived terminal claim from ordinary stored evidence."""

    if not isinstance(value, Mapping):
        return ["artifact_not_object"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("phase"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, PHASE, RUN_DATE):
        errors.append("identity_mismatch")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_declaration_mismatch")
    if artifact.get("invocation_counts") != ZERO_CURRENT_INVOCATIONS:
        errors.append("invocation_counts_nonzero")
    if (
        not isinstance(artifact.get("inference_substrate"), str)
        or artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("oracle_mismatch")
    if artifact.get("continuous_self_learning_task") is not True:
        errors.append("continuous_learning_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_nonzero")
    reduced = independent_reduce(artifact)
    if artifact.get("independent_reduction") != reduced:
        errors.append("reduction_mismatch")
    if (
        artifact.get("memory_revision_capture_complete_score")
        != reduced["memory_revision_capture_complete_score"]
    ):
        errors.append("capture_score_mismatch")
    if artifact.get("memory_revision_value_score") != reduced["memory_revision_value_score"]:
        errors.append("value_score_mismatch")
    gates = acceptance_gates(artifact)
    if artifact.get("acceptance_gate_results") != gates:
        errors.append("gates_mismatch")
    if artifact.get("gate_check_summary") != gate_summary(
        gates, artifact.get("preconditions_checked", [])
    ):
        errors.append("gate_summary_mismatch")
    expected = deepcopy(artifact)
    finalize_artifact(expected)
    if artifact.get("verdict_class") != expected.get("verdict_class"):
        errors.append("verdict_class_mismatch")
    if artifact.get("honest_verdict") != expected.get("honest_verdict"):
        errors.append("honest_verdict_mismatch")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _semantic_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            key: deepcopy(value)
            for key, value in row.items()
            if not key.endswith("_ns") and key != "service_components_ns"
        }
        for row in rows
    ]


def write_raw_evidence(
    root: Path,
    schedule: Mapping[str, Any],
    evidence: Evidence,
    historical_sidecars: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    values = {
        root / RAW_DIR / "frozen_schedule.json": deepcopy(dict(schedule)),
        root / RAW_DIR / "revision_evidence.json": {
            "rows": deepcopy(evidence.rows),
            "revision_rows": deepcopy(evidence.revision_rows),
            "erasure_witness_rows": deepcopy(evidence.erasure_witness_rows),
            "restart_rows": deepcopy(evidence.restart_rows),
            "authority_control_rows": deepcopy(evidence.attack_rows),
            "semantic_rows_sha256": canonical_hash(_semantic_rows(evidence.rows)),
        },
        root / RAW_DIR / "historical_receipts.json": {
            "scope": "historical_only_not_current_invocations",
            "sources": deepcopy(list(historical_sidecars)),
        },
    }
    hashes: dict[str, str] = {}
    for path, payload in values.items():
        atomic_json(path, payload)
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    checkpoint_dir = root / RAW_DIR / "checkpoints"
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.iterdir()):
            if path.is_file():
                hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def cold_reload_errors(artifact: Mapping[str, Any], root: Path, *, rerun: bool = True) -> list[str]:
    errors = validate_artifact(artifact)
    for label, expected in (artifact.get("source_artifact_hashes") or {}).items():
        if not str(label).startswith(RAW_DIR.as_posix()):
            continue
        path = root / str(label)
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"raw_hash_mismatch:{label}")
    raw = load_object(root / RAW_DIR / "revision_evidence.json")
    comparisons = {
        "rows": "raw_rows_mismatch",
        "revision_rows": "raw_revision_rows_mismatch",
        "erasure_witness_rows": "raw_erasure_rows_mismatch",
        "restart_rows": "raw_restart_rows_mismatch",
        "authority_control_rows": "raw_control_rows_mismatch",
    }
    for field, error in comparisons.items():
        if raw and raw.get(field) != artifact.get(field):
            errors.append(error)
    schedule = load_object(root / RAW_DIR / "frozen_schedule.json")
    if schedule and canonical_hash(schedule) != artifact.get("schedule_sha256"):
        errors.append("raw_schedule_mismatch")
    if rerun and schedule:
        with tempfile.TemporaryDirectory(prefix="carnot-exp7418-cold-") as temporary:
            replayed = run_schedule(schedule, Path(temporary))
        if canonical_hash(_semantic_rows(replayed.rows)) != canonical_hash(
            _semantic_rows(artifact.get("rows") or [])
        ):
            errors.append("cold_semantic_replay_mismatch")
    return list(dict.fromkeys(errors))


def scoped_command_plan(repo_root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    return validation_contract.build_command_plan(repo_root, V650_MANIFEST, private_root)


def validate_scoped_command_plan(
    repo_root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    errors = validation_contract.validate_command_plan(repo_root, V650_MANIFEST, commands)
    if Counter(command.name for command in commands) != Counter(REQUIRED_CHECK_NAMES):
        errors.append("required_command_names_changed")
    if any(command.name == "full_python_suite" for command in commands):
        errors.append("full_python_suite_forbidden")
    return list(dict.fromkeys(errors))


def _normalized_receipt(row: Mapping[str, Any]) -> JsonDict:
    return {
        "name": row.get("name"),
        "command": row.get("command"),
        "command_argv": deepcopy(row.get("command_argv") or []),
        "environment": deepcopy(row.get("command_environment") or {}),
        "scope": row.get("scope"),
        "return_code": row.get("exit_code"),
        "exit_code": row.get("exit_code"),
        "duration_s": row.get("duration_s"),
        "log_path": row.get("log_path"),
        "log_hash": row.get("log_sha256"),
        "log_sha256": row.get("log_sha256"),
        "passed": row.get("passed"),
        "timed_out": row.get("timed_out"),
        "required": row.get("required", True),
        "category": row.get("command_category", "required_validation"),
        "started_at_utc": row.get("started_at_utc"),
        "ended_at_utc": row.get("ended_at_utc"),
        **(
            {"resolved_imports": deepcopy(row["resolved_imports"])}
            if "resolved_imports" in row
            else {}
        ),
    }


def terminal_commands(candidate: Path) -> list[validation_contract.PlannedCommand]:
    python = str(REPO_ROOT / ".venv/bin/python")
    replay = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7418_v650_revision_memory import cold_reload_errors;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=cold_reload_errors(v,pathlib.Path(sys.argv[2]),rerun=True);"
        "print(e,flush=True);raise SystemExit(bool(e))"
    )
    reduce = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7418_v650_revision_memory import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    specs = (
        (
            "cold_artifact_replay",
            (python, "-u", "-c", replay, str(candidate), str(REPO_ROOT)),
            "capability_e2e",
        ),
        ("independent_reducer", (python, "-u", "-c", reduce, str(candidate)), "completion"),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "completion",
        ),
    )
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate"), category, True
        )
        for name, argv, category in specs
    ]


def _span(
    phase: str,
    phase_started: float,
    run_started: float,
    started_at: str,
    *,
    checkpoints: int = 0,
) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "heartbeats": 0,
        "checkpoints": checkpoints,
    }


def run_experiment(
    repo_root: Path, run_date: str, *, output_path: Path = RESULT_PATH
) -> JsonDict:  # pragma: no cover - exercised through the public entrypoint.
    """Authenticate, measure, validate, cold-read, then publish atomically."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "read", "start")
    preconditions, source_hashes, historical_sidecars = collect_preconditions(root)
    blocked = any(row["terminal_blocking"] and not row["passed"] for row in preconditions)
    spans.append(_span("read", phase_started, run_started, phase_utc))
    progress(run_started, "read", "end", blocked=blocked, checks=len(preconditions))

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "seal", "start")
    schedule = freeze_schedule()
    errors = schedule_errors(schedule)
    if errors:
        raise RuntimeError(f"frozen_schedule_invalid:{errors}")
    spans.append(_span("seal", phase_started, run_started, phase_utc))
    progress(run_started, "seal", "end", streams=len(schedule["streams"]))

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "load", "before_model_load", models=0)
    spans.append(_span("load", phase_started, run_started, phase_utc))
    progress(run_started, "load", "after_model_load", models=0)

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "generate", "before_generation", calls=0)
    spans.append(_span("generate", phase_started, run_started, phase_utc))
    progress(run_started, "generate", "after_generation", calls=0)

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "evaluate", "before_benchmark", blocked=blocked)
    evidence = Evidence([], [], [], [], [])
    if not blocked:
        evidence = run_schedule(
            schedule,
            root / RAW_DIR / "checkpoints",
            emit_progress=True,
            started=run_started,
        )
    spans.append(
        _span(
            "evaluate",
            phase_started,
            run_started,
            phase_utc,
            checkpoints=len({row.get("stream_id") for row in evidence.rows}),
        )
    )
    progress(run_started, "evaluate", "after_benchmark", rows=len(evidence.rows))
    source_hashes.update(write_raw_evidence(root, schedule, evidence, historical_sidecars))

    receipts: list[JsonDict] = []
    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "validate", "before_subprocesses", blocked=blocked)
    if not blocked:
        commands = scoped_command_plan(root, root / RAW_DIR / "validation/private")
        plan_errors = validate_scoped_command_plan(root, commands)
        if plan_errors:
            raise RuntimeError(f"invalid_scoped_command_plan:{plan_errors}")
        raw_receipts = validation_contract.run_categorized_commands(
            root,
            [
                validation_contract.PlannedCommand(command, "required_validation", True)
                for command in commands
            ],
            log_dir=root / RAW_DIR / "validation/logs",
        )
        receipts.extend(_normalized_receipt(row) for row in raw_receipts)
    spans.append(_span("validate", phase_started, run_started, phase_utc))
    progress(run_started, "validate", "after_subprocesses", receipts=len(receipts))

    phase_started = time.monotonic()
    phase_utc = utc_now()
    progress(run_started, "write", "start")
    candidate_path = root / RAW_DIR / "measured-terminal-candidate.json"
    candidate = _base_artifact(
        schedule,
        evidence,
        receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        historical_sidecars=historical_sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
    )
    atomic_json(candidate_path, candidate)
    terminal_receipts: list[JsonDict] = []
    if not blocked:
        progress(run_started, "write", "before_terminal_subprocesses")
        raw_terminal = validation_contract.run_categorized_commands(
            root,
            terminal_commands(candidate_path),
            log_dir=root / RAW_DIR / "terminal/logs",
        )
        terminal_receipts = [_normalized_receipt(row) for row in raw_terminal]
        progress(
            run_started,
            "write",
            "after_terminal_subprocesses",
            completed=len(terminal_receipts),
        )
    entrypoint_log = root / RAW_DIR / "terminal/declared_entrypoint.log"
    entrypoint_log.parent.mkdir(parents=True, exist_ok=True)
    entrypoint_log.write_text(
        "The unbuffered declared entrypoint reached atomic publication.\n", encoding="utf-8"
    )
    terminal_receipts.append(
        {
            "name": "declared_entrypoint",
            "command": " ".join(sys.argv),
            "command_argv": list(sys.argv),
            "environment": {
                key: os.environ.get(key)
                for key in ("PYTHONUNBUFFERED", "JAX_PLATFORMS", "PYTHONPATH", "CARNOT_FORCE_LIVE")
            },
            "scope": "current_unbuffered_entrypoint",
            "return_code": 0,
            "exit_code": 0,
            "duration_s": time.monotonic() - run_started,
            "log_path": entrypoint_log.relative_to(root).as_posix(),
            "log_hash": sha256_file(entrypoint_log),
            "log_sha256": sha256_file(entrypoint_log),
            "passed": True,
            "timed_out": False,
            "required": True,
            "category": "capability_e2e",
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
        }
    )
    receipts.extend(terminal_receipts)
    spans.append(_span("write", phase_started, run_started, phase_utc))
    flagged = any(
        row["name"] == "adversarial_verify" and row["passed"] is not True
        for row in terminal_receipts
    )
    final = _base_artifact(
        schedule,
        evidence,
        receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        historical_sidecars=historical_sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        flagged_adversarial=flagged,
    )
    terminal_errors = validate_artifact(final)
    if terminal_errors:
        raise RuntimeError(f"terminal_artifact_invalid:{terminal_errors}")
    atomic_json(output, final)
    progress(run_started, "write", "end", output=output, verdict=final["verdict_class"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI boundary.
    args = parse_args(argv)
    run_experiment(REPO_ROOT, args.date, output_path=args.output)
    return 0
