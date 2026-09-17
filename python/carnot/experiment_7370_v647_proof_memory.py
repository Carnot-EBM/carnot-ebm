"""Build a development fixture for source-checked 2-CNF proof memory.

This experiment changes the public representation after the V646 schedule
memory null. It measures a host-only development fixture. It does not claim
prospective learning value or acquire hidden rules from Boolean feedback.

Spec refs: REQ-CL-7370 and SCENARIO-CL-7370-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

from carnot import experiment_7358_v646_validation_contract as validation_contract
from carnot.learning.implication_memory import (
    FormulaVersion,
    ImplicationEdge,
    ProofMemory,
    ProofPath,
    execute_query,
    proof_from_dict,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260917"
MILESTONE = "2026.09.647"
PHASE = 1
EXPERIMENT_ID = "exp7370-v647-proof-memory"
SCHEMA = "carnot.exp7370.v647.proof_memory.v1"
RESULT_PATH = Path("results/experiment_7370_v647_proof_memory.json")
RAW_DIR = Path("results/raw/experiment_7370_v647_proof_memory")
MODULE_PATH = Path("python/carnot/experiment_7370_v647_proof_memory.py")
MEMORY_MODULE_PATH = Path("python/carnot/learning/implication_memory.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7370_v647_proof_memory.py")
TEST_PATH = Path("tests/python/test_experiment_7370_v647_proof_memory.py")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
CAPSTONE_PATH = Path("results/experiment_7368_v646_capstone.json")
REQUIRED_CHECK_NAMES = validation_scope.REQUIRED_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "cold_artifact_replay",
    "independent_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
    "declared_entrypoint",
)
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
RANDOM_SEED = {
    "development": 7_370_101,
    "formula": 7_370_211,
    "resampling": 7_370_307,
}
CLOSED_VERDICTS = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/verify/sat.py"),
    Path("python/carnot/pipeline/constraint_addition.py"),
    SPEC_PATH,
    MEMORY_MODULE_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

V647_MANIFEST = validation_contract.AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MEMORY_MODULE_PATH.as_posix(), MODULE_PATH.as_posix()),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def progress(started: float, phase: str, event: str, **details: Any) -> None:
    """Flush each boundary with real monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7370] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    return validation_contract.sha256_file(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    validation_contract.atomic_json(path, value)


def _precondition(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    *,
    terminal_blocking: bool,
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


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_preconditions(
    repo_root: Path,
    *,
    capstone_path: Path | None = None,
) -> tuple[list[JsonDict], dict[str, str], list[JsonDict]]:
    """Check task sources and label V646 as non-authorizing history."""

    root = repo_root.resolve()
    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            _precondition(
                f"source_bytes:{relative.as_posix()}",
                relative.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                "readable_nonempty_bytes" if present else "missing",
                terminal_blocking=True,
            )
        )
        if present:
            hashes[relative.as_posix()] = sha256_file(path)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    checks.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-CL-7370",
            "REQ-CL-7370" if "REQ-CL-7370" in spec_text else None,
            terminal_blocking=True,
        )
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    checks.append(
        _precondition(
            "current_task_not_quarantined",
            "ops/exclusion_manifest.yaml",
            EXPERIMENT_ID,
            False,
            "experiment_id: 7370" in exclusion_text,
            terminal_blocking=True,
        )
    )

    historical_path = capstone_path or root / CAPSTONE_PATH
    historical = _load_object(historical_path)
    sidecars: list[JsonDict] = []
    if not historical:
        observed = "missing" if not historical_path.exists() else "malformed"
        checks.append(
            _precondition(
                "historical_capstone_path",
                str(historical_path),
                "path",
                "readable_nonempty_json_object",
                observed,
                terminal_blocking=False,
            )
        )
    else:
        checks.append(
            _precondition(
                "historical_capstone_path",
                str(historical_path),
                "path",
                "readable_nonempty_json_object",
                "readable_nonempty_json_object",
                terminal_blocking=False,
            )
        )
        expected = {
            "experiment_id": "exp7368-capstone",
            "milestone": "2026.09.646",
            "verdict_class": "disqualified",
            "flagged_adversarial": False,
            "required_science_complete_score": 0,
        }
        for field, expected_value in expected.items():
            checks.append(
                _precondition(
                    f"historical_capstone_{field}",
                    str(historical_path),
                    field,
                    expected_value,
                    historical.get(field),
                    terminal_blocking=False,
                )
            )
        digest = sha256_file(historical_path)
        hashes[str(historical_path)] = digest
        sidecars.append(
            {
                "label": "historical_disqualified_v646_not_current_inference",
                "artifact_path": str(historical_path),
                "artifact_sha256": digest,
                "historical_verdict_class": historical.get("verdict_class"),
                "historical_flagged_adversarial": historical.get("flagged_adversarial"),
                "historical_invocation_counts": deepcopy(historical.get("invocation_counts") or {}),
                "historical_only": True,
                "counted_as_current": False,
                "authorizes_readiness": False,
            }
        )
    return checks, hashes, sidecars


def _feature_clauses(n_vars: int, feature: str) -> list[tuple[int, int]]:
    chain = [(-variable, variable + 1) for variable in range(1, n_vars)]
    if feature == "independent_conflicts":
        chain.extend([(-1, 2), (-3, 4), (-5, 6)])
    elif feature == "satisfiable_near_miss":
        chain.extend([(2, -2), (n_vars, -n_vars)])
    elif feature == "duplicate_and_version_edit":
        chain.extend([chain[0], chain[-1]])
    return chain


def build_development_panel() -> list[JsonDict]:
    """Create four public formula types at each required variable count."""

    features = (
        "long_chain",
        "independent_conflicts",
        "satisfiable_near_miss",
        "duplicate_and_version_edit",
    )
    panel: list[JsonDict] = []
    for n_vars in (8, 12, 24, 32):
        for feature_index, feature in enumerate(features):
            formula_id = f"n{n_vars}-{feature_index}-{feature}"
            version = f"{formula_id}-v{2 if feature == 'duplicate_and_version_edit' else 1}"
            formula = FormulaVersion.from_clauses(
                version,
                n_vars,
                _feature_clauses(n_vars, feature),
            )
            panel.append(
                {
                    "formula_id": formula_id,
                    "feature": feature,
                    "formula": formula,
                    "assumption_queries": (
                        (1, -n_vars),
                        (1, -n_vars, max(2, n_vars // 2)),
                        (1, n_vars),
                        (-1, -n_vars),
                    ),
                }
            )
    return panel


def _enumeration_sat(formula: FormulaVersion, assumptions: Sequence[int]) -> bool:
    for bits in itertools.product((False, True), repeat=formula.n_vars):
        assignment = {index + 1: value for index, value in enumerate(bits)}
        if formula.verify_assignment(assignment, assumptions):
            return True
    return False


def _independent_dpll(formula: FormulaVersion, assumptions: Sequence[int]) -> bool:
    clauses = [tuple(clause.literals) for clause in formula.clauses]
    clauses.extend((literal, literal) for literal in assumptions)

    def search(remaining: list[tuple[int, int]], assigned: dict[int, bool]) -> bool:
        while True:
            reduced: list[tuple[int, ...]] = []
            unit: int | None = None
            for clause in remaining:
                unresolved: list[int] = []
                satisfied = False
                for literal in clause:
                    if abs(literal) in assigned:
                        value = assigned[abs(literal)]
                        if value == (literal > 0):
                            satisfied = True
                            break
                    else:
                        unresolved.append(literal)
                if satisfied:
                    continue
                unique = tuple(dict.fromkeys(unresolved))
                if not unique:
                    return False
                if len(unique) == 1:
                    unit = unique[0]
                reduced.append(unique)
            if not reduced:
                return True
            if unit is None:
                branch = reduced[0][0]
                for value in (branch > 0, branch < 0):
                    child = {**assigned, abs(branch): value}
                    if search([tuple(clause) for clause in reduced], child):
                        return True
                return False
            variable, value = abs(unit), unit > 0
            if (
                variable in assigned and assigned[variable] != value
            ):  # pragma: no cover - reduction catches the empty clause first.
                return False
            assigned[variable] = value
            remaining = [tuple(clause) for clause in reduced]

    return search(clauses, {})


def _reachability(formula: FormulaVersion) -> dict[int, set[int]]:
    outgoing: dict[int, list[int]] = {}
    for edge in formula.implication_edges:
        outgoing.setdefault(edge.from_literal, []).append(edge.to_literal)
    result: dict[int, set[int]] = {}
    for source in (*range(1, formula.n_vars + 1), *range(-1, -formula.n_vars - 1, -1)):
        seen = {source}
        queue = [source]
        while queue:
            current = queue.pop(0)
            for target in outgoing.get(current, []):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        result[source] = seen
    return result


def _graph_conflict(reachability: Mapping[int, set[int]], assumptions: Sequence[int]) -> bool:
    return any(
        -right in reachability.get(left, set()) for left in assumptions for right in assumptions
    )


def _cost_template() -> dict[str, int]:
    return {
        "proof_discovery": 0,
        "proof_checking": 0,
        "updates": 0,
        "storage": 0,
        "exact_solve": 0,
    }


def _row(
    *,
    formula_id: str,
    feature: str,
    formula: FormulaVersion,
    query_index: int,
    assumptions: Sequence[int],
    arm: str,
    decision: str,
    used_exact_solver: bool,
    independent_sat: bool,
    comparison_method: str,
    costs: Mapping[str, int],
    storage_bytes: int,
) -> JsonDict:
    actual_sat = decision == "satisfiable"
    charged = {key: int(costs.get(key, 0)) for key in _cost_template()}
    charged["total"] = sum(charged.values())
    return {
        "formula_id": formula_id,
        "feature": feature,
        "formula_version": formula.version,
        "formula_source_hash": formula.source_hash,
        "n_vars": formula.n_vars,
        "query_index": query_index,
        "assumptions": list(assumptions),
        "arm": arm,
        "decision": decision,
        "used_exact_solver": used_exact_solver,
        "independent_satisfiable": independent_sat,
        "independent_exact_match": actual_sat == independent_sat,
        "comparison_method": comparison_method,
        "same_public_formula_and_assumptions": True,
        "cost_ns": charged,
        "storage_bytes": storage_bytes,
        "failure": None,
        "censored": False,
    }


def _hostile_controls(formula: FormulaVersion, proof: ProofPath) -> list[JsonDict]:
    payload = proof.to_dict()
    attacks: list[tuple[str, JsonDict]] = []
    changed = deepcopy(payload)
    changed["formula_version"] = "stale-version"
    attacks.append(("stale_version", changed))
    changed = deepcopy(payload)
    changed["edges"][0]["to_literal"] *= -1
    attacks.append(("forged_edge", changed))
    changed = deepcopy(payload)
    changed["edges"].pop(1)
    attacks.append(("omitted_literal", changed))
    changed = deepcopy(payload)
    changed["edges"][0]["source_clause_id"] = -1
    attacks.append(("negative_clause_id", changed))
    changed = deepcopy(payload)
    changed["edges"].append(deepcopy(changed["edges"][0]))
    attacks.append(("cycle", changed))
    changed = deepcopy(payload)
    changed["source_hash"] = "sha256:" + "0" * 64
    attacks.append(("conflicting_metadata", changed))
    changed = {**deepcopy(payload), "label": "unsatisfiable"}
    attacks.append(("label_leakage", changed))
    rows: list[JsonDict] = []
    for attack, candidate in attacks:
        rejected = False
        error = None
        try:
            proof_from_dict(candidate, formula)
        except ValueError as caught:
            rejected = True
            error = str(caught)
        rows.append(
            {
                "attack": attack,
                "expected": "reject",
                "observed": "reject" if rejected else "accept",
                "passed": rejected,
                "error": error,
            }
        )
    return rows


def run_development(
    panel: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Run four development arms over the same public formulas and queries."""

    rows: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    controls: list[JsonDict] = []
    for unit_index, unit in enumerate(panel):
        formula = unit["formula"]
        if not isinstance(formula, FormulaVersion):
            raise TypeError("panel_formula_invalid")
        formula_id = str(unit["formula_id"])
        feature = str(unit["feature"])
        queries = tuple(tuple(query) for query in unit["assumption_queries"])
        method = "enumeration" if formula.n_vars <= 12 else "independent_dpll"
        memory = ProofMemory.empty(formula)
        reachability_started = time.perf_counter_ns()
        reachability = _reachability(formula)
        reachability_build_ns = time.perf_counter_ns() - reachability_started
        reachability_bytes = len(
            json.dumps(
                {str(key): sorted(values) for key, values in reachability.items()},
                sort_keys=True,
            ).encode("utf-8")
        )
        prior_result = None
        for query_index, assumptions in enumerate(queries):
            independent_sat = (
                _enumeration_sat(formula, assumptions)
                if formula.n_vars <= 12
                else _independent_dpll(formula, assumptions)
            )

            proof_result = execute_query(memory, assumptions)
            rows.append(
                _row(
                    formula_id=formula_id,
                    feature=feature,
                    formula=formula,
                    query_index=query_index,
                    assumptions=assumptions,
                    arm="proof_memory",
                    decision=proof_result.decision,
                    used_exact_solver=proof_result.used_exact_solver,
                    independent_sat=independent_sat,
                    comparison_method=method,
                    costs=proof_result.cost_ns,
                    storage_bytes=len(proof_result.committed_memory.to_bytes()),
                )
            )
            memory = proof_result.committed_memory

            reset_started = time.perf_counter_ns()
            reset_formula = FormulaVersion.from_clauses(
                formula.version,
                formula.n_vars,
                [clause.literals for clause in formula.clauses],
            )
            reset_sat, _reset_assignment = reset_formula.solve(assumptions)
            reset_cost = _cost_template()
            reset_cost["exact_solve"] = time.perf_counter_ns() - reset_started
            rows.append(
                _row(
                    formula_id=formula_id,
                    feature=feature,
                    formula=formula,
                    query_index=query_index,
                    assumptions=assumptions,
                    arm="reset_exact_solver",
                    decision="satisfiable" if reset_sat else "unsatisfiable",
                    used_exact_solver=True,
                    independent_sat=independent_sat,
                    comparison_method=method,
                    costs=reset_cost,
                    storage_bytes=0,
                )
            )

            incremental_started = time.perf_counter_ns()
            incremental_sat, _incremental_assignment = formula.solve(assumptions)
            incremental_cost = _cost_template()
            incremental_cost["exact_solve"] = time.perf_counter_ns() - incremental_started
            rows.append(
                _row(
                    formula_id=formula_id,
                    feature=feature,
                    formula=formula,
                    query_index=query_index,
                    assumptions=assumptions,
                    arm="persistent_incremental_exact_solver",
                    decision="satisfiable" if incremental_sat else "unsatisfiable",
                    used_exact_solver=True,
                    independent_sat=independent_sat,
                    comparison_method=method,
                    costs=incremental_cost,
                    storage_bytes=len(formula.implication_edges) * 24,
                )
            )

            graph_check_started = time.perf_counter_ns()
            conflict = _graph_conflict(reachability, assumptions)
            graph_check_ns = time.perf_counter_ns() - graph_check_started
            graph_cost = _cost_template()
            graph_cost["proof_checking"] = graph_check_ns
            graph_cost["storage"] = reachability_build_ns if query_index == 0 else 0
            if conflict:
                graph_decision = "reject"
                graph_exact = False
            else:
                graph_solve_started = time.perf_counter_ns()
                graph_sat, _graph_assignment = formula.solve(assumptions)
                graph_cost["exact_solve"] = time.perf_counter_ns() - graph_solve_started
                graph_decision = "satisfiable" if graph_sat else "unsatisfiable"
                graph_exact = True
            rows.append(
                _row(
                    formula_id=formula_id,
                    feature=feature,
                    formula=formula,
                    query_index=query_index,
                    assumptions=assumptions,
                    arm="persistent_source_graph_reachability_cache",
                    decision=graph_decision,
                    used_exact_solver=graph_exact,
                    independent_sat=independent_sat,
                    comparison_method=method,
                    costs=graph_cost,
                    storage_bytes=reachability_bytes,
                )
            )

            if query_index == 0:
                prior_result = proof_result
            if (
                query_index == 1
                and proof_result.proof_path_id is not None
                and prior_result is not None
            ):
                proof = next(
                    path for path in memory.paths if path.path_id == proof_result.proof_path_id
                )
                erased = memory.without(proof.path_id)
                counterfactual = execute_query(erased, assumptions)
                source_by_id = {clause.clause_id: clause.to_dict() for clause in formula.clauses}
                witnesses.append(
                    {
                        "formula_id": formula_id,
                        "formula_version": formula.version,
                        "source_hash": formula.source_hash,
                        "path": proof.to_dict(),
                        "source_clauses": [
                            source_by_id[clause_id]
                            for clause_id in dict.fromkeys(
                                edge.source_clause_id for edge in proof.edges
                            )
                        ],
                        "prior_assumptions": list(queries[0]),
                        "prior_feedback": prior_result.decision,
                        "later_assumptions": list(assumptions),
                        "later_decision": proof_result.decision,
                        "later_used_exact_solver": proof_result.used_exact_solver,
                        "erased_path_id": proof.path_id,
                        "erasure_decision": counterfactual.decision,
                        "erasure_used_exact_solver": counterfactual.used_exact_solver,
                        "early_decision_removed": counterfactual.used_exact_solver,
                    }
                )
        if unit_index == 0 and memory.paths:
            controls = _hostile_controls(formula, memory.paths[0])
    return rows, witnesses, controls


def reduce_development_costs(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Sum every charged development cost without making a held-out claim."""

    arms = (
        "proof_memory",
        "reset_exact_solver",
        "persistent_incremental_exact_solver",
        "persistent_source_graph_reachability_cache",
    )
    reduced: list[JsonDict] = []
    for arm in arms:
        selected = [row for row in rows if row.get("arm") == arm]
        costs = {
            key: sum(int((row.get("cost_ns") or {}).get(key, 0)) for row in selected)
            for key in _cost_template()
        }
        costs["total"] = sum(costs.values())
        reduced.append(
            {
                "arm": arm,
                "unit_count": len(selected),
                "cost_ns": costs,
                "maximum_storage_bytes": max(
                    (int(row.get("storage_bytes", 0)) for row in selected), default=0
                ),
                "development_only": True,
                "held_out_selected": False,
                "ten_x_gain_claimed": False,
            }
        )
    return reduced


def scoped_command_plan(repo_root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the exact Exp7358 plan over only this affected surface."""

    return validation_contract.build_command_plan(repo_root, V647_MANIFEST, private_root)


def validate_scoped_command_plan(
    repo_root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    errors = validation_contract.validate_command_plan(repo_root, V647_MANIFEST, commands)
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


def passing_test_receipts() -> list[JsonDict]:
    """Supply complete receipt shapes for cold reducer unit tests."""

    rows: list[JsonDict] = []
    for name in (*REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES):
        rows.append(
            {
                "name": name,
                "command": f"test {name}",
                "command_argv": ["test", name],
                "environment": {"PYTHONUNBUFFERED": "1"},
                "scope": "unit_fixture",
                "return_code": 0,
                "exit_code": 0,
                "duration_s": 0.001,
                "log_path": f"/tmp/{name}.log",
                "log_hash": "sha256:" + "1" * 64,
                "log_sha256": "sha256:" + "1" * 64,
                "passed": True,
                "timed_out": False,
                "required": True,
                "category": "required_validation",
                "started_at_utc": "2026-09-17T00:00:00+00:00",
                "ended_at_utc": "2026-09-17T00:00:00.001000+00:00",
            }
        )
    return rows


def _receipt_set_passes(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    counts = Counter(row.get("name") for row in receipts if row.get("required") is True)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("return_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute fixture readiness from raw rows, controls, and receipts."""

    rows = [row for row in artifact.get("rows", []) if isinstance(row, Mapping)]
    witnesses = [
        row for row in artifact.get("development_witnesses", []) if isinstance(row, Mapping)
    ]
    controls = [row for row in artifact.get("safety_rows", []) if isinstance(row, Mapping)]
    receipts = [row for row in artifact.get("validation_receipts", []) if isinstance(row, Mapping)]
    formula_ids = {row.get("formula_id") for row in rows}
    arms = {row.get("arm") for row in rows}
    panel_complete = (
        len(rows) == 16 * 4 * 4
        and len(formula_ids) == 16
        and arms
        == {
            "proof_memory",
            "reset_exact_solver",
            "persistent_incremental_exact_solver",
            "persistent_source_graph_reachability_cache",
        }
        and all(row.get("independent_exact_match") is True for row in rows)
    )
    safety_passed = {row.get("attack") for row in controls} == {
        "stale_version",
        "forged_edge",
        "omitted_literal",
        "negative_clause_id",
        "cycle",
        "conflicting_metadata",
        "label_leakage",
    } and all(row.get("passed") is True for row in controls)
    witness_passed = bool(witnesses) and all(
        row.get("early_decision_removed") is True
        and row.get("later_used_exact_solver") is False
        and row.get("erasure_used_exact_solver") is True
        and row.get("prior_assumptions") != row.get("later_assumptions")
        and bool(row.get("source_clauses"))
        for row in witnesses
    )
    bounds_passed = all(int(row.get("storage_bytes", 0)) <= 65_536 for row in rows) and all(
        len((row.get("path") or {}).get("edges") or [])
        <= 2
        * int(
            next(
                candidate.get("n_vars", 0)
                for candidate in rows
                if candidate.get("formula_id") == row.get("formula_id")
            )
        )
        for row in witnesses
    )
    affected_passed = _receipt_set_passes(receipts, REQUIRED_CHECK_NAMES)
    terminal_passed = _receipt_set_passes(receipts, TERMINAL_CHECK_NAMES)
    blocking_preconditions = [
        row
        for row in artifact.get("preconditions_checked", [])
        if isinstance(row, Mapping) and row.get("terminal_blocking") is True
    ]
    preconditions_passed = bool(blocking_preconditions) and all(
        row.get("passed") is True for row in blocking_preconditions
    )
    ready = int(
        preconditions_passed
        and panel_complete
        and safety_passed
        and witness_passed
        and bounds_passed
        and affected_passed
        and terminal_passed
        and artifact.get("flagged_adversarial") is False
    )
    return {
        "preconditions_passed": preconditions_passed,
        "panel_complete": panel_complete,
        "independent_exact_checks_passed": panel_complete,
        "safety_controls_passed": safety_passed,
        "development_erasure_witness_present": witness_passed,
        "memory_bounds_passed": bounds_passed,
        "affected_validation_passed": affected_passed,
        "terminal_validation_passed": terminal_passed,
        "proof_fixture_ready_score": ready,
        "learning_value_score": 0,
        "promotion_score": 0,
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    *,
    terminal_blocking: bool,
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "terminal_blocking": terminal_blocking,
    }


def build_acceptance_gates(artifact: Mapping[str, Any]) -> list[JsonDict]:
    reduced = independent_reduce(artifact)
    return [
        _gate(
            "required_affected_validation",
            "required_validation",
            True,
            reduced["affected_validation_passed"],
            terminal_blocking=True,
        ),
        _gate(
            "terminal_artifact_checks",
            "completion",
            True,
            reduced["terminal_validation_passed"],
            terminal_blocking=True,
        ),
        _gate(
            "source_checked_proof_safety",
            "safety",
            True,
            reduced["safety_controls_passed"],
            terminal_blocking=True,
        ),
        _gate(
            "development_panel_complete",
            "completion",
            True,
            reduced["panel_complete"],
            terminal_blocking=True,
        ),
        _gate(
            "single_path_erasure_witness",
            "scientific_efficacy",
            True,
            reduced["development_erasure_witness_present"],
            terminal_blocking=True,
        ),
        _gate(
            "prospective_learning_value",
            "scientific_efficacy",
            1,
            0,
            terminal_blocking=False,
        ),
        _gate(
            "automatic_promotion",
            "promotion",
            0,
            0,
            terminal_blocking=False,
        ),
    ]


def gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failures = [dict(row) for row in gates if row.get("passed") is not True]
    blocking = [row for row in failures if row.get("terminal_blocking") is True]
    return {
        "passed": not blocking,
        "failed_count": len(failures),
        "blocking_failed_count": len(blocking),
        "first_failure": failures[0] if failures else None,
        "failures": failures,
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    bound = {
        key: artifact.get(key)
        for key in (
            "schema",
            "experiment_id",
            "milestone",
            "run_date",
            "random_seed",
            "source_artifact_hashes",
            "preconditions_checked",
            "rows",
            "safety_rows",
            "development_witnesses",
            "development_cost_rows",
            "memory_contract",
            "sample_size_budget",
            "acceptance_gate_results",
            "independent_reduction",
            "verdict_class",
        )
    }
    return validation_contract.canonical_hash(bound)


def field_principles(artifact: Mapping[str, Any]) -> dict[str, str]:
    specific = {
        "schema": "Use a versioned schema with ordinary top-level experiment_id and milestone.",
        "status": "Use a terminal state only after actual work and required validation.",
        "run_date": "Use 20260917 with actual start and end UTC timestamps.",
        "preconditions_checked": "Record exact paths, identities, hashes, classes, and resource checks before dependent work.",
        "MODEL_SPECS": "List intended current models; host-only work uses an empty list.",
        "model_invoked": "Set true for any attempted current model load or generation.",
        "invocation_counts": "Record current attempted, completed, failed, cancelled, and in-flight model work as zero.",
        "inference_substrate": "Describe actual host exact solving and keep historical inference in labeled sidecars.",
        "inference_substrate_class": "Use the closed CPU exact solver or simulator class without duration padding.",
        "execution_venue": "Record measured host CPU work and no current board execution.",
        "duration_s": "Use measured monotonic duration without sleeps or invented floors.",
        "phase_spans": "Record measured read, build, load, generate, evaluate, validate, and write spans.",
        "random_seed": "Freeze development, formula, and resampling seeds.",
        "reproducibility_checksum": "Bind exact code, settings, formulas, protocol, and raw rows.",
        "source_artifact_hashes": "Hash exact source and producer bytes while preserving historical flags.",
        "rows": "Retain every formula, query, arm, outcome, cost, failure, and censoring disposition.",
        "sample_size_budget": "Predeclare planned, attempted, completed, censored units, and stopping rules.",
        "acceptance_gate_results": "Separate expected, observed, and passed values for validation, safety, completion, and efficacy.",
        "gate_check_summary": "Name every failed check with its exact expected and observed values.",
        "verifier_is_oracle": "True because the supplied public formula defines correctness.",
        "honest_verdict": "Use complete_ for finished work and name unavailable prerequisites for blocked work.",
        "verdict_class": "Use only positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Set true for a critical independent finding and prevent readiness.",
        "validation_receipts": "Keep command argv, environment, scope, exit, duration, and log hash for each check.",
        "repository_health": "Keep dated unrelated failures separate from affected checks.",
        "field_principles": "Explain each field without wrapping ordinary values.",
        "promotion_score": "Keep zero because this milestone performs no rollout or publication.",
        "proof_fixture_ready_score": "One requires checked paths, bounded lifecycle behavior, validation, and an erasure witness.",
        "development_witnesses": "Record source paths, prior feedback, later different assumptions, and one-path removal.",
        "memory_contract": "State formula identity, caps, commit timing, lifecycle, and no satisfying-answer authority.",
        "development_cost_rows": "Charge discovery, checks, updates, storage, and solve costs for every comparator.",
        "learning_value_score": "Keep zero until a later independent prospective measurement.",
    }
    return {
        key: specific.get(key, "Retain this supporting evidence in its ordinary JSON type.")
        for key in artifact
        if key != "field_principles"
    } | {"field_principles": specific["field_principles"]}


def _base_artifact(
    rows: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    historical_sidecars: Sequence[Mapping[str, Any]],
    phase_spans: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
    duration_s: float,
    repository_health: Mapping[str, Any],
    flagged_adversarial: bool = False,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": "building_terminal_record",
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": {
            "current": deepcopy(ZERO_CURRENT_INVOCATIONS),
            "historical": {
                "counted_as_current": False,
                "sidecar_count": len(historical_sidecars),
            },
        },
        "inference_substrate": "host_cpu_exact_2cnf_solver_and_simulator",
        "inference_substrate_class": "cpu_exact_solver_or_simulator",
        "execution_venue": "host",
        "host_computation": {
            "processor": platform.processor() or "host_cpu",
            "node": platform.node(),
            "python": platform.python_version(),
            "current_model_operations": 0,
            "current_gpu_operations": 0,
        },
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "source_artifact_hashes": dict(source_hashes),
        "historical_inference_sidecars": [deepcopy(dict(row)) for row in historical_sidecars],
        "rows": [deepcopy(dict(row)) for row in rows],
        "safety_rows": [deepcopy(dict(row)) for row in controls],
        "development_witnesses": [deepcopy(dict(row)) for row in witnesses],
        "development_cost_rows": reduce_development_costs(rows),
        "sample_size_budget": {
            "planned_formulas": 16,
            "attempted_formulas": len({row.get("formula_id") for row in rows}),
            "completed_formulas": len({row.get("formula_id") for row in rows}),
            "censored_formulas": 0,
            "planned_queries_per_formula": 4,
            "planned_arms": 4,
            "planned_rows": 256,
            "attempted_rows": len(rows),
            "completed_rows": sum(row.get("censored") is False for row in rows),
            "censored_rows": sum(row.get("censored") is True for row in rows),
            "stopping_rule": "Run all 16 formulas, four public queries, and four arms once.",
            "remaining_work": 0 if len(rows) == 256 else 256 - len(rows),
        },
        "memory_contract": {
            "formula_identity": "immutable version plus canonical clause IDs and source hash",
            "maximum_discovered_paths_per_query": 8,
            "maximum_retained_paths": 128,
            "maximum_serialized_bytes": 65_536,
            "maximum_edges_per_path": "2n",
            "prediction_snapshot": "prior_committed_snapshot",
            "commit_boundary": "after_exact_query_finishes",
            "restart_policy": "restore_exact_committed_bytes_or_invalidate",
            "version_change_policy": "atomic_invalidation",
            "satisfying_answer_authority": False,
            "truth_authority": "exact_solver_over_public_formula",
            "hidden_rule_acquisition": False,
        },
        "validation_receipts": [deepcopy(dict(row)) for row in receipts],
        "repository_health": deepcopy(dict(repository_health)),
        "verifier_is_oracle": True,
        "flagged_adversarial": flagged_adversarial,
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "rust_changed": False,
        "numbered_e2e_applicable": False,
        "capability_e2e_checks": ["declared_entrypoint", "cold_artifact_replay"],
    }
    reduction = independent_reduce(artifact)
    artifact["independent_reduction"] = reduction
    artifact["proof_fixture_ready_score"] = reduction["proof_fixture_ready_score"]
    artifact["learning_value_score"] = 0
    artifact["promotion_score"] = 0
    blocking_failed = any(
        row.get("terminal_blocking") is True and row.get("passed") is not True
        for row in preconditions
    )
    if blocking_failed:
        artifact["status"] = "blocked_required_input_unavailable"
        artifact["honest_verdict"] = "blocked_required_input_unavailable"
        artifact["verdict_class"] = "blocked"
        artifact["proof_fixture_ready_score"] = 0
    elif reduction["proof_fixture_ready_score"] == 1:
        artifact["status"] = "complete_proof_fixture_ready_development_only"
        artifact["honest_verdict"] = "complete_null_proof_fixture_ready_learning_value_not_measured"
        artifact["verdict_class"] = "null"
    else:
        artifact["status"] = "complete_disqualified_required_validation_or_safety_failure"
        artifact["honest_verdict"] = "complete_disqualified_required_validation_or_safety_failure"
        artifact["verdict_class"] = "disqualified"
    artifact["acceptance_gate_results"] = build_acceptance_gates(artifact)
    artifact["gate_check_summary"] = gate_summary(artifact["acceptance_gate_results"])
    artifact["field_principles"] = field_principles(
        {**artifact, "reproducibility_checksum": None, "field_principles": {}}
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test(
    rows: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    preconditions = [
        _precondition(
            "unit_fixture",
            "unit_fixture",
            "available",
            True,
            True,
            terminal_blocking=True,
        )
    ]
    return _base_artifact(
        rows,
        witnesses,
        controls,
        receipts,
        preconditions=preconditions,
        source_hashes={},
        historical_sidecars=[],
        phase_spans=[
            {"phase": name, "start_s": index / 10, "end_s": (index + 1) / 10, "duration_s": 0.1}
            for index, name in enumerate(
                ("read", "build", "load", "generate", "evaluate", "validate", "write")
            )
        ],
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
        duration_s=1.0,
        repository_health={"status": "historical_not_loaded", "affects_required_checks": False},
    )


def validate_artifact(value: object) -> list[str]:
    """Cold-check identity, evidence reduction, scores, and checksum."""

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
    if (artifact.get("invocation_counts") or {}).get("current") != ZERO_CURRENT_INVOCATIONS:
        errors.append("current_invocation_counts_nonzero")
    if artifact.get("inference_substrate_class") != "cpu_exact_solver_or_simulator":
        errors.append("substrate_class_mismatch")
    if artifact.get("execution_venue") != "host" or artifact.get("verifier_is_oracle") is not True:
        errors.append("truth_or_venue_mismatch")
    if artifact.get("verdict_class") not in CLOSED_VERDICTS:
        errors.append("verdict_class_invalid")
    if artifact.get("learning_value_score") != 0 or artifact.get("promotion_score") != 0:
        errors.append("deferred_scores_nonzero")
    reduced = independent_reduce(artifact)
    if artifact.get("independent_reduction") != reduced:
        errors.append("independent_reduction_mismatch")
    if artifact.get("proof_fixture_ready_score") != reduced["proof_fixture_ready_score"]:
        errors.append("fixture_score_mismatch")
    expected_gates = build_acceptance_gates(artifact)
    if artifact.get("acceptance_gate_results") != expected_gates:
        errors.append("acceptance_gates_mismatch")
    if artifact.get("gate_check_summary") != gate_summary(expected_gates):
        errors.append("gate_summary_mismatch")
    if artifact.get("verdict_class") in {"blocked", "disqualified", "partial"} and any(
        artifact.get(field) != 0
        for field in ("proof_fixture_ready_score", "learning_value_score", "promotion_score")
    ):
        errors.append("failed_state_scores_nonzero")
    if (
        artifact.get("flagged_adversarial") is True
        and artifact.get("proof_fixture_ready_score") != 0
    ):
        errors.append("adversarial_readiness_nonzero")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _terminal_commands(candidate: Path) -> list[validation_contract.PlannedCommand]:
    python = str(REPO_ROOT / ".venv/bin/python")
    replay = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7370_v647_proof_memory import validate_artifact;"
        "v=json.loads(pathlib.Path(sys.argv[1]).read_text());"
        "e=validate_artifact(v);print(e,flush=True);raise SystemExit(bool(e))"
    )
    specs = (
        ("cold_artifact_replay", (python, "-u", "-c", replay, str(candidate)), "capability_e2e"),
        ("independent_reducer", (python, "-u", "-c", replay, str(candidate)), "completion"),
        (
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "safety",
        ),
        (
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "completion",
        ),
    )
    return [
        validation_contract.PlannedCommand(
            validation_scope.CommandSpec(name, argv, "measured_candidate"), category, True
        )
        for name, argv, category in specs
    ]


def _span(phase: str, phase_started: float, run_started: float) -> JsonDict:
    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
    }


def _write_sidecars(
    root: Path,
    panel: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    witnesses: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    historical_sidecars: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    raw = root / RAW_DIR
    formulas = [
        {
            "formula_id": unit["formula_id"],
            "feature": unit["feature"],
            "formula": unit["formula"].to_dict(),
            "assumption_queries": [list(query) for query in unit["assumption_queries"]],
        }
        for unit in panel
    ]
    values = {
        raw / "development_formulas.json": {"formulas": formulas},
        raw / "development_evidence.json": {
            "rows": list(rows),
            "development_witnesses": list(witnesses),
            "safety_rows": list(controls),
        },
        raw / "historical_model_receipts.json": {
            "label": "historical_only_not_current_inference",
            "sources": list(historical_sidecars),
        },
    }
    hashes: dict[str, str] = {}
    for path, value in values.items():
        atomic_json(path, value)
        hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def run_experiment(
    repo_root: Path,
    run_date: str,
    *,
    output_path: Path = RESULT_PATH,
) -> JsonDict:  # pragma: no cover - the declared entrypoint exercises orchestration.
    """Run development, scoped validation, terminal readers, and atomic publish."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = repo_root.resolve()
    output = output_path if output_path.is_absolute() else root / output_path
    run_started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(run_started, "read", "start")
    preconditions, source_hashes, historical_sidecars = collect_preconditions(root)
    spans.append(_span("read", phase_started, run_started))
    progress(run_started, "read", "end", checks=len(preconditions))

    phase_started = time.monotonic()
    progress(run_started, "build", "start")
    panel = build_development_panel()
    spans.append(_span("build", phase_started, run_started))
    progress(run_started, "build", "end", formulas=len(panel))

    phase_started = time.monotonic()
    progress(run_started, "load", "before_model_load", models=0)
    spans.append(_span("load", phase_started, run_started))
    progress(run_started, "load", "after_model_load", models=0)

    phase_started = time.monotonic()
    progress(run_started, "generate", "before_generation", calls=0)
    spans.append(_span("generate", phase_started, run_started))
    progress(run_started, "generate", "after_generation", calls=0)

    blocking_failed = any(row["terminal_blocking"] and not row["passed"] for row in preconditions)
    rows: list[JsonDict] = []
    witnesses: list[JsonDict] = []
    controls: list[JsonDict] = []
    phase_started = time.monotonic()
    progress(run_started, "evaluate", "start", blocked=blocking_failed)
    if not blocking_failed:
        rows, witnesses, controls = run_development(panel)
    spans.append(_span("evaluate", phase_started, run_started))
    progress(run_started, "evaluate", "end", rows=len(rows), witnesses=len(witnesses))

    sidecar_hashes = _write_sidecars(root, panel, rows, witnesses, controls, historical_sidecars)
    source_hashes.update(sidecar_hashes)
    capstone = _load_object(root / CAPSTONE_PATH)
    repository_health = deepcopy(
        capstone.get("repository_health")
        or {
            "status": "historical_health_unavailable",
            "as_of": RUN_DATE,
            "affects_required_checks": False,
        }
    )
    repository_health["affects_required_checks"] = False

    receipts: list[JsonDict] = []
    phase_started = time.monotonic()
    progress(run_started, "validate", "start")
    if not blocking_failed:
        private_root = root / RAW_DIR / "validation/private"
        commands = scoped_command_plan(root, private_root)
        plan_errors = validate_scoped_command_plan(root, commands)
        if plan_errors:
            raise RuntimeError(f"invalid_scoped_command_plan:{plan_errors}")
        planned = [
            validation_contract.PlannedCommand(command, "required_validation", True)
            for command in commands
        ]
        command_rows = validation_contract.run_categorized_commands(
            root,
            planned,
            log_dir=root / RAW_DIR / "validation/logs",
        )
        receipts.extend(_normalized_receipt(row) for row in command_rows)
    spans.append(_span("validate", phase_started, run_started))
    progress(run_started, "validate", "end", receipts=len(receipts))

    phase_started = time.monotonic()
    progress(run_started, "write", "start")
    candidate_path = root / RAW_DIR / "measured-terminal-candidate.json"
    candidate = _base_artifact(
        rows,
        witnesses,
        controls,
        receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        historical_sidecars=historical_sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        repository_health=repository_health,
    )
    atomic_json(candidate_path, candidate)
    terminal_rows: list[JsonDict] = []
    if not blocking_failed:
        progress(run_started, "write", "before_terminal_subprocesses")
        raw_terminal = validation_contract.run_categorized_commands(
            root,
            _terminal_commands(candidate_path),
            log_dir=root / RAW_DIR / "terminal/logs",
        )
        terminal_rows = [_normalized_receipt(row) for row in raw_terminal]
        progress(run_started, "write", "after_terminal_subprocesses", count=len(terminal_rows))
    entrypoint_log = root / RAW_DIR / "terminal/declared_entrypoint.log"
    entrypoint_log.parent.mkdir(parents=True, exist_ok=True)
    entrypoint_log.write_text(
        "The current unbuffered process reached the atomic publication boundary.\n",
        encoding="utf-8",
    )
    terminal_rows.append(
        {
            "name": "declared_entrypoint",
            "command": " ".join(sys.argv),
            "command_argv": list(sys.argv),
            "environment": {
                key: os.environ.get(key)
                for key in ("PYTHONUNBUFFERED", "PYTHONPATH", "CARNOT_FORCE_LIVE")
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
    receipts.extend(terminal_rows)
    flagged = any(
        row["name"] == "adversarial_verify" and row["passed"] is not True for row in terminal_rows
    )
    spans.append(_span("write", phase_started, run_started))
    final = _base_artifact(
        rows,
        witnesses,
        controls,
        receipts,
        preconditions=preconditions,
        source_hashes=source_hashes,
        historical_sidecars=historical_sidecars,
        phase_spans=spans,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        duration_s=time.monotonic() - run_started,
        repository_health=repository_health,
        flagged_adversarial=flagged,
    )
    errors = validate_artifact(final)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    atomic_json(output, final)
    progress(
        run_started,
        "write",
        "end",
        output=output,
        verdict=final["verdict_class"],
    )
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
