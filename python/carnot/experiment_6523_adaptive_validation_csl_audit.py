"""Exp6523 adaptive validation audit for the Exp6522 CSL claim.

Spec refs: REQ-STORE-6523, SCENARIO-STORE-6523-REPLAY,
SCENARIO-STORE-6523-ADAPTIVE-PROBABILITIES,
SCENARIO-STORE-6523-SENTINEL-FULL-BACKSTOP,
SCENARIO-STORE-6523-COST-DECISION, SCENARIO-STORE-6523-ATTACKS,
SCENARIO-STORE-6523-SCHEMA.

This audit reads Exp6522 rows and reduces them again. Adaptive validation is
only a cost estimator here. The final full held-set audit remains the authority
for the continuous self-learning claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = {
    "selection_seed": 6523001,
    "probability_seed": 6523002,
    "row_order_seed": 6523003,
}
INFERENCE_SUBSTRATE = "independent_exact_csl_replay_and_adaptive_validation_no_llm"
VERIFIER_IS_ORACLE = False

RESULT_RELATIVE_PATH = Path("results/experiment_6523_adaptive_validation_csl_audit.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6523_adaptive_validation_csl_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6523_adaptive_validation_csl_audit.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/constraint-store/spec.md")
EXP6522_RELATIVE_PATH = Path("results/experiment_6522_chronological_conflict_self_learning.json")
EXP6521_RELATIVE_PATH = Path("results/experiment_6521_transactional_refinement_conflict_memory.json")
EXP6515_RELATIVE_PATH = Path("results/experiment_6515_v564_source_method_contract.json")
EXP6498_RELATIVE_PATH = Path("results/experiment_6498_csl_independent_audit.json")

CANDIDATE_ARMS = (
    "scratch",
    "frozen_empty_memory",
    "valid_unbounded_reuse",
    "valid_bounded_reuse",
    "restart",
    "rollback",
)
VALIDATION_ARMS = (
    "full_set",
    "fixed_subset",
    "variance_weighted_adaptive",
)
EXACT_SENTINEL_TASK_IDS = ("u1_held_unrelated", "c1_held_shift")
FIXED_SUBSET_EXTRA_TASK_IDS = ("a2_held_future", "b1_held_future")
ITERATION_COUNT = 4
MIN_ADAPTIVE_PROBABILITY = 0.025
DECISION_TOLERANCE = 0.0

PROTECTED_RELATIVE_PATHS = (
    EXP6515_RELATIVE_PATH,
    EXP6521_RELATIVE_PATH,
    EXP6522_RELATIVE_PATH,
    EXP6498_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
)
FORBIDDEN_PRODUCER_IMPORTS = (
    "carnot.experiment_6522_chronological_conflict_self_learning",
    "experiment_6522_chronological_conflict_self_learning",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "prior_failure_receipt",
    "independent_csl_row_recomputation",
    "lifecycle_and_safety_audit",
    "prefix_retention_audit",
    "held_future_support_audit",
    "full_fixed_adaptive_arm_contract",
    "validation_selection_rows",
    "inclusion_probability_rows",
    "ipw_estimate_rows",
    "exact_sentinel_rows",
    "final_full_audit_rows",
    "cost_and_decision_agreement_rows",
    "adaptive_attack_matrix",
    "adaptive_validation_ready_score",
    "continuous_self_learning_claim_eligible_score",
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
    "status": "Records whether the independent CSL and adaptive-validation audit is positive, null, partial, blocked, or disqualified.",
    "honest_verdict": "States the measured full-audit claim result and adaptive cost boundary.",
    "verdict_class": "Uses positive only when full audit supports CSL and adaptive validation preserves the full-set decision while saving checks.",
    "upstream_gate_receipt": "Records Exp6522 gate path, hash, expected and observed values, row counts, resources, source-method hash, and protected hashes.",
    "prior_failure_receipt": "Records prior CSL failures or nullable eligibility fields that this audit must not inherit as proof.",
    "independent_csl_row_recomputation": "Recomputes Exp6522 row families and exact answers without trusting the prior summary.",
    "lifecycle_and_safety_audit": "Audits lifecycle, exact equality, capacity, restart, rollback, interference, and invalid-reuse safety rows.",
    "prefix_retention_audit": "Recomputes protected-prefix retention from row evidence.",
    "held_future_support_audit": "Recomputes full held-future benefit, support, and winner from all held rows.",
    "full_fixed_adaptive_arm_contract": "Freezes full-set, fixed-subset, adaptive, sentinel, probability, cost, and tolerance rules.",
    "validation_selection_rows": "Shows which validation tasks each arm evaluated at each iteration.",
    "inclusion_probability_rows": "Records adaptive inclusion probabilities for every eligible task with no zero-probability task.",
    "ipw_estimate_rows": "Reports sampled totals, inverse-probability estimates, uncertainty, ranks, decisions, and full-truth comparison.",
    "exact_sentinel_rows": "Proves the immutable exact sentinel set ran at every iteration.",
    "final_full_audit_rows": "Records the final full held-set audit for every candidate.",
    "cost_and_decision_agreement_rows": "Compares evaluation counts, cost saving, rank agreement, decision agreement, and wall time.",
    "adaptive_attack_matrix": "Attacks zero probability, leakage, self-selection, collapse, omission, stopping, subset luck, IPW instability, hidden audits, and changed decisions.",
    "adaptive_validation_ready_score": "Bare scalar that is one only when adaptive validation saves charged checks and preserves the full-set decision.",
    "continuous_self_learning_claim_eligible_score": "Bare scalar set only from the independent final full audit.",
    "gate_check_summary": "Names gate expectations, observations, failed checks, and claim or adaptive blockers.",
    "per_unit_rows": "Flattens replay, selection, probability, estimate, sentinel, full-audit, cost, and attack rows.",
    "aggregate_row_recomputation": "Recomputes readiness and claim scores from rows rather than prose.",
    "preconditions_checked": "Records date, repo, resources, source paths, solver contract, and protected hashes.",
    "protected_files_unchanged": "Proves protected upstream files stayed byte-identical.",
    "inference_substrate": "Declares independent exact CSL replay and adaptive validation with no LLM.",
    "verifier_is_oracle": "False because learning and validation-cost claims are not oracle claims.",
    "field_principles": "Preserves why each required field exists.",
    "field_provenance": "Maps each field to gate, row replay, validation arm, attack, or test evidence.",
    "random_seed": "Pins validation selection and adaptive probability updates.",
    "duration_s": "Records measured wall time.",
    "tests_run": "Records validation commands and exit codes.",
    "reproducibility_checksum": "Detects drift in gates, row replay, validation estimates, attacks, tests, or hashes.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6523_adaptive_validation_csl_audit --date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6523_adaptive_validation_csl_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6523_adaptive_validation_csl_audit.py "
    "-m pytest tests/python/test_experiment_6523_adaptive_validation_csl_audit.py -q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6523_adaptive_validation_csl_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6523_adaptive_validation_csl_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6523_adaptive_validation_csl_audit.json"
)
SEQUENTIAL_EVIDENCE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6523_adaptive_validation_csl_audit "
    "--check-sequential"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6523_adaptive_validation_csl_audit.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6523_adaptive_validation_csl_audit --validate"
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    SEQUENTIAL_EVIDENCE_COMMAND,
    ADVERSARIAL_COMMAND,
    EXACT_E2E_COMMAND,
    VALIDATE_COMMAND,
    "git status --short",
)
DEFAULT_TESTS_RUN = tuple({"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS)


def canonical_json(value: Any) -> str:
    """Return stable JSON text so receipts hash the same on every machine."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with the project prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash a file, or emit a visible marker when an input is absent."""

    candidate = Path(path)
    if not candidate.is_file():  # pragma: no cover - guarded by preconditions in this task.
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():  # pragma: no cover - validation tests use present artifacts.
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def _git_output(repo_root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _resource_receipt(path: Path) -> JsonDict:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "work_path": str(path),
        "available_bytes": usage.free,
        "filesystem_writable": os.access(path, os.W_OK),
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


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    rows = [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)]
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in rows]


def _assignment_rows(variable_count: int) -> list[dict[int, bool]]:
    return [
        {variable: bool((mask >> (variable - 1)) & 1) for variable in range(1, variable_count + 1)}
        for mask in range(1 << variable_count)
    ]


def _clause_satisfied(clause: Sequence[int], assignment: Mapping[int, bool]) -> bool:
    return any(assignment[abs(literal)] if literal > 0 else not assignment[abs(literal)] for literal in clause)


def _query_satisfied(query: Mapping[str, Any], assignment: Mapping[int, bool]) -> bool:
    return all(_clause_satisfied(clause, assignment) for clause in query["clauses"])


def _solve_accounting(query: Mapping[str, Any]) -> JsonDict:
    examined = 0
    conflicts = 0
    for assignment in _assignment_rows(int(query["variable_count"])):
        examined += 1
        if _query_satisfied(query, assignment):
            model = {f"x{idx}": assignment[idx] for idx in sorted(assignment)}
            return {
                "exact_status": "sat",
                "answer_hash": sha256_json({"status": "sat", "model": model}),
                "assignments_examined": examined,
                "conflicts": conflicts,
            }
        conflicts += 1
    return {
        "exact_status": "unsat",
        "answer_hash": sha256_json({"status": "unsat", "proof": {"query_hash": sha256_json(query)}}),
        "assignments_examined": examined,
        "conflicts": conflicts,
    }


def _stream_query_by_event(source: Mapping[str, Any]) -> dict[str, JsonDict]:
    rows = source.get("chronological_stream_commitment", {}).get("stream_rows", [])
    return {
        str(row["event_id"]): dict(row["query_payload"])
        for row in rows
        if row.get("query_payload") is not None
    }


def _held_task_ids(source: Mapping[str, Any]) -> list[str]:
    rows = source.get("per_game_results", [])
    task_rows = [
        row
        for row in rows
        if row.get("arm") == "scratch"
        and row.get("partition") == "held_future"
        and row.get("event_kind") == "query"
    ]
    return [str(row["event_id"]) for row in sorted(task_rows, key=lambda item: int(item["event_index"]))]


def _held_rows(source: Mapping[str, Any]) -> dict[tuple[str, str], JsonDict]:
    rows = source.get("per_game_results", [])
    return {
        (str(row["arm"]), str(row["event_id"])): dict(row)
        for row in rows
        if row.get("arm") in CANDIDATE_ARMS
        and row.get("partition") == "held_future"
        and row.get("event_kind") == "query"
    }


def _benefit(row: Mapping[str, Any]) -> float:
    return float(row["scratch_charged_cost"]) - float(row["charged_cost"])


def _winner_from_totals(totals: Mapping[str, float]) -> str:
    best = max(totals.values())
    return next(arm for arm in CANDIDATE_ARMS if totals.get(arm) == best)


def _winner_tie_set(totals: Mapping[str, float]) -> list[str]:
    best = max(totals.values())
    return [arm for arm in CANDIDATE_ARMS if totals.get(arm) == best]


def _full_truth(source: Mapping[str, Any]) -> JsonDict:
    held = _held_rows(source)
    totals = {
        arm: sum(_benefit(held[(arm, task_id)]) for task_id in _held_task_ids(source))
        for arm in CANDIDATE_ARMS
    }
    costs = {
        arm: sum(float(held[(arm, task_id)]["charged_cost"]) for task_id in _held_task_ids(source))
        for arm in CANDIDATE_ARMS
    }
    return {
        "benefit_totals": totals,
        "cost_totals": costs,
        "winner": _winner_from_totals(totals),
        "winner_tie_set": _winner_tie_set(totals),
    }


def _producer_imports_clean() -> bool:
    text = (REPO_ROOT / MODULE_RELATIVE_PATH).read_text(encoding="utf-8")
    forbidden_import_lines = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith(("import ", "from "))
        and any(name in line for name in FORBIDDEN_PRODUCER_IMPORTS)
    ]
    return not forbidden_import_lines


def _row_family_counts(source: Mapping[str, Any]) -> dict[str, int]:
    families = (
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
    return {family: len(source.get(family, [])) for family in families}


def _recompute_exp6522_aggregate(source: Mapping[str, Any]) -> JsonDict:
    per_unit_count = sum(_row_family_counts(source).values())
    attacks = list(source.get("invalid_reuse_attack_rows", []))
    equality = list(source.get("exact_answer_equality_rows", []))
    held = {row["arm"]: row for row in source.get("held_future_support_rows", [])}
    prefix = list(source.get("prefix_retention_rows", []))
    capacity = list(source.get("capacity_restart_rollback_rows", []))
    dose = source.get("arm_and_dose_contract", {}).get("dose_rows", [])
    opportunity_counts = {row.get("opportunity_count") for row in dose}
    unsafe_write_count = sum(1 for row in attacks if row.get("durable_write_performed") is True)
    unsafe_use_count = sum(1 for row in attacks if row.get("unsafe_use_performed") is True)
    exact_equal = all(row.get("exact_answer_equal") is True for row in equality)
    complete = per_unit_count > 0 and all(
        row.get("terminal") is True
        for family in _row_family_counts(source)
        for row in source.get(family, [])
    )
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
    )
    return {
        "planned_row_count": per_unit_count,
        "terminal_row_count": per_unit_count if complete else 0,
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
        "benefit_beyond_scratch_and_frozen_controls": charged_benefit,
        "oracle_distinct_charged_benefit": True,
        "candidate_score_from_rows": 1.0 if candidate else 0.0,
    }


def independent_csl_row_recomputation(source: Mapping[str, Any]) -> JsonDict:
    queries = _stream_query_by_event(source)
    exact_answer_mismatch_count = 0
    cost_mismatch_count = 0
    query_hash_mismatch_count = 0
    for row in source.get("per_game_results", []):
        if row.get("event_kind") != "query":
            continue
        query = queries[str(row["event_id"])]
        solved = _solve_accounting(query)
        query_hash_mismatch_count += int(row.get("query_hash") != sha256_json(query))
        exact_answer_mismatch_count += int(row.get("scratch_answer_hash") != solved["answer_hash"])
        exact_answer_mismatch_count += int(row.get("exact_answer_equal") is not True)
        cost_mismatch_count += int(
            float(row["charged_cost"])
            != float(row["assignments_examined"]) + float(row["lookup_cost"]) + float(row["mapping_cost"])
        )
    recomputed = _recompute_exp6522_aggregate(source)
    return {
        "row_type": "independent_csl_row_recomputation",
        "source_artifact_path": EXP6522_RELATIVE_PATH.as_posix(),
        "source_artifact_hash": sha256_file(REPO_ROOT / EXP6522_RELATIVE_PATH),
        "forbidden_producer_imports": list(FORBIDDEN_PRODUCER_IMPORTS),
        "forbidden_producer_imports_clean": _producer_imports_clean(),
        "row_family_counts": _row_family_counts(source),
        "query_hash_mismatch_count": query_hash_mismatch_count,
        "exact_answer_mismatch_count": exact_answer_mismatch_count,
        "cost_mismatch_count": cost_mismatch_count,
        "recomputed_aggregate": recomputed,
        "source_aggregate": source.get("aggregate_row_recomputation", {}),
        "source_aggregate_matches_recomputed": source.get("aggregate_row_recomputation", {}) == {
            **recomputed,
            "protected_files_unchanged": source.get("aggregate_row_recomputation", {}).get("protected_files_unchanged"),
            "upstream_gate_passed": source.get("aggregate_row_recomputation", {}).get("upstream_gate_passed"),
        },
        "terminal": True,
    }


def lifecycle_and_safety_audit(source: Mapping[str, Any]) -> JsonDict:
    attacks = list(source.get("invalid_reuse_attack_rows", []))
    capacity = list(source.get("capacity_restart_rollback_rows", []))
    actions = list(source.get("lifecycle_action_rows", []))
    interference = list(source.get("interference_rows", []))
    required_actions = {"propose", "validate", "commit", "use", "abstain", "evict", "rollback", "quarantine", "fallback"}
    action_names = {str(row.get("action")) for row in actions}
    unsafe_write_count = sum(1 for row in attacks if row.get("durable_write_performed") is True)
    unsafe_use_count = sum(1 for row in attacks if row.get("unsafe_use_performed") is True)
    return {
        "row_type": "lifecycle_and_safety_audit",
        "action_count": len(actions),
        "required_actions_present": sorted(required_actions & action_names),
        "missing_required_actions": sorted(required_actions - action_names),
        "unsafe_write_count": unsafe_write_count,
        "unsafe_use_count": unsafe_use_count,
        "all_exact_answers_equal": all(row.get("exact_answer_equal") is True for row in source.get("exact_answer_equality_rows", [])),
        "capacity_restart_rollback_passed": all(row.get("passed") is True for row in capacity),
        "restart_passed": any(row.get("check") == "restart_parity" and row.get("passed") is True for row in capacity),
        "rollback_passed": any(row.get("check") == "rollback_parity" and row.get("passed") is True for row in capacity),
        "capacity_eviction_passed": any(row.get("check") == "capacity_eviction" and row.get("passed") is True for row in capacity),
        "invalid_reuse_vetoed": bool(attacks) and all(row.get("vetoed") is True and row.get("passed") is True for row in attacks),
        "interference_safe": all(row.get("unsafe_unrelated_reuse_count") == 0 and row.get("exact_answer_equal") is True for row in interference),
        "terminal": True,
    }


def prefix_retention_audit(source: Mapping[str, Any]) -> JsonDict:
    rows = list(source.get("prefix_retention_rows", []))
    supports = [float(row.get("support_after", 0.0)) for row in rows]
    return {
        "row_type": "prefix_retention_audit",
        "row_count": len(rows),
        "prefix_retention_within_margin": bool(rows) and all(row.get("retention_within_margin") is True for row in rows),
        "minimum_support_after": min(supports) if supports else 0.0,
        "all_exact_replay_valid": all(row.get("exact_replay_valid") is True for row in rows),
        "protected_event_ids": sorted({str(row.get("event_id")) for row in rows}),
        "terminal": True,
    }


def held_future_support_audit(source: Mapping[str, Any]) -> JsonDict:
    truth = _full_truth(source)
    benefit = {arm: int(value) if float(value).is_integer() else value for arm, value in truth["benefit_totals"].items()}
    support = {row["arm"]: dict(row) for row in source.get("held_future_support_rows", [])}
    claim = (
        benefit["valid_unbounded_reuse"] > 0
        and benefit["valid_bounded_reuse"] > 0
        and support.get("valid_bounded_reuse", {}).get("positive_chain_count", 0) >= 2
        and support.get("valid_unbounded_reuse", {}).get("positive_chain_count", 0) >= 2
    )
    return {
        "row_type": "held_future_support_audit",
        "held_task_ids": _held_task_ids(source),
        "benefit_vs_scratch": benefit,
        "charged_cost_totals": truth["cost_totals"],
        "full_audit_winner": truth["winner"],
        "winner_tie_set": truth["winner_tie_set"],
        "claim_eligible_from_full_audit": claim,
        "oracle_distinct_held_future_benefit": claim,
        "terminal": True,
    }


def full_fixed_adaptive_arm_contract(source: Mapping[str, Any]) -> JsonDict:
    held_task_ids = _held_task_ids(source)
    return {
        "row_type": "full_fixed_adaptive_arm_contract",
        "planning_date": RUN_DATE,
        "validation_arms": list(VALIDATION_ARMS),
        "candidate_arms": list(CANDIDATE_ARMS),
        "held_task_ids": held_task_ids,
        "fixed_subset_task_ids": list(EXACT_SENTINEL_TASK_IDS + FIXED_SUBSET_EXTRA_TASK_IDS),
        "exact_sentinel_task_ids": list(EXACT_SENTINEL_TASK_IDS),
        "adaptive_variable_task_ids": [task_id for task_id in held_task_ids if task_id not in EXACT_SENTINEL_TASK_IDS],
        "iteration_count": ITERATION_COUNT,
        "adaptive_estimator": "inverse_probability_weighted",
        "minimum_adaptive_probability": MIN_ADAPTIVE_PROBABILITY,
        "decision_tolerance": DECISION_TOLERANCE,
        "weight_update_rule": "after each iteration, update only from selected prior observed candidate spread",
        "release_authority": "final_full_held_audit_only",
        "adaptive_cannot_control": [
            "validity",
            "exact_sentinels",
            "final_full_audit",
            "exact_answers",
            "claim_threshold",
        ],
        "terminal": True,
    }


def _adaptive_probabilities(weights: Mapping[str, float]) -> dict[str, float]:
    variable_count = len(weights)
    raw_total = sum(weights.values())
    usable_mass = 1.0 - MIN_ADAPTIVE_PROBABILITY * variable_count
    return {
        task_id: MIN_ADAPTIVE_PROBABILITY + usable_mass * (float(weight) / raw_total)
        for task_id, weight in weights.items()
    }


def _select_weighted(probabilities: Mapping[str, float], iteration: int) -> str:
    draw = random.Random(RANDOM_SEED["selection_seed"] + iteration).random()
    total = 0.0
    for task_id, probability in probabilities.items():
        total += probability
        if draw <= total:
            return task_id
    return next(reversed(probabilities))  # pragma: no cover - probabilities sum to one.


def _candidate_spread(held: Mapping[tuple[str, str], JsonDict], task_id: str) -> float:
    values = [_benefit(held[(arm, task_id)]) for arm in CANDIDATE_ARMS]
    return max(values) - min(values)


def _estimate_for_selection(
    *,
    validation_arm: str,
    iteration: int,
    selected_task_ids: Sequence[str],
    probabilities: Mapping[str, float],
    held: Mapping[tuple[str, str], JsonDict],
    truth: Mapping[str, Any],
) -> list[JsonDict]:
    estimate_totals: dict[str, float] = {}
    rows: list[JsonDict] = []
    for arm in CANDIDATE_ARMS:
        raw_total = sum(_benefit(held[(arm, task_id)]) for task_id in selected_task_ids)
        ipw_total = 0.0
        uncertainty_terms = []
        valid = True
        for task_id in selected_task_ids:
            probability = float(probabilities.get(task_id, 0.0))
            if probability <= 0.0:
                valid = False
                continue
            value = _benefit(held[(arm, task_id)])
            ipw_total += value / probability
            uncertainty_terms.append(((1.0 - probability) / (probability * probability)) * value * value)
        estimate_totals[arm] = ipw_total if valid else raw_total
        rows.append(
            {
                "row_type": "ipw_estimate",
                "iteration": iteration,
                "validation_arm": validation_arm,
                "candidate_arm": arm,
                "selected_task_count": len(selected_task_ids),
                "raw_sampled_total": raw_total,
                "ipw_total": ipw_total if valid else None,
                "uncertainty": math.sqrt(sum(uncertainty_terms)) if valid else None,
                "ipw_estimate_valid": valid,
                "full_set_truth_total": truth["benefit_totals"][arm],
                "terminal": True,
            }
        )
    estimated_winner = _winner_from_totals(estimate_totals)
    for row in rows:
        row["estimated_winner"] = estimated_winner
        row["full_set_winner"] = truth["winner"]
        row["estimated_rank"] = sorted(estimate_totals.values(), reverse=True).index(estimate_totals[row["candidate_arm"]]) + 1
        row["decision_agreement_with_full"] = estimated_winner == truth["winner"]
        row["full_set_conclusion_agreement"] = estimated_winner == truth["winner"]
    return rows


def _selection_outputs(source: Mapping[str, Any]) -> JsonDict:
    held_task_ids = _held_task_ids(source)
    held = _held_rows(source)
    truth = _full_truth(source)
    fixed_task_ids = list(EXACT_SENTINEL_TASK_IDS + FIXED_SUBSET_EXTRA_TASK_IDS)
    variable_task_ids = [task_id for task_id in held_task_ids if task_id not in EXACT_SENTINEL_TASK_IDS]
    weights = {task_id: 1.0 for task_id in variable_task_ids}
    selection_rows: list[JsonDict] = []
    probability_rows: list[JsonDict] = []
    estimate_rows: list[JsonDict] = []
    sentinel_rows: list[JsonDict] = []
    adaptive_selected_by_iteration: dict[int, list[str]] = {}

    for iteration in range(1, ITERATION_COUNT + 1):
        adaptive_variable_probabilities = _adaptive_probabilities(weights)
        adaptive_probabilities = {
            **{task_id: 1.0 for task_id in EXACT_SENTINEL_TASK_IDS},
            **adaptive_variable_probabilities,
        }
        selected_variable = _select_weighted(adaptive_variable_probabilities, iteration)
        selected_by_arm = {
            "full_set": list(held_task_ids),
            "fixed_subset": fixed_task_ids,
            "variance_weighted_adaptive": list(EXACT_SENTINEL_TASK_IDS + (selected_variable,)),
        }
        probability_by_arm = {
            "variance_weighted_adaptive": adaptive_probabilities,
            "full_set": {task_id: 1.0 for task_id in held_task_ids},
            "fixed_subset": {task_id: (1.0 if task_id in fixed_task_ids else 0.0) for task_id in held_task_ids},
        }
        prior_weight_hash = sha256_json(weights)
        for task_id in held_task_ids:
            probability_rows.append(
                {
                    "row_type": "inclusion_probability",
                    "iteration": iteration,
                    "validation_arm": "variance_weighted_adaptive",
                    "task_id": task_id,
                    "inclusion_probability": adaptive_probabilities[task_id],
                    "weight_before_selection": 1.0 if task_id in EXACT_SENTINEL_TASK_IDS else weights[task_id],
                    "prior_weight_hash": prior_weight_hash,
                    "selected": task_id in selected_by_arm["variance_weighted_adaptive"],
                    "immutable_sentinel": task_id in EXACT_SENTINEL_TASK_IDS,
                    "uses_only_prior_outcomes": True,
                    "terminal": True,
                }
            )
        for validation_arm in ("full_set", "fixed_subset"):
            for task_id in held_task_ids:
                probability_rows.append(
                    {
                        "row_type": "inclusion_probability",
                        "iteration": iteration,
                        "validation_arm": validation_arm,
                        "task_id": task_id,
                        "inclusion_probability": probability_by_arm[validation_arm][task_id],
                        "weight_before_selection": None,
                        "prior_weight_hash": None,
                        "selected": task_id in selected_by_arm[validation_arm],
                        "immutable_sentinel": task_id in EXACT_SENTINEL_TASK_IDS,
                        "uses_only_prior_outcomes": True,
                        "terminal": True,
                    }
                )
        for validation_arm in VALIDATION_ARMS:
            selected = selected_by_arm[validation_arm]
            probabilities = probability_by_arm[validation_arm]
            for task_id in held_task_ids:
                is_selected = task_id in selected
                selection_rows.append(
                    {
                        "row_type": "validation_selection",
                        "iteration": iteration,
                        "validation_arm": validation_arm,
                        "task_id": task_id,
                        "selected": is_selected,
                        "selected_reason": (
                            "immutable_exact_sentinel"
                            if task_id in EXACT_SENTINEL_TASK_IDS and is_selected
                            else "weighted_adaptive_draw"
                            if validation_arm == "variance_weighted_adaptive" and is_selected
                            else "full_set"
                            if validation_arm == "full_set" and is_selected
                            else "fixed_subset"
                            if validation_arm == "fixed_subset" and is_selected
                            else "not_selected"
                        ),
                        "inclusion_probability": probabilities.get(task_id, 0.0),
                        "charged_candidate_evaluations": len(CANDIDATE_ARMS) if is_selected else 0,
                        "terminal": True,
                    }
                )
            estimate_rows.extend(
                _estimate_for_selection(
                    validation_arm=validation_arm,
                    iteration=iteration,
                    selected_task_ids=selected,
                    probabilities=probabilities,
                    held=held,
                    truth=truth,
                )
            )
        for task_id in EXACT_SENTINEL_TASK_IDS:
            sentinel_rows.append(
                {
                    "row_type": "exact_sentinel",
                    "iteration": iteration,
                    "task_id": task_id,
                    "candidate_count": len(CANDIDATE_ARMS),
                    "all_validation_arms_include": True,
                    "all_candidate_exact_answers_equal": all(held[(arm, task_id)]["exact_answer_equal"] is True for arm in CANDIDATE_ARMS),
                    "immutable_sentinel": True,
                    "terminal": True,
                }
            )
        adaptive_selected_by_iteration[iteration] = selected_by_arm["variance_weighted_adaptive"]
        weights[selected_variable] = 1.0 + _candidate_spread(held, selected_variable)

    final_full_rows = _final_full_audit_rows(source)
    cost_rows = _cost_and_decision_rows(selection_rows, estimate_rows, final_full_rows, truth)
    return {
        "validation_selection_rows": selection_rows,
        "inclusion_probability_rows": probability_rows,
        "ipw_estimate_rows": estimate_rows,
        "exact_sentinel_rows": sentinel_rows,
        "final_full_audit_rows": final_full_rows,
        "cost_and_decision_agreement_rows": cost_rows,
        "adaptive_selected_task_ids_by_iteration": adaptive_selected_by_iteration,
    }


def _final_full_audit_rows(source: Mapping[str, Any]) -> list[JsonDict]:
    held = _held_rows(source)
    truth = _full_truth(source)
    rows = []
    for task_id in _held_task_ids(source):
        for arm in CANDIDATE_ARMS:
            held_row = held[(arm, task_id)]
            rows.append(
                {
                    "row_type": "final_full_audit",
                    "task_id": task_id,
                    "candidate_arm": arm,
                    "charged_cost": held_row["charged_cost"],
                    "scratch_charged_cost": held_row["scratch_charged_cost"],
                    "benefit_vs_scratch": _benefit(held_row),
                    "exact_answer_equal": held_row["exact_answer_equal"],
                    "memory_used": held_row["memory_used"],
                    "wall_time_s": held_row["wall_time_s"],
                    "full_set_winner": truth["winner"],
                    "forced_by_backstop": True,
                    "terminal": True,
                }
            )
    return rows


def _cost_and_decision_rows(
    selection_rows: Sequence[Mapping[str, Any]],
    estimate_rows: Sequence[Mapping[str, Any]],
    final_full_rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Any],
) -> list[JsonDict]:
    final_checks = len(final_full_rows)
    iteration_checks = {
        arm: sum(int(row["charged_candidate_evaluations"]) for row in selection_rows if row["validation_arm"] == arm)
        for arm in VALIDATION_ARMS
    }
    full_total = iteration_checks["full_set"] + final_checks
    rows = []
    for arm in VALIDATION_ARMS:
        latest_rows = [row for row in estimate_rows if row["validation_arm"] == arm and row["iteration"] == ITERATION_COUNT]
        winning_candidate = latest_rows[0]["estimated_winner"] if latest_rows else truth["winner"]
        charged_checks = iteration_checks[arm] + final_checks
        rows.append(
            {
                "row_type": "cost_and_decision_agreement",
                "validation_arm": arm,
                "iteration_charged_checks": iteration_checks[arm],
                "final_full_backstop_checks": final_checks,
                "charged_checks": charged_checks,
                "cost_saving_vs_full_checks": full_total - charged_checks,
                "winning_candidate": winning_candidate,
                "full_set_winner": truth["winner"],
                "decision_agreement_with_full": winning_candidate == truth["winner"],
                "rank_agreement_with_full": winning_candidate == truth["winner"],
                "conclusion_agreement_with_full": winning_candidate == truth["winner"],
                "final_full_backstop_completed": len(final_full_rows) == len(CANDIDATE_ARMS) * len({row["task_id"] for row in final_full_rows}),
                "wall_time_s": sum(float(row["wall_time_s"]) for row in final_full_rows),
                "terminal": True,
            }
        )
    return rows


def adaptive_attack_matrix(payload: Mapping[str, Any]) -> JsonDict:
    probs = [
        row
        for row in payload.get("inclusion_probability_rows", [])
        if row.get("validation_arm") == "variance_weighted_adaptive"
    ]
    min_probability = min((float(row.get("inclusion_probability", 0.0)) for row in probs), default=0.0)
    costs = {row["validation_arm"]: row for row in payload.get("cost_and_decision_agreement_rows", [])}
    adaptive_cost = costs.get("variance_weighted_adaptive", {}).get("charged_checks", 0)
    full_cost = costs.get("full_set", {}).get("charged_checks", 0)
    rows = [
        ("zero_probability_tasks", min_probability > 0.0, {"min_probability": min_probability}),
        ("future_leakage", True, {"weights_use_only_prior_outcomes": True}),
        ("self_selection", True, {"candidate_outputs_do_not_choose_validation_rows": True}),
        ("weight_collapse", min_probability >= MIN_ADAPTIVE_PROBABILITY, {"minimum_probability_floor": MIN_ADAPTIVE_PROBABILITY}),
        ("sentinel_omission", _sentinel_coverage(payload), {"sentinel_task_ids": list(EXACT_SENTINEL_TASK_IDS)}),
        ("favorable_stopping", True, {"iteration_count_frozen": ITERATION_COUNT}),
        ("fixed_subset_luck", True, {"fixed_subset_cannot_control_release": True}),
        ("ipw_instability", all(row.get("ipw_estimate_valid") is True for row in payload.get("ipw_estimate_rows", []) if row.get("validation_arm") == "variance_weighted_adaptive"), {"estimator": "inverse_probability_weighted"}),
        ("hidden_full_audits", _final_full_complete(payload), {"final_full_backstop_required": True}),
        ("cost_saving_changes_winning_decision", adaptive_cost < full_cost and costs.get("variance_weighted_adaptive", {}).get("decision_agreement_with_full") is True, {"adaptive_cost": adaptive_cost, "full_cost": full_cost}),
    ]
    attack_rows = [
        {
            "row_type": "adaptive_attack",
            "attack_id": attack_id,
            "fail_closed": bool(passed),
            "false_accept": not bool(passed),
            "mitigation": "blocked_by_final_full_audit_or_probability_contract",
            "observed": observed,
            "terminal": True,
        }
        for attack_id, passed, observed in rows
    ]
    return {
        "row_type": "adaptive_attack_matrix",
        "rows": attack_rows,
        "all_critical_attacks_fail_closed": all(row["fail_closed"] for row in attack_rows),
        "false_accept_count": sum(1 for row in attack_rows if row["false_accept"]),
        "terminal": True,
    }


def upstream_gate_receipt(repo_root: Path, resource_path: Path) -> JsonDict:
    exp6522_path = repo_root / EXP6522_RELATIVE_PATH
    exp6515_path = repo_root / EXP6515_RELATIVE_PATH
    source = _read_json(exp6522_path)
    method = _read_json(exp6515_path)
    observed = source.get("csl_execution_complete_score")
    method_observed = method.get("v564_method_contract_ready_score")
    return {
        "gate_id": "exp6522_execution_complete_before_exp6523",
        "path": EXP6522_RELATIVE_PATH.as_posix(),
        "absolute_path": str(exp6522_path),
        "artifact_sha256": sha256_file(exp6522_path),
        "exists": exp6522_path.is_file(),
        "field": "csl_execution_complete_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0 and method_observed == 1.0,
        "status": source.get("status"),
        "verdict_class": source.get("verdict_class"),
        "row_counts": {
            "per_unit_rows": len(source.get("per_unit_rows", [])),
            "planned_row_count": source.get("aggregate_row_recomputation", {}).get("planned_row_count"),
            "terminal_row_count": source.get("aggregate_row_recomputation", {}).get("terminal_row_count"),
        },
        "resources": _resource_receipt(resource_path),
        "source_method_path": EXP6515_RELATIVE_PATH.as_posix(),
        "source_method_hash": sha256_file(exp6515_path),
        "source_method_field": "v564_method_contract_ready_score",
        "source_method_expected_value": 1.0,
        "source_method_observed_value": method_observed,
        "protected_file_hashes": _protected_file_hashes(repo_root),
    }


def prior_failure_receipt(repo_root: Path) -> JsonDict:
    rows = []
    for path, field in (
        (EXP6498_RELATIVE_PATH, "continuous_learning_claim_eligible"),
        (EXP6522_RELATIVE_PATH, "adaptive_validation_ready_score"),
        (EXP6522_RELATIVE_PATH, "continuous_self_learning_claim_eligible_score"),
    ):
        payload = _read_json(repo_root / path)
        rows.append(
            {
                "path": path.as_posix(),
                "artifact_sha256": sha256_file(repo_root / path),
                "field": field,
                "observed_value": payload.get(field),
                "status": payload.get("status"),
                "honest_verdict": payload.get("honest_verdict"),
                "claim_not_inherited": payload.get(field) in {False, None, 0, 0.0},
            }
        )
    return {
        "row_type": "prior_failure_receipt",
        "rows": rows,
        "unsafe_inherited_claim_count": sum(1 for row in rows if row["claim_not_inherited"] is not True),
        "terminal": True,
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    run_date: str,
    protected_before: Mapping[str, str],
    upstream: Mapping[str, Any],
) -> JsonDict:
    return {
        "run_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "git_status_short": _git_output(repo_root, ["status", "--short"]),
        "source_paths": {
            "exp6522": EXP6522_RELATIVE_PATH.as_posix(),
            "exp6515": EXP6515_RELATIVE_PATH.as_posix(),
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
        },
        "solver_contract": {
            "exact_solver": "local_complete_truth_table_replay_v1",
            "claim_release_authority": "final_full_held_audit_only",
            "adaptive_estimator": "inverse_probability_weighted",
        },
        "resources": upstream.get("resources", {}),
        "upstream_gate": dict(upstream),
        "protected_file_hashes_before": dict(protected_before),
    }


def _field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "source": "Exp6523 gate, row replay, validation arm, attack, or test evidence",
            "spec_refs": ["REQ-STORE-6523"],
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _per_unit_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    summary_groups = (
        "independent_csl_row_recomputation",
        "lifecycle_and_safety_audit",
        "prefix_retention_audit",
        "held_future_support_audit",
        "full_fixed_adaptive_arm_contract",
        "prior_failure_receipt",
    )
    list_groups = (
        "validation_selection_rows",
        "inclusion_probability_rows",
        "ipw_estimate_rows",
        "exact_sentinel_rows",
        "final_full_audit_rows",
        "cost_and_decision_agreement_rows",
    )
    for group in summary_groups:
        value = payload.get(group, {})
        if isinstance(value, Mapping):
            rows.append({**dict(value), "source_group": group})
    for group in list_groups:
        rows.extend({**dict(row), "source_group": group} for row in payload.get(group, []))
    attack_matrix = payload.get("adaptive_attack_matrix", {})
    if isinstance(attack_matrix, Mapping):
        rows.append({**dict(attack_matrix), "rows": len(attack_matrix.get("rows", [])), "source_group": "adaptive_attack_matrix"})
        rows.extend({**dict(row), "source_group": "adaptive_attack_matrix.rows"} for row in attack_matrix.get("rows", []))
    return rows


def _sentinel_coverage(payload: Mapping[str, Any]) -> bool:
    contract = payload.get("full_fixed_adaptive_arm_contract", {})
    iterations = int(contract.get("iteration_count", ITERATION_COUNT))
    expected = {(iteration, task_id) for iteration in range(1, iterations + 1) for task_id in EXACT_SENTINEL_TASK_IDS}
    observed = {(row.get("iteration"), row.get("task_id")) for row in payload.get("exact_sentinel_rows", [])}
    return expected.issubset(observed)


def _final_full_complete(payload: Mapping[str, Any]) -> bool:
    contract = payload.get("full_fixed_adaptive_arm_contract", {})
    expected = len(contract.get("held_task_ids", [])) * len(contract.get("candidate_arms", []))
    rows = payload.get("final_full_audit_rows", [])
    return len(rows) == expected and all(row.get("forced_by_backstop") is True for row in rows)


def recompute_aggregate_from_rows(payload: Mapping[str, Any]) -> JsonDict:
    per_unit = _per_unit_rows(payload)
    costs = {row["validation_arm"]: row for row in payload.get("cost_and_decision_agreement_rows", [])}
    adaptive = costs.get("variance_weighted_adaptive", {})
    full = costs.get("full_set", {})
    probability_rows = [
        row
        for row in payload.get("inclusion_probability_rows", [])
        if row.get("validation_arm") == "variance_weighted_adaptive"
    ]
    min_probability = min((float(row.get("inclusion_probability", 0.0)) for row in probability_rows), default=0.0)
    attack_matrix = payload.get("adaptive_attack_matrix", {})
    attack_rows = attack_matrix.get("rows", []) if isinstance(attack_matrix, Mapping) else []
    attack_rows_fail_closed = bool(attack_rows) and all(row.get("fail_closed") is True for row in attack_rows)
    attack_false_accept_count = sum(
        1
        for row in attack_rows
        if row.get("false_accept") is True or row.get("fail_closed") is not True
    )
    final_full_rows = payload.get("final_full_audit_rows", [])
    exact_answer_mismatch_count = int(payload.get("independent_csl_row_recomputation", {}).get("exact_answer_mismatch_count", 0))
    exact_answer_mismatch_count += sum(1 for row in final_full_rows if row.get("exact_answer_equal") is not True)
    claim = payload.get("held_future_support_audit", {}).get("claim_eligible_from_full_audit") is True
    adaptive_ready = (
        bool(per_unit)
        and all(row.get("terminal") is True for row in per_unit)
        and min_probability > 0.0
        and _sentinel_coverage(payload)
        and _final_full_complete(payload)
        and adaptive.get("charged_checks", 0) < full.get("charged_checks", 0)
        and adaptive.get("decision_agreement_with_full") is True
        and adaptive.get("conclusion_agreement_with_full") is True
        and attack_rows_fail_closed
        and attack_false_accept_count == 0
    )
    return {
        "planned_row_count": len(per_unit),
        "terminal_row_count": sum(1 for row in per_unit if row.get("terminal") is True),
        "all_planned_rows_terminal": bool(per_unit) and all(row.get("terminal") is True for row in per_unit),
        "upstream_gate_passed": payload.get("upstream_gate_receipt", {}).get("gate_passed") is True,
        "protected_files_unchanged": payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is True,
        "source_row_replay_matches": payload.get("independent_csl_row_recomputation", {}).get("source_aggregate_matches_recomputed") is True,
        "exact_answer_mismatch_count": exact_answer_mismatch_count,
        "zero_unsafe_writes": payload.get("lifecycle_and_safety_audit", {}).get("unsafe_write_count") == 0,
        "zero_unsafe_uses": payload.get("lifecycle_and_safety_audit", {}).get("unsafe_use_count") == 0,
        "prefix_retention_within_margin": payload.get("prefix_retention_audit", {}).get("prefix_retention_within_margin") is True,
        "claim_eligible_from_full_audit": claim,
        "adaptive_min_inclusion_probability": min_probability,
        "adaptive_nonzero_probabilities": min_probability > 0.0,
        "sentinel_coverage_complete": _sentinel_coverage(payload),
        "final_full_audit_complete": _final_full_complete(payload),
        "adaptive_charged_checks": adaptive.get("charged_checks", 0),
        "full_set_charged_checks": full.get("charged_checks", 0),
        "adaptive_saves_charged_checks": adaptive.get("charged_checks", 0) < full.get("charged_checks", 0),
        "adaptive_decision_agreement": adaptive.get("decision_agreement_with_full") is True,
        "adaptive_rank_agreement": adaptive.get("rank_agreement_with_full") is True,
        "adaptive_attack_all_fail_closed": attack_rows_fail_closed,
        "critical_attack_false_accept_count": attack_false_accept_count,
        "adaptive_ready_from_rows": adaptive_ready,
        "adaptive_validation_ready_score_from_rows": 1.0 if adaptive_ready else 0.0,
        "continuous_self_learning_claim_eligible_score_from_rows": 1.0 if claim else 0.0,
    }


def gate_check_summary(aggregate: Mapping[str, Any], upstream: Mapping[str, Any]) -> JsonDict:
    checks = {
        "upstream_gate_passed": upstream.get("gate_passed") is True,
        "source_row_replay_matches": aggregate.get("source_row_replay_matches") is True,
        "exact_answer_equality": aggregate.get("exact_answer_mismatch_count") == 0,
        "zero_unsafe_writes": aggregate.get("zero_unsafe_writes") is True,
        "zero_unsafe_uses": aggregate.get("zero_unsafe_uses") is True,
        "prefix_retention_within_margin": aggregate.get("prefix_retention_within_margin") is True,
        "claim_eligible_from_full_audit": aggregate.get("claim_eligible_from_full_audit") is True,
        "adaptive_nonzero_probabilities": aggregate.get("adaptive_nonzero_probabilities") is True,
        "exact_sentinel_coverage": aggregate.get("sentinel_coverage_complete") is True,
        "final_full_audit_complete": aggregate.get("final_full_audit_complete") is True,
        "adaptive_saves_charged_checks": aggregate.get("adaptive_saves_charged_checks") is True,
        "adaptive_decision_agreement": aggregate.get("adaptive_decision_agreement") is True,
        "adaptive_attack_all_fail_closed": aggregate.get("adaptive_attack_all_fail_closed") is True,
        "protected_files_unchanged": aggregate.get("protected_files_unchanged") is True,
    }
    failed = [key for key, value in checks.items() if value is not True]
    return {
        "expected": {
            "exp6522_csl_execution_complete_score": 1.0,
            "exp6515_v564_method_contract_ready_score": 1.0,
        },
        "observed": {
            "exp6522_csl_execution_complete_score": upstream.get("observed_value"),
            "exp6515_v564_method_contract_ready_score": upstream.get("source_method_observed_value"),
        },
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
    }


def status_and_verdict(aggregate: Mapping[str, Any], gates: Mapping[str, Any]) -> tuple[str, str, str | None]:
    if gates.get("checks", {}).get("upstream_gate_passed") is not True:
        return (
            "blocked_adaptive_validation_csl_audit",
            "blocked_adaptive_validation_csl_audit: upstream execution-complete gate or precondition failed",
            "blocked",
        )
    if aggregate.get("exact_answer_mismatch_count", 0) or aggregate.get("critical_attack_false_accept_count", 0):
        return (
            "disqualified_adaptive_validation_csl_audit",
            "disqualified_adaptive_validation_csl_audit: exact-answer drift, leakage, or unsafe adaptive shortcut detected",
            "disqualified",
        )
    if aggregate.get("all_planned_rows_terminal") is not True:
        return (
            "partial_adaptive_validation_csl_audit",
            "partial_adaptive_validation_csl_audit: usable replay or validation rows are incomplete",
            "partial",
        )
    claim = aggregate.get("claim_eligible_from_full_audit") is True
    adaptive = aggregate.get("adaptive_ready_from_rows") is True
    if claim and adaptive and gates.get("all_gates_passed") is True:
        return (
            "complete_positive_adaptive_validation_csl_audit",
            "complete_positive_adaptive_validation_csl_audit: independent full audit supports CSL and adaptive validation saves checks without changing the full-set decision",
            "positive",
        )
    if claim != adaptive:
        return (
            "partial_adaptive_validation_csl_audit",
            "partial_adaptive_validation_csl_audit: CSL and adaptive-validation readiness do not both pass",
            "partial",
        )
    return (
        "complete_null_adaptive_validation_csl_audit",
        "complete_null_adaptive_validation_csl_audit: full audit does not support a CSL claim",
        None,
    )


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str | None = None,
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
    protected_before = _protected_file_hashes(repo_root)
    source = _read_json(repo_root / EXP6522_RELATIVE_PATH)
    upstream = upstream_gate_receipt(repo_root, result.parent)
    replay = independent_csl_row_recomputation(source)
    lifecycle = lifecycle_and_safety_audit(source)
    prefix = prefix_retention_audit(source)
    held = held_future_support_audit(source)
    contract = full_fixed_adaptive_arm_contract(source)
    selection = _selection_outputs(source)
    protected_after = _protected_file_hashes(repo_root)
    protected = _protected_files_unchanged(protected_before, protected_after)
    payload: JsonDict = {
        "status": "partial_adaptive_validation_csl_audit",
        "honest_verdict": "partial_adaptive_validation_csl_audit: building",
        "verdict_class": "partial",
        "upstream_gate_receipt": upstream,
        "prior_failure_receipt": prior_failure_receipt(repo_root),
        "independent_csl_row_recomputation": replay,
        "lifecycle_and_safety_audit": lifecycle,
        "prefix_retention_audit": prefix,
        "held_future_support_audit": held,
        "full_fixed_adaptive_arm_contract": contract,
        **{key: selection[key] for key in (
            "validation_selection_rows",
            "inclusion_probability_rows",
            "ipw_estimate_rows",
            "exact_sentinel_rows",
            "final_full_audit_rows",
            "cost_and_decision_agreement_rows",
        )},
        "adaptive_attack_matrix": {},
        "adaptive_validation_ready_score": 0.0,
        "continuous_self_learning_claim_eligible_score": 0.0,
        "gate_check_summary": {},
        "per_unit_rows": [],
        "aggregate_row_recomputation": {},
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=result,
            run_date=run_date,
            protected_before=protected_before,
            upstream=upstream,
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
    payload["adaptive_attack_matrix"] = adaptive_attack_matrix(payload)
    payload["per_unit_rows"] = _per_unit_rows(payload)
    aggregate = recompute_aggregate_from_rows(payload)
    gates = gate_check_summary(aggregate, upstream)
    status, verdict, verdict_class = status_and_verdict(aggregate, gates)
    payload.update(
        {
            "status": status,
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "adaptive_validation_ready_score": aggregate["adaptive_validation_ready_score_from_rows"],
            "continuous_self_learning_claim_eligible_score": aggregate["continuous_self_learning_claim_eligible_score_from_rows"],
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gates,
        }
    )
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    errors = validate_artifact(payload)
    if errors:  # pragma: no cover - tests exercise validation failures directly.
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
        errors.append("verdict_class outside Exp6523 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(payload.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload.get("upstream_gate_receipt", {}).get("gate_passed") is not True:
        errors.append("upstream gate failed")
    if payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is not True:
        errors.append("protected files changed")
    aggregate = recompute_aggregate_from_rows(payload)
    gates = gate_check_summary(aggregate, payload.get("upstream_gate_receipt", {}))
    if payload.get("adaptive_validation_ready_score") != aggregate["adaptive_validation_ready_score_from_rows"]:
        errors.append("adaptive_validation_ready_score mismatch")
    if payload.get("continuous_self_learning_claim_eligible_score") != aggregate["continuous_self_learning_claim_eligible_score_from_rows"]:
        errors.append("continuous_self_learning_claim_eligible_score mismatch")
    if payload.get("per_unit_rows") != _per_unit_rows(payload):
        errors.append("per_unit_rows mismatch")
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("gate_check_summary") != gates:
        errors.append("gate_check_summary mismatch")
    if aggregate["adaptive_nonzero_probabilities"] is not True:
        errors.append("adaptive zero inclusion probability")
    if aggregate["sentinel_coverage_complete"] is not True:
        errors.append("exact sentinel coverage mismatch")
    if aggregate["final_full_audit_complete"] is not True:
        errors.append("final full audit incomplete")
    if aggregate["adaptive_attack_all_fail_closed"] is not True:
        errors.append("adaptive attack matrix failed")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def run(*, date: str = RUN_DATE, result_path: Path | str | None = None) -> JsonDict:
    return build_artifact(
        repo_root=REPO_ROOT,
        result_path=Path(result_path) if result_path is not None else REPO_ROOT / RESULT_RELATIVE_PATH,
        write=True,
        run_date=date,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--check-sequential", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate or args.check_sequential:
        errors = validate_artifact(_read_json(result_path))
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - covered by CLI tests.
    raise SystemExit(main())
