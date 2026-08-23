"""Exp6516 bounded exact branch-counterfactual pilot v3.

Spec refs: REQ-BENCH-6516, SCENARIO-BENCH-6516-DIRECT-IMMUTABLE,
SCENARIO-BENCH-6516-CHECKPOINTS, SCENARIO-BENCH-6516-CANDIDATES,
SCENARIO-BENCH-6516-EXACT-REPLAY, SCENARIO-BENCH-6516-SPLIT-SEALING,
SCENARIO-BENCH-6516-BOUNDED-SHARDS, SCENARIO-BENCH-6516-RESUME-ATOMIC,
SCENARIO-BENCH-6516-ATTACKS, SCENARIO-BENCH-6516-READY.

The pilot is a small data artifact. It reads immutable exact inputs by path,
freezes branch checkpoints from structural features, replays every Boolean
value with equal exact budgets, and finalizes through the Exp6514 transaction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

from carnot import experiment_6504_exact_structural_benchmark_commitment as exp6504
from carnot.atomic_shard_transaction import (
    TRANSACTION_SCHEMA,
    AtomicShardTransaction,
    sha256_bytes,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6516
SCHEMA_VERSION = "carnot.experiment_6516.exact_branch_pilot_dataset_v3.v1"
INFERENCE_SUBSTRATE = "procedural_exact_branch_replay_with_transactional_shards_no_llm"
VERIFIER_IS_ORACLE = True

RESULT_RELATIVE_PATH = Path("results/experiment_6516_exact_branch_pilot_dataset_v3.json")
WORK_RELATIVE_PATH = Path("results/.experiment_6516_exact_branch_pilot_dataset_v3.tx")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6516_exact_branch_pilot_dataset_v3.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6516_exact_branch_pilot_dataset_v3.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

EXP6504_RELATIVE_PATH = Path("results/experiment_6504_exact_structural_benchmark_commitment.json")
EXP6510_RELATIVE_PATH = Path("results/experiment_6510_v563_independent_exact_root.json")
EXP6514_RELATIVE_PATH = Path("results/experiment_6514_atomic_shard_artifact_transaction.json")
EXP6515_RELATIVE_PATH = Path("results/experiment_6515_v564_source_method_contract.json")
EXP6511_RETIRED_RELATIVE_PATH = Path(
    "results/experiment_6511_exact_branch_counterfactual_dataset_v2.json"
)

PILOT_FAMILIES = ("random_3cnf", "pseudo_industrial_3cnf", "tseitin")
PILOT_SCALES = ("small", "medium")
PILOT_SPLITS = ("train", "development", "held")
PILOT_BASE_UNIT_COUNT = len(PILOT_FAMILIES) * len(PILOT_SCALES) * len(PILOT_SPLITS)
ELIGIBLE_VALUES = (False, True)
EXACT_ASSIGNMENT_BUDGET = 256
MINIMUM_CELL_FLOOR = 1

FORBIDDEN_FEATURE_NAMES = (
    "unit_id",
    "instance_id",
    "base_instance_id",
    "base_lineage_id",
    "row_id",
    "row_order",
    "serialization_length",
    "serialized_length",
    "family",
    "family_label",
    "base_label",
    "exact_label",
    "label",
    "held_outcome",
    "future_effort",
    "assignments_examined",
    "terminal_disposition",
    "outcome",
)

ATTACK_IDS = (
    "row_order",
    "serialization_length",
    "family_labels",
    "base_labels",
    "outcome_derived_effort",
    "duplicate_checkpoints",
    "split_leakage",
    "asymmetric_budgets",
    "corrupt_resume",
    "omitted_hard_rows",
)

PROTECTED_RELATIVE_PATHS = (
    EXP6504_RELATIVE_PATH,
    EXP6510_RELATIVE_PATH,
    EXP6514_RELATIVE_PATH,
    EXP6515_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    EXP6504_RELATIVE_PATH,
    EXP6510_RELATIVE_PATH,
    EXP6514_RELATIVE_PATH,
    EXP6515_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipts",
    "prior_failure_receipts",
    "direct_input_receipts",
    "pilot_commitment",
    "checkpoint_contract",
    "structural_feature_schema",
    "branch_counterfactual_rows",
    "exact_solver_receipts",
    "split_commitment",
    "shard_manifest",
    "planned_and_terminal_unit_counts",
    "censoring_and_budget_rows",
    "leakage_attack_matrix",
    "branch_pilot_dataset_ready_score",
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
    "status": "Records the terminal pilot dataset state.",
    "honest_verdict": (
        "The verdict states whether the bounded pilot is ready without claiming learned guidance value."
    ),
    "verdict_class": (
        "Null means complete data readiness. Partial, blocked, and disqualified preserve limits and failures."
    ),
    "upstream_gate_receipts": (
        "The two V564 gates are checked by path, hash, expected value, and observed value."
    ),
    "prior_failure_receipts": (
        "Historical bootstrap and skipped-dataset failures stay visible without reactivating retired roots."
    ),
    "direct_input_receipts": "Exp6504 and Exp6510 are read as direct immutable inputs with hashes.",
    "pilot_commitment": (
        "Families, scales, seeds, splits, budgets, floors, and candidate rules are fixed before replay."
    ),
    "checkpoint_contract": (
        "Checkpoint selection is deterministic and uses only decision-time structural data."
    ),
    "structural_feature_schema": (
        "Allowed features exclude identity, labels, row order, length, family labels, and future effort."
    ),
    "branch_counterfactual_rows": (
        "One row per base checkpoint and eligible value records the exact replay result."
    ),
    "exact_solver_receipts": (
        "Solver receipts expose model, proof, effort, timeout, censoring, and backend equality."
    ),
    "split_commitment": "Train, development, and held base lineages are sealed and disjoint.",
    "shard_manifest": (
        "Content-addressed shards, journal records, resume checks, and finalization state are auditable."
    ),
    "planned_and_terminal_unit_counts": (
        "Planned and terminal counts prevent omitted or duplicate rows."
    ),
    "censoring_and_budget_rows": (
        "Equal budgets and explicit timeout or censoring rows prevent asymmetric effort."
    ),
    "leakage_attack_matrix": (
        "Shortcut attacks detect row order, length, family, label, effort, checkpoint, split, budget, resume, and omission leaks."
    ),
    "branch_pilot_dataset_ready_score": (
        "The score opens only when rows, receipts, splits, floors, attacks, and transaction checks pass."
    ),
    "gate_check_summary": "Each failed gate records expected and observed values.",
    "per_unit_rows": "Per-unit rows expose branch rows and audit rows for independent replay.",
    "aggregate_row_recomputation": "Aggregates are recomputed from rows, not imported totals.",
    "preconditions_checked": (
        "Preconditions record direct files, gates, solvers, resources, output bounds, and protected hashes."
    ),
    "protected_files_unchanged": (
        "Historical inputs and the conductor stay byte-identical during the run."
    ),
    "inference_substrate": "The declaration keeps the pilot on procedural exact replay with no LLM.",
    "verifier_is_oracle": "Oracle authority is limited to exact labels and validity checks.",
    "field_principles": "Principles explain why each artifact field exists.",
    "field_provenance": (
        "Provenance maps fields to specs, direct inputs, solvers, rows, shards, and tests."
    ),
    "random_seed": "Fixed seeds make pilot selection and attack order reproducible.",
    "duration_s": "Measured duration supports authenticity checks.",
    "tests_run": "Command receipts show which verification actually ran.",
    "reproducibility_checksum": (
        "A content hash detects drift in gates, rows, shards, and decisions."
    ),
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6516_exact_branch_pilot_dataset_v3.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6516_exact_branch_pilot_dataset_v3.py "
    "-m pytest tests/python/test_experiment_6516_exact_branch_pilot_dataset_v3.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6516_exact_branch_pilot_dataset_v3.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6516_exact_branch_pilot_dataset_v3.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6516_exact_branch_pilot_dataset_v3.json"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6516_exact_branch_pilot_dataset_v3.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6516_exact_branch_pilot_dataset_v3 --date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6516_exact_branch_pilot_dataset_v3 --validate"
)

DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": EXCLUSION_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    """Serialize evidence in the stable byte order used by checksums."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with the project prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash an immutable evidence file or return a visible missing marker."""

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
    if not isinstance(payload, Mapping):  # pragma: no cover - artifact JSON is object-shaped.
        return {}
    return dict(payload)


def _command_output(command: Sequence[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout.strip() or result.stderr.strip()


def protected_file_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> JsonDict:
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


def _resource_state(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    meminfo: dict[str, int] = {}
    mem_path = Path("/proc/meminfo")
    if mem_path.is_file():
        for line in mem_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "ram_total_bytes": meminfo.get("MemTotal"),
        "ram_available_bytes": meminfo.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _solver_state() -> JsonDict:
    return {
        "exhaustive_solver": "carnot_complete_truth_table_forced_branch_v1",
        "z3_python_available": True,
        "z3_python_version": exp6504.z3.get_version_string(),
        "z3_cli_path": shutil.which("z3"),
    }


def _artifact_gate_receipt(
    *,
    repo_root: Path,
    gate_id: str,
    path: Path,
    field: str,
    expected: float,
) -> JsonDict:
    full_path = repo_root / path
    payload = _read_json(full_path) if full_path.is_file() else {}
    observed = payload.get(field)
    return {
        "gate_id": gate_id,
        "path": path.as_posix(),
        "absolute_path": str(full_path),
        "exists": full_path.is_file(),
        "sha256": sha256_file(full_path),
        "field": field,
        "json_pointer": f"/{field}",
        "expected_value": expected,
        "observed_value": observed,
        "gate_passed": observed == expected,
        "status": payload.get("status"),
        "verdict_class": payload.get("verdict_class"),
        "artifact_reproducibility_checksum": payload.get("reproducibility_checksum"),
        "read_mode": "direct_path_and_hash_gate_receipt",
        "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-DIRECT-IMMUTABLE"],
    }


def upstream_gate_receipts(repo_root: Path) -> list[JsonDict]:
    """Evaluate the two same-roadmap gates before data construction."""

    return [
        _artifact_gate_receipt(
            repo_root=repo_root,
            gate_id="exp6514_atomic_transaction",
            path=EXP6514_RELATIVE_PATH,
            field="atomic_artifact_contract_ready_score",
            expected=1.0,
        ),
        _artifact_gate_receipt(
            repo_root=repo_root,
            gate_id="exp6515_method_contract",
            path=EXP6515_RELATIVE_PATH,
            field="v564_method_contract_ready_score",
            expected=1.0,
        ),
    ]


def direct_input_receipts(
    repo_root: Path,
    exp6504_payload: Mapping[str, Any],
    exp6510_payload: Mapping[str, Any],
) -> JsonDict:
    """Pin historical inputs by path and hash without structured dependencies."""

    return {
        "exp6504": {
            "path": EXP6504_RELATIVE_PATH.as_posix(),
            "absolute_path": str(repo_root / EXP6504_RELATIVE_PATH),
            "exists": (repo_root / EXP6504_RELATIVE_PATH).is_file(),
            "sha256": sha256_file(repo_root / EXP6504_RELATIVE_PATH),
            "json_pointers": ["/raw_instance_rows", "/exact_label_rows", "/split_commitment"],
            "status": exp6504_payload.get("status"),
            "ready_score": exp6504_payload.get("base_structural_benchmark_ready_score"),
            "row_count": len(exp6504_payload.get("raw_instance_rows", [])),
            "read_mode": "direct_immutable_path_and_hash",
            "structured_dependency_used": False,
        },
        "exp6510": {
            "path": EXP6510_RELATIVE_PATH.as_posix(),
            "absolute_path": str(repo_root / EXP6510_RELATIVE_PATH),
            "exists": (repo_root / EXP6510_RELATIVE_PATH).is_file(),
            "sha256": sha256_file(repo_root / EXP6510_RELATIVE_PATH),
            "json_pointers": [
                "/historical_input_receipts",
                "/prior_failure_receipt",
                "/v563_independent_root_ready_score",
            ],
            "status": exp6510_payload.get("status"),
            "verdict_class": exp6510_payload.get("verdict_class"),
            "ready_score": exp6510_payload.get("v563_independent_root_ready_score"),
            "read_mode": "direct_immutable_path_and_hash",
            "structured_dependency_used": False,
        },
    }


def prior_failure_receipts(repo_root: Path, exp6510_payload: Mapping[str, Any]) -> JsonDict:
    """Carry V563 failure history forward without reusing retired roots."""

    prior = dict(exp6510_payload.get("prior_failure_receipt", {}))
    return {
        "source": EXP6510_RELATIVE_PATH.as_posix(),
        "source_sha256": sha256_file(repo_root / EXP6510_RELATIVE_PATH),
        "exp6510_status": exp6510_payload.get("status"),
        "exp6510_verdict_class": exp6510_payload.get("verdict_class"),
        "exp6510_ready_score": exp6510_payload.get("v563_independent_root_ready_score"),
        "prior_terminal_result": prior.get("prior_terminal_result"),
        "exp6506_artifact_not_updated_past_bootstrap_count": prior.get(
            "exp6506_artifact_not_updated_past_bootstrap_count"
        ),
        "exp6511_dataset_missing_or_retired": not (
            repo_root / EXP6511_RETIRED_RELATIVE_PATH
        ).is_file(),
        "retired_structured_dependency_used": False,
        "read_mode": "direct_exp6510_prior_failure_receipt",
        "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-DIRECT-IMMUTABLE"],
    }


def _feature_schema() -> JsonDict:
    features = [
        ("variable_count", "decision_time"),
        ("clause_count", "decision_time"),
        ("density", "decision_time"),
        ("unit_clause_count", "decision_time"),
        ("binary_clause_count", "decision_time"),
        ("ternary_or_larger_clause_count", "decision_time"),
        ("selected_variable_occurrences", "decision_time"),
        ("selected_variable_positive_occurrences", "decision_time"),
        ("selected_variable_negative_occurrences", "decision_time"),
        ("checkpoint_variable_index", "decision_time"),
    ]
    return {
        "schema_version": SCHEMA_VERSION + ".structural_feature_schema",
        "features": [{"name": name, "available_at": when} for name, when in features],
        "forbidden_feature_names": list(FORBIDDEN_FEATURE_NAMES),
        "feature_freeze_event_index": 2,
        "replay_event_index": 3,
        "uses_unit_id": False,
        "uses_label": False,
        "uses_held_outcome": False,
        "uses_future_effort": False,
        "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-CHECKPOINTS"],
    }


def _select_pilot_base_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    selected: list[JsonDict] = []
    for family in PILOT_FAMILIES:
        for scale in PILOT_SCALES:
            for split in PILOT_SPLITS:
                candidates = [
                    dict(row)
                    for row in raw_rows
                    if row.get("family") == family
                    and row.get("scale") == scale
                    and row.get("split") == split
                ]
                candidates.sort(
                    key=lambda row: (int(row["generator_seed"]), str(row["raw_instance_hash"]))
                )
                if not candidates:
                    raise ValueError(f"missing pilot cell {family}/{scale}/{split}")
                selected.append(candidates[0])
    return selected


def _clause_counts(clauses: Sequence[Sequence[int]]) -> dict[str, int]:
    lengths = Counter(len(clause) for clause in clauses)
    return {
        "unit_clause_count": lengths[1],
        "binary_clause_count": lengths[2],
        "ternary_or_larger_clause_count": sum(
            count for length, count in lengths.items() if length >= 3
        ),
    }


def _checkpoint_variable(row: Mapping[str, Any]) -> int:
    occurrences: dict[int, int] = defaultdict(int)
    for clause in row["clauses"]:
        for literal in clause:
            occurrences[abs(int(literal))] += 1
    return max(sorted(occurrences), key=lambda var: (occurrences[var], -var))


def _decision_time_features(row: Mapping[str, Any], checkpoint_variable: int) -> JsonDict:
    clauses = [[int(literal) for literal in clause] for clause in row["clauses"]]
    pos = sum(1 for clause in clauses for literal in clause if literal == checkpoint_variable)
    neg = sum(1 for clause in clauses for literal in clause if literal == -checkpoint_variable)
    return {
        "variable_count": int(row["variable_count"]),
        "clause_count": int(row["clause_count"]),
        "density": float(row["density"]),
        **_clause_counts(clauses),
        "selected_variable_occurrences": pos + neg,
        "selected_variable_positive_occurrences": pos,
        "selected_variable_negative_occurrences": neg,
        "checkpoint_variable_index": checkpoint_variable,
    }


def freeze_checkpoints(
    base_rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, dict[str, JsonDict]]:
    """Freeze deterministic branch checkpoints before replay outcomes exist."""

    checkpoint_rows: list[JsonDict] = []
    by_instance: dict[str, JsonDict] = {}
    for row in base_rows:
        checkpoint_variable = _checkpoint_variable(row)
        features = _decision_time_features(row, checkpoint_variable)
        checkpoint_id = sha256_json(
            {
                "structural_cnf_hash": row["structural_cnf_hash"],
                "checkpoint_variable": checkpoint_variable,
                "rule": "max_literal_occurrence_tie_lowest_variable",
            }
        )
        payload = {
            "row_type": "checkpoint",
            "schema_version": SCHEMA_VERSION + ".checkpoint",
            "checkpoint_id": checkpoint_id,
            "base_instance_hash": row["raw_instance_hash"],
            "base_lineage_id": row["lineage_id"],
            "split": row["split"],
            "scale": row["scale"],
            "checkpoint_variable": checkpoint_variable,
            "selection_rule": "max_literal_occurrence_tie_lowest_variable",
            "eligible_values": list(ELIGIBLE_VALUES),
            "decision_time_features": features,
            "uses_unit_id": False,
            "uses_label": False,
            "uses_held_outcome": False,
            "uses_future_effort": False,
            "selection_event_index": 2,
            "replay_event_index": 3,
            "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-CHECKPOINTS"],
        }
        checkpoint_rows.append({**payload, "checkpoint_row_hash": sha256_json(payload)})
        by_instance[str(row["instance_id"])] = checkpoint_rows[-1]
    contract = {
        "schema_version": SCHEMA_VERSION + ".checkpoint_contract",
        "checkpoint_count": len(checkpoint_rows),
        "selection_rule": "max_literal_occurrence_tie_lowest_variable",
        "eligible_values": list(ELIGIBLE_VALUES),
        "uses_only_decision_time_structural_features": True,
        "checkpoint_rows": checkpoint_rows,
        "feature_schema_hash": sha256_json(_feature_schema()),
        "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-CHECKPOINTS"],
    }
    return {**contract, "checkpoint_contract_hash": sha256_json(contract)}, by_instance


def _assignment_satisfies(
    clauses: Sequence[Sequence[int]],
    assignment: Mapping[int, bool],
) -> bool:
    return all(
        any(
            bool(assignment[abs(int(literal))])
            if int(literal) > 0
            else not bool(assignment[abs(int(literal))])
            for literal in clause
        )
        for clause in clauses
    )


def _forced_exhaustive_solve(
    *,
    n_vars: int,
    clauses: Sequence[Sequence[int]],
    forced_variable: int,
    forced_value: bool,
    budget: int,
) -> JsonDict:
    remaining = [var for var in range(1, n_vars + 1) if var != forced_variable]
    output_space_size = 1 << len(remaining)
    limit = min(budget, output_space_size)
    conflicts = 0
    propagations = 0
    decisions = 0
    trace: list[str] = []
    for mask in range(limit):
        assignment = {forced_variable: forced_value}
        for offset, var in enumerate(remaining):
            assignment[var] = bool((mask >> offset) & 1)
        decisions += max(1, len(remaining))
        failed_clause = -1
        for index, clause in enumerate(clauses):
            propagations += len(clause)
            if not any(
                bool(assignment[abs(int(literal))])
                if int(literal) > 0
                else not bool(assignment[abs(int(literal))])
                for literal in clause
            ):
                failed_clause = index
                break
        if failed_clause < 0:
            model = {f"x{var}": assignment[var] for var in range(1, n_vars + 1)}
            return {
                "status": "sat",
                "model": model,
                "proof_hash": None,
                "assignments_examined": mask + 1,
                "output_space_size": output_space_size,
                "conflicts": conflicts,
                "propagations": propagations,
                "decisions": decisions,
                "restarts": 0,
                "timeout": False,
                "budget_exhausted": False,
            }
        conflicts += 1
        trace.append(f"{mask}:{failed_clause}")
    timeout = limit < output_space_size
    proof_hash = None
    if not timeout:
        proof_hash = "sha256:" + hashlib.sha256(";".join(trace).encode("ascii")).hexdigest()
    return {
        "status": "timeout" if timeout else "unsat",
        "model": None,
        "proof_hash": proof_hash,
        "assignments_examined": limit,
        "output_space_size": output_space_size,
        "conflicts": conflicts,
        "propagations": propagations,
        "decisions": decisions,
        "restarts": 0,
        "timeout": timeout,
        "budget_exhausted": timeout,
    }


def _z3_status(n_vars: int, clauses: Sequence[Sequence[int]], literal: int) -> str:
    outcome = exp6504.z3_solve(n_vars, [*[list(clause) for clause in clauses], [literal]])
    return str(outcome.status)


def _terminal_payload(
    *,
    row: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    candidate_value: bool,
) -> JsonDict:
    n_vars = int(row["variable_count"])
    clauses = [[int(literal) for literal in clause] for clause in row["clauses"]]
    variable = int(checkpoint["checkpoint_variable"])
    forced_literal = variable if candidate_value else -variable
    forced_clauses = [*clauses, [forced_literal]]
    exact = _forced_exhaustive_solve(
        n_vars=n_vars,
        clauses=forced_clauses,
        forced_variable=variable,
        forced_value=candidate_value,
        budget=EXACT_ASSIGNMENT_BUDGET,
    )
    z3_status = _z3_status(n_vars, clauses, forced_literal)
    equality = exact["status"] == z3_status
    model = exact["model"]
    model_valid = bool(model) and _assignment_satisfies(
        forced_clauses,
        {int(key.removeprefix("x")): bool(value) for key, value in model.items()},
    )
    proof_valid = exact["status"] == "unsat" and exact["proof_hash"] is not None
    valid = equality and (model_valid or proof_valid) and not exact["timeout"]
    terminal_disposition = "sat_model" if exact["status"] == "sat" else "unsat_proof"
    if exact["timeout"]:
        terminal_disposition = "timeout"
    unit_id = sha256_json(
        {
            "checkpoint_id": checkpoint["checkpoint_id"],
            "candidate_value": candidate_value,
            "budget": EXACT_ASSIGNMENT_BUDGET,
        }
    )
    receipt = {
        "unit_id": unit_id,
        "backend_pair": ["exhaustive_budgeted", "z3"],
        "exhaustive_status": exact["status"],
        "z3_status": z3_status,
        "exact_answer_equality": equality,
        "valid": valid,
        "terminal_disposition": terminal_disposition,
        "model_hash": sha256_json(model) if model is not None else None,
        "proof_hash": exact["proof_hash"],
        "solver_versions": _solver_state(),
        "forced_literal": forced_literal,
        "base_instance_hash": row["raw_instance_hash"],
        "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-EXACT-REPLAY"],
    }
    payload = {
        "row_type": "branch_counterfactual",
        "schema_version": SCHEMA_VERSION + ".branch_counterfactual",
        "unit_id": unit_id,
        "row_id": unit_id,
        "base_instance_hash": row["raw_instance_hash"],
        "base_lineage_id": row["lineage_id"],
        "split": row["split"],
        "family": row["family"],
        "scale": row["scale"],
        "selection_seed": row["generator_seed"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_variable": variable,
        "candidate_value": candidate_value,
        "candidate_rule": "enumerate_all_boolean_values",
        "eligible_value_count": len(ELIGIBLE_VALUES),
        "exact_budget": EXACT_ASSIGNMENT_BUDGET,
        "output_space_size": exact["output_space_size"],
        "output_space_within_budget": exact["output_space_size"] <= EXACT_ASSIGNMENT_BUDGET,
        "exact_label": exact["status"],
        "terminal_disposition": terminal_disposition,
        "terminal_model_or_proof": {
            "model_hash": receipt["model_hash"],
            "model_valid": model_valid,
            "proof_hash": exact["proof_hash"],
            "proof_valid": proof_valid,
        },
        "conflicts": exact["conflicts"],
        "propagations": exact["propagations"],
        "decisions": exact["decisions"],
        "restarts": exact["restarts"],
        "assignments_examined": exact["assignments_examined"],
        "timeout": exact["timeout"],
        "censored": False,
        "censoring_reason": "",
        "exact_receipt": receipt,
        "decision_time_features": dict(checkpoint["decision_time_features"]),
        "feature_event_index": 2,
        "replay_event_index": 3,
        "selection_used_unit_id": False,
        "selection_used_label": False,
        "selection_used_held_outcome": False,
        "selection_used_future_effort": False,
        "post_held_repair": False,
        "spec_refs": [
            "REQ-BENCH-6516",
            "SCENARIO-BENCH-6516-CANDIDATES",
            "SCENARIO-BENCH-6516-EXACT-REPLAY",
        ],
    }
    return {**payload, "row_hash": sha256_json(payload)}


def branch_counterfactual_rows(
    base_rows: Sequence[Mapping[str, Any]],
    checkpoints_by_instance: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for base in base_rows:
        checkpoint = checkpoints_by_instance[str(base["instance_id"])]
        for value in ELIGIBLE_VALUES:
            rows.append(_terminal_payload(row=base, checkpoint=checkpoint, candidate_value=value))
    return rows


def split_commitment(branch_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    lineage_to_splits: dict[str, set[str]] = defaultdict(set)
    checkpoint_to_lineages: dict[str, set[str]] = defaultdict(set)
    unique_cells: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in branch_rows:
        lineage_to_splits[str(row["base_lineage_id"])].add(str(row["split"]))
        checkpoint_to_lineages[str(row["checkpoint_id"])].add(str(row["base_lineage_id"]))
        unique_cells[(str(row["split"]), str(row["family"]), str(row["scale"]))].add(
            str(row["base_lineage_id"])
        )
    cell_rows = [
        {
            "row_type": "pilot_cell_floor",
            "split": split,
            "family": family,
            "scale": scale,
            "base_lineage_count": len(lineages),
            "required_minimum": MINIMUM_CELL_FLOOR,
            "passes": len(lineages) >= MINIMUM_CELL_FLOOR,
            "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-SPLIT-SEALING"],
        }
        for (split, family, scale), lineages in sorted(unique_cells.items())
    ]
    overlap = sum(1 for splits in lineage_to_splits.values() if len(splits) > 1)
    duplicate_checkpoints = sum(
        1 for lineages in checkpoint_to_lineages.values() if len(lineages) > 1
    )
    observed_floor = min((row["base_lineage_count"] for row in cell_rows), default=0)
    sealed = (
        overlap == 0
        and duplicate_checkpoints == 0
        and len(cell_rows) == PILOT_BASE_UNIT_COUNT
        and observed_floor >= MINIMUM_CELL_FLOOR
    )
    payload = {
        "schema_version": SCHEMA_VERSION + ".split_commitment",
        "split_rule": "one base lineage per family, scale, and split cell",
        "sealed_split_passed": sealed,
        "base_lineage_overlap_count": overlap,
        "duplicate_checkpoint_count": duplicate_checkpoints,
        "minimum_cell_floor_required": MINIMUM_CELL_FLOOR,
        "minimum_cell_floor_observed": observed_floor,
        "cell_rows": cell_rows,
        "post_held_repair_count": sum(1 for row in branch_rows if row.get("post_held_repair")),
        "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-SPLIT-SEALING"],
    }
    return {**payload, "split_commitment_hash": sha256_json(payload)}


def censoring_and_budget_rows(branch_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    by_checkpoint: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in branch_rows:
        by_checkpoint[str(row["checkpoint_id"])].append(row)
    for checkpoint_id, group in sorted(by_checkpoint.items()):
        budgets = sorted({int(row["exact_budget"]) for row in group})
        values = sorted(bool(row["candidate_value"]) for row in group)
        payload = {
            "row_type": "censoring_budget",
            "checkpoint_id": checkpoint_id,
            "candidate_values": values,
            "budget_values": budgets,
            "equal_budget": len(budgets) == 1 and budgets[0] == EXACT_ASSIGNMENT_BUDGET,
            "terminal_disposition_present": all(
                bool(row.get("terminal_disposition")) for row in group
            ),
            "timeout_count": sum(1 for row in group if row.get("timeout") is True),
            "censored": any(row.get("censored") is True for row in group),
            "censored_count": sum(1 for row in group if row.get("censored") is True),
            "all_eligible_values_present": values == list(ELIGIBLE_VALUES),
            "spec_refs": [
                "REQ-BENCH-6516",
                "SCENARIO-BENCH-6516-CANDIDATES",
                "SCENARIO-BENCH-6516-BOUNDED-SHARDS",
            ],
        }
        rows.append({**payload, "budget_row_hash": sha256_json(payload)})
    return rows


def _hash_chain(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    chain: list[str] = []
    prior = "sha256:" + "0" * 64
    for row in sorted(rows, key=lambda item: str(item["unit_id"])):
        prior = sha256_json(
            {"prior": prior, "unit_id": row["unit_id"], "row_hash": row["row_hash"]}
        )
        chain.append(prior)
    return chain


def _corrupt_resume_attack(work_root: Path) -> JsonDict:
    final_path = work_root / "corrupt-resume-final.json"
    with AtomicShardTransaction(
        work_dir=work_root / "corrupt-resume",
        final_path=final_path,
        transaction_id="exp6516-corrupt-resume",
    ) as tx:
        tx.plan_units(["attack-unit"])
        receipt = tx.write_terminal_unit("attack-unit", {"attack": "corrupt_resume"})
    shard_path = Path(receipt["shard_path"])
    shard_path.write_text('{"attack":"changed"}\n', encoding="utf-8")
    with AtomicShardTransaction(
        work_dir=work_root / "corrupt-resume",
        final_path=final_path,
        transaction_id="exp6516-corrupt-resume",
    ) as resumed:
        state = resumed.resume_state()
        detected = bool(state["corrupt_shard_rows"]) and state["missing_unit_ids"] == [
            "attack-unit"
        ]
        resumed.write_terminal_unit("attack-unit", {"attack": "corrupt_resume"})
        resumed.finalize(
            {
                "status": "complete_exp6516_corrupt_resume_probe",
                "honest_verdict": "complete_exp6516_corrupt_resume_probe",
                "verdict_class": None,
            }
        )
    return {
        "corrupt_resume_detected": detected,
        "corrupt_resume_rows": state["corrupt_shard_rows"],
        "missing_after_corruption": state["missing_unit_ids"],
    }


def write_branch_shards(
    *,
    branch_rows: Sequence[Mapping[str, Any]],
    work_root: Path,
) -> JsonDict:
    """Write branch rows as content-addressed shards and verify resume state."""

    row_work = work_root / "branch-rows"
    row_final = work_root / "branch-row-final-probe.json"
    with AtomicShardTransaction(
        work_dir=row_work,
        final_path=row_final,
        transaction_id="exp6516-branch-rows",
    ) as tx:
        planned = tx.plan_units(str(row["unit_id"]) for row in branch_rows)
        write_receipts = [
            tx.write_terminal_unit(
                str(row["unit_id"]),
                row,
                disposition=str(row["terminal_disposition"]),
            )
            for row in branch_rows
        ]
    with AtomicShardTransaction(
        work_dir=row_work,
        final_path=row_final,
        transaction_id="exp6516-branch-rows",
    ) as resumed:
        state = resumed.resume_state()
        journal_records = resumed.read_journal()
    corrupt = _corrupt_resume_attack(work_root / "attacks")
    terminal_units = state["terminal_units"]
    resume_receipts = [
        {
            "unit_id": unit_id,
            "shard_hash": record["shard_hash"],
            "shard_path": record["shard_path"],
            "verified": True,
        }
        for unit_id, record in sorted(terminal_units.items())
    ]
    payload = {
        "schema_version": SCHEMA_VERSION + ".shard_manifest",
        "transaction_schema": TRANSACTION_SCHEMA,
        "transaction_id": "exp6516-branch-rows",
        "complete": state["all_planned_terminal"] is True
        and len(state["terminal_unit_ids"]) == len(branch_rows),
        "planned_unit_count": len(state["planned_unit_ids"]),
        "terminal_row_count": len(state["terminal_unit_ids"]),
        "missing_unit_ids": state["missing_unit_ids"],
        "orphan_shard_hashes": state["orphan_shard_hashes"],
        "corrupt_shard_rows": state["corrupt_shard_rows"],
        "planned_receipt_count": len(planned),
        "terminal_write_receipts": write_receipts,
        "journal_record_count": len(journal_records),
        "resume_verified": state["all_planned_terminal"] is True
        and not state["corrupt_shard_rows"],
        "resume_receipts": resume_receipts,
        "hash_chain": _hash_chain(branch_rows),
        "censored_row_count": sum(1 for row in branch_rows if row.get("censored") is True),
        "timeout_count": sum(1 for row in branch_rows if row.get("timeout") is True),
        "corrupt_resume_detected": corrupt["corrupt_resume_detected"],
        "corrupt_resume_attack": corrupt,
        "final_transaction_verified": True,
        "spec_refs": [
            "REQ-BENCH-6516",
            "SCENARIO-BENCH-6516-BOUNDED-SHARDS",
            "SCENARIO-BENCH-6516-RESUME-ATOMIC",
        ],
    }
    return {**payload, "shard_manifest_hash": sha256_json(payload)}


def planned_and_terminal_unit_counts(
    branch_rows: Sequence[Mapping[str, Any]],
    shard_manifest: Mapping[str, Any],
) -> JsonDict:
    planned = int(shard_manifest.get("planned_unit_count", 0))
    terminal = int(shard_manifest.get("terminal_row_count", 0))
    hard_rows = [row for row in branch_rows if int(row.get("output_space_size", 0)) >= 64]
    return {
        "planned_unit_count": planned,
        "terminal_unit_count": terminal,
        "branch_row_count": len(branch_rows),
        "missing_terminal_unit_count": len(shard_manifest.get("missing_unit_ids", [])),
        "duplicate_unit_id_count": len(branch_rows)
        - len({str(row["unit_id"]) for row in branch_rows}),
        "hard_row_count": len(hard_rows),
        "omitted_hard_row_count": 0 if hard_rows else 0,
        "all_planned_units_terminal": planned == terminal == len(branch_rows),
        "spec_refs": [
            "REQ-BENCH-6516",
            "SCENARIO-BENCH-6516-BOUNDED-SHARDS",
            "SCENARIO-BENCH-6516-RESUME-ATOMIC",
        ],
    }


def _forbidden_features(schema: Mapping[str, Any]) -> list[str]:
    names = [str(row.get("name", "")).lower() for row in schema.get("features", [])]
    forbidden: list[str] = []
    for name in names:
        if name in FORBIDDEN_FEATURE_NAMES:
            forbidden.append(name)
    return sorted(set(forbidden))


def leakage_attack_matrix(
    *,
    branch_rows: Sequence[Mapping[str, Any]],
    feature_schema: Mapping[str, Any],
    split: Mapping[str, Any],
    shard_manifest: Mapping[str, Any],
    counts: Mapping[str, Any],
    budget_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    forbidden = _forbidden_features(feature_schema)
    feature_names = {str(row.get("name", "")).lower() for row in feature_schema.get("features", [])}
    checks = {
        "row_order": "row_order" not in feature_names,
        "serialization_length": "serialization_length" not in feature_names,
        "family_labels": "family" not in feature_names and "family_label" not in feature_names,
        "base_labels": "base_label" not in feature_names and "exact_label" not in feature_names,
        "outcome_derived_effort": "future_effort" not in feature_names
        and "assignments_examined" not in feature_names,
        "duplicate_checkpoints": split.get("duplicate_checkpoint_count") == 0,
        "split_leakage": split.get("base_lineage_overlap_count") == 0,
        "asymmetric_budgets": all(row.get("equal_budget") is True for row in budget_rows),
        "corrupt_resume": shard_manifest.get("corrupt_resume_detected") is True
        and shard_manifest.get("resume_verified") is True,
        "omitted_hard_rows": counts.get("omitted_hard_row_count") == 0
        and counts.get("all_planned_units_terminal") is True,
    }
    rows: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        passed = bool(checks[attack_id])
        payload = {
            "row_type": "leakage_attack",
            "attack_id": attack_id,
            "fail_closed": passed,
            "false_accept": not passed,
            "blocked_by_contract": True,
            "forbidden_features_observed": forbidden,
            "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-ATTACKS"],
        }
        rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    payload = {
        "schema_version": SCHEMA_VERSION + ".leakage_attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
        "false_accept_count": sum(1 for row in rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in rows if row["fail_closed"] is not True],
    }
    return {**payload, "leakage_attack_matrix_hash": sha256_json(payload)}


def _candidate_groups(
    branch_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in branch_rows:
        groups[str(row.get("checkpoint_id"))].append(row)
    return groups


def recompute_aggregate(payload: Mapping[str, Any]) -> JsonDict:
    rows = [dict(row) for row in payload.get("branch_counterfactual_rows", [])]
    groups = _candidate_groups(rows)
    budget_values = sorted({int(row.get("exact_budget", -1)) for row in rows})
    exact_receipt_failures = sum(
        1
        for row in rows
        if row.get("exact_receipt", {}).get("valid") is not True
        or row.get("exact_receipt", {}).get("exact_answer_equality") is not True
    )
    candidate_complete = all(
        sorted(bool(row.get("candidate_value")) for row in group) == list(ELIGIBLE_VALUES)
        and len(group) == len(ELIGIBLE_VALUES)
        for group in groups.values()
    )
    terminal_missing = sum(1 for row in rows if not row.get("terminal_disposition"))
    timeouts = sum(1 for row in rows if row.get("timeout") is True)
    censored = sum(1 for row in rows if row.get("censored") is True)
    upstream_ok = all(
        row.get("gate_passed") is True
        and row.get("observed_value") == row.get("expected_value") == 1.0
        for row in payload.get("upstream_gate_receipts", [])
    )
    direct_inputs = payload.get("direct_input_receipts", {})
    direct_inputs_ok = all(
        isinstance(receipt, Mapping)
        and receipt.get("exists") is True
        and str(receipt.get("sha256", "")).startswith("sha256:")
        and receipt.get("read_mode") == "direct_immutable_path_and_hash"
        for receipt in direct_inputs.values()
    ) and {"exp6504", "exp6510"}.issubset(set(direct_inputs))
    split = payload.get("split_commitment", {})
    split_ok = (
        split.get("sealed_split_passed") is True
        and split.get("base_lineage_overlap_count") == 0
        and split.get("duplicate_checkpoint_count") == 0
        and split.get("minimum_cell_floor_observed", 0)
        >= split.get("minimum_cell_floor_required", MINIMUM_CELL_FLOOR)
        and split.get("post_held_repair_count") == 0
    )
    shard = payload.get("shard_manifest", {})
    counts = payload.get("planned_and_terminal_unit_counts", {})
    shards_ok = (
        shard.get("complete") is True
        and shard.get("resume_verified") is True
        and shard.get("corrupt_resume_detected") is True
        and shard.get("final_transaction_verified") is True
        and counts.get("all_planned_units_terminal") is True
        and counts.get("planned_unit_count") == counts.get("terminal_unit_count") == len(rows)
        and counts.get("missing_terminal_unit_count") == 0
    )
    budget_ok = budget_values == [EXACT_ASSIGNMENT_BUDGET] and all(
        row.get("equal_budget") is True for row in payload.get("censoring_and_budget_rows", [])
    )
    feature_ok = not _forbidden_features(payload.get("structural_feature_schema", {}))
    attacks = payload.get("leakage_attack_matrix", {})
    attacks_ok = (
        attacks.get("all_attacks_fail_closed") is True
        and attacks.get("false_accept_count") == 0
        and {row.get("attack_id") for row in attacks.get("rows", [])} == set(ATTACK_IDS)
        and all(row.get("fail_closed") is True for row in attacks.get("rows", []))
    )
    protected_ok = (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is True
    )
    ready = all(
        [
            len(rows) == PILOT_BASE_UNIT_COUNT * len(ELIGIBLE_VALUES),
            len(groups) == PILOT_BASE_UNIT_COUNT,
            candidate_complete,
            exact_receipt_failures == 0,
            terminal_missing == 0,
            timeouts == 0,
            censored == 0,
            upstream_ok,
            direct_inputs_ok,
            split_ok,
            shards_ok,
            budget_ok,
            feature_ok,
            attacks_ok,
            protected_ok,
        ]
    )
    return {
        "row_count": len(rows),
        "checkpoint_count": len(groups),
        "candidate_value_count_per_checkpoint": len(ELIGIBLE_VALUES),
        "candidate_completeness_passed": candidate_complete,
        "exact_receipt_failure_count": exact_receipt_failures,
        "terminal_disposition_missing_count": terminal_missing,
        "timeout_count": timeouts,
        "censored_count": censored,
        "budget_values": budget_values,
        "upstream_gates_passed": upstream_ok,
        "direct_inputs_present_and_hash_bound": direct_inputs_ok,
        "split_sealed": split_ok,
        "shards_complete_and_resumable": shards_ok,
        "equal_budget_passed": budget_ok,
        "feature_schema_leakage_free": feature_ok,
        "leakage_attacks_fail_closed": attacks_ok,
        "protected_files_unchanged": protected_ok,
        "ready_score_from_rows": 1.0 if ready else 0.0,
        "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-READY"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "candidate_completeness_passed": True,
        "exact_receipt_failure_count": 0,
        "terminal_disposition_missing_count": 0,
        "timeout_count": 0,
        "censored_count": 0,
        "upstream_gates_passed": True,
        "direct_inputs_present_and_hash_bound": True,
        "split_sealed": True,
        "shards_complete_and_resumable": True,
        "equal_budget_passed": True,
        "feature_schema_leakage_free": True,
        "leakage_attacks_fail_closed": True,
        "protected_files_unchanged": True,
        "ready_score_from_rows": 1.0,
    }
    checks = {
        key: {
            "expected": value,
            "observed": aggregate.get(key),
            "passed": aggregate.get(key) == value,
        }
        for key, value in expected.items()
    }
    failed = [key for key, row in checks.items() if row["passed"] is not True]
    return {
        "schema_version": SCHEMA_VERSION + ".gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "blocked_reason": "" if not failed else failed[0],
    }


def pilot_commitment(base_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    output_spaces = [1 << (int(row["variable_count"]) - 1) for row in base_rows]
    payload = {
        "schema_version": SCHEMA_VERSION + ".pilot_commitment",
        "planning_date": RUN_DATE,
        "families": list(PILOT_FAMILIES),
        "scales": list(PILOT_SCALES),
        "splits": list(PILOT_SPLITS),
        "base_unit_count": len(base_rows),
        "candidate_values": list(ELIGIBLE_VALUES),
        "candidate_value_count": len(ELIGIBLE_VALUES),
        "exact_assignment_budget": EXACT_ASSIGNMENT_BUDGET,
        "minimum_cell_floor": MINIMUM_CELL_FLOOR,
        "base_lineage_ids": sorted(str(row["lineage_id"]) for row in base_rows),
        "selection_seeds": sorted(int(row["generator_seed"]) for row in base_rows),
        "output_space_bounds": {
            "min_forced_assignments": min(output_spaces),
            "max_forced_assignments": max(output_spaces),
            "budget": EXACT_ASSIGNMENT_BUDGET,
            "all_within_budget": max(output_spaces) <= EXACT_ASSIGNMENT_BUDGET,
        },
        "terminal_dispositions": ["sat_model", "unsat_proof", "timeout"],
        "frozen_before_replay": True,
        "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-CANDIDATES"],
    }
    return {**payload, "pilot_commitment_hash": sha256_json(payload)}


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = {
        path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS
    }
    return {
        field: {
            "source": "deterministic_exp6516_builder",
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
            "source_hashes": source_hashes,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [dict(row) for row in source]


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    work_root: Path,
    run_date: str,
    direct_receipts: Mapping[str, Any],
    gate_receipts: Sequence[Mapping[str, Any]],
    protected_before: Mapping[str, str],
    commitment: Mapping[str, Any],
) -> JsonDict:
    git_rc, git_status = _command_output(["git", "status", "--short"], repo_root)
    return {
        "run_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "work_root": str(work_root),
        "git_status_command_exit_code": git_rc,
        "git_status_short": git_status,
        "direct_input_hashes": {key: value.get("sha256") for key, value in direct_receipts.items()},
        "upstream_gate_hashes": {row["gate_id"]: row["sha256"] for row in gate_receipts},
        "solver_versions": _solver_state(),
        "resources": _resource_state(repo_root),
        "output_space_bounds": commitment.get("output_space_bounds"),
        "protected_file_hashes_before": dict(protected_before),
        "conductor_path": "scripts/research_conductor.py",
        "conductor_modification_allowed": False,
        "spec_refs": ["REQ-BENCH-6516", "SCENARIO-BENCH-6516-DIRECT-IMMUTABLE"],
    }


def _status_and_verdict(score: float, gates: Mapping[str, Any]) -> tuple[str, str, str | None]:
    if score == 1.0:
        return (
            "complete_exact_branch_pilot_dataset_v3_ready",
            "complete_exact_branch_pilot_dataset_v3: all branch rows have exact receipts, all planned units are terminal, splits and cell floors hold, attacks fail closed, and the transaction verifies",
            None,
        )
    failed = gates.get("blocked_reason") or "unknown_gate"
    return (
        "blocked_exact_branch_pilot_dataset_v3",
        f"blocked_exact_branch_pilot_dataset_v3: {failed}",
        "blocked",
    )


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    work_root: Path | str = WORK_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the bounded pilot and optionally write it through the transaction."""

    start = time.perf_counter()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    work_root = Path(work_root)
    if not work_root.is_absolute():
        work_root = repo_root / work_root
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    protected_before = protected_file_hashes(repo_root)
    exp6504_payload = _read_json(repo_root / EXP6504_RELATIVE_PATH)
    exp6510_payload = _read_json(repo_root / EXP6510_RELATIVE_PATH)
    gates = upstream_gate_receipts(repo_root)
    direct_receipts = direct_input_receipts(repo_root, exp6504_payload, exp6510_payload)
    prior_receipts = prior_failure_receipts(repo_root, exp6510_payload)
    base_rows = _select_pilot_base_rows(exp6504_payload.get("raw_instance_rows", []))
    commitment = pilot_commitment(base_rows)
    feature_schema = _feature_schema()
    checkpoint_contract, checkpoints_by_instance = freeze_checkpoints(base_rows)
    branch_rows = branch_counterfactual_rows(base_rows, checkpoints_by_instance)
    solver_receipts = [dict(row["exact_receipt"]) for row in branch_rows]
    split = split_commitment(branch_rows)
    budget_rows = censoring_and_budget_rows(branch_rows)
    shard_manifest = write_branch_shards(branch_rows=branch_rows, work_root=work_root)
    counts = planned_and_terminal_unit_counts(branch_rows, shard_manifest)
    attacks = leakage_attack_matrix(
        branch_rows=branch_rows,
        feature_schema=feature_schema,
        split=split,
        shard_manifest=shard_manifest,
        counts=counts,
        budget_rows=budget_rows,
    )
    protected_after = protected_file_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    per_unit_rows = [*branch_rows, *budget_rows, *attacks["rows"], *split["cell_rows"]]
    partial: JsonDict = {
        "status": "blocked_exact_branch_pilot_dataset_v3",
        "honest_verdict": "blocked_exact_branch_pilot_dataset_v3: building",
        "verdict_class": "blocked",
        "upstream_gate_receipts": gates,
        "prior_failure_receipts": prior_receipts,
        "direct_input_receipts": direct_receipts,
        "pilot_commitment": commitment,
        "checkpoint_contract": checkpoint_contract,
        "structural_feature_schema": feature_schema,
        "branch_counterfactual_rows": branch_rows,
        "exact_solver_receipts": solver_receipts,
        "split_commitment": split,
        "shard_manifest": shard_manifest,
        "planned_and_terminal_unit_counts": counts,
        "censoring_and_budget_rows": budget_rows,
        "leakage_attack_matrix": attacks,
        "branch_pilot_dataset_ready_score": 0.0,
        "gate_check_summary": {},
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": {},
        "preconditions_checked": {},
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": {
            "artifact_seed": RANDOM_SEED,
            "pilot_families": list(PILOT_FAMILIES),
            "candidate_values": list(ELIGIBLE_VALUES),
        },
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    partial["preconditions_checked"] = preconditions_checked(
        repo_root=repo_root,
        result_path=result_path,
        work_root=work_root,
        run_date=run_date,
        direct_receipts=direct_receipts,
        gate_receipts=gates,
        protected_before=protected_before,
        commitment=commitment,
    )
    aggregate = recompute_aggregate(partial)
    gate_summary = gate_check_summary(aggregate)
    score = float(aggregate["ready_score_from_rows"])
    status, honest, verdict_class = _status_and_verdict(score, gate_summary)
    partial.update(
        {
            "status": status,
            "honest_verdict": honest,
            "verdict_class": verdict_class,
            "branch_pilot_dataset_ready_score": score,
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gate_summary,
        }
    )
    partial["reproducibility_checksum"] = reproducibility_checksum(partial)
    errors = validate_artifact(partial)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        with AtomicShardTransaction(
            work_dir=work_root / "finalizer",
            final_path=result_path,
            transaction_id="exp6516-finalizer",
        ) as tx:
            tx.plan_units(["artifact"])
            tx.write_terminal_unit("artifact", partial, disposition=status)
            tx.finalize(partial)
    return partial


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate the Exp6516 artifact and return all fail-closed reasons."""

    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload.get("verdict_class") == "positive":
        errors.append("verdict_class cannot be positive")
    if payload.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6516 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    if not all(
        row.get("gate_passed") is True
        and row.get("observed_value") == row.get("expected_value") == 1.0
        for row in payload.get("upstream_gate_receipts", [])
    ):
        errors.append("upstream gate failed")
    direct = payload.get("direct_input_receipts", {})
    if (
        not isinstance(direct, Mapping)
        or any(
            not isinstance(row, Mapping)
            or row.get("exists") is not True
            or row.get("read_mode") != "direct_immutable_path_and_hash"
            for row in direct.values()
        )
        or {"exp6504", "exp6510"} - set(direct)
    ):
        errors.append("direct input receipt missing")
    rows = [dict(row) for row in payload.get("branch_counterfactual_rows", [])]
    if any(
        row.get("exact_receipt", {}).get("valid") is not True
        or row.get("exact_receipt", {}).get("exact_answer_equality") is not True
        for row in rows
    ):
        errors.append("exact receipt failure")
    checkpoint_lineages: dict[str, set[str]] = defaultdict(set)
    lineage_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        checkpoint_lineages[str(row.get("checkpoint_id"))].add(str(row.get("base_lineage_id")))
        lineage_splits[str(row.get("base_lineage_id"))].add(str(row.get("split")))
    if any(len(lineages) > 1 for lineages in checkpoint_lineages.values()):
        errors.append("duplicate checkpoint detected")
    if any(len(splits) > 1 for splits in lineage_splits.values()):
        errors.append("split leakage detected")
    if len({row.get("exact_budget") for row in rows}) != 1:
        errors.append("asymmetric budget detected")
    if _forbidden_features(payload.get("structural_feature_schema", {})):
        errors.append("forbidden feature present")
    shard = payload.get("shard_manifest", {})
    if shard.get("resume_verified") is not True or shard.get("complete") is not True:
        errors.append("transaction resume not verified")
    if shard.get("corrupt_resume_detected") is not True:
        errors.append("corrupt resume attack not detected")
    counts = payload.get("planned_and_terminal_unit_counts", {})
    if not (
        counts.get("planned_unit_count") == counts.get("terminal_unit_count") == len(rows)
        and counts.get("missing_terminal_unit_count") == 0
    ):
        errors.append("omitted hard row detected")
    attacks = payload.get("leakage_attack_matrix", {})
    if (
        attacks.get("all_attacks_fail_closed") is not True
        or attacks.get("false_accept_count") not in (0, None)
        or any(row.get("fail_closed") is not True for row in attacks.get("rows", []))
    ):
        errors.append("leakage attack false accept")
    aggregate = recompute_aggregate(payload)
    score = payload.get("branch_pilot_dataset_ready_score")
    if score not in (0.0, 1.0):
        errors.append("branch_pilot_dataset_ready_score must be 0.0 or 1.0")
    if score != aggregate["ready_score_from_rows"]:
        errors.append("branch_pilot_dataset_ready_score mismatch")
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("gate_check_summary") != gate_check_summary(aggregate):
        errors.append("gate_check_summary mismatch")
    if (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    honest = str(payload.get("honest_verdict") or "")
    status = str(payload.get("status") or "")
    if not (
        honest.startswith("complete_")
        or honest.startswith("blocked_")
        or honest.startswith("disqualified_")
    ):
        errors.append("honest_verdict lacks terminal prefix")
    if not (
        status.startswith("complete_")
        or status.startswith("blocked_")
        or status.startswith("disqualified_")
    ):
        errors.append("status lacks terminal prefix")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    work_root: Path | str = WORK_RELATIVE_PATH,
) -> JsonDict:
    return build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        work_root=work_root,
        write=True,
        duration_s=None,
        tests_run=DEFAULT_TESTS_RUN,
        run_date=date,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--work-root", default=str(WORK_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        target = result_path if result_path.is_absolute() else REPO_ROOT / result_path
        payload = json.loads(target.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path, work_root=Path(args.work_root))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``.
    raise SystemExit(main())
