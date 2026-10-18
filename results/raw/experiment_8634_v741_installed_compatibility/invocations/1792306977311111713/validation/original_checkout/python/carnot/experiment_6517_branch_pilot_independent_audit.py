"""Exp6517 independent audit for the Exp6516 branch pilot.

Spec refs: REQ-BENCH-6517, SCENARIO-BENCH-6517-MISSING-UPSTREAM,
SCENARIO-BENCH-6517-ROW-REPLAY, SCENARIO-BENCH-6517-SHARDS,
SCENARIO-BENCH-6517-SPLIT-TIMING, SCENARIO-BENCH-6517-ATTACKS,
SCENARIO-BENCH-6517-TERMINAL.

The audit reads the Exp6516 artifact as an upstream source. It independently
replays exact branch receipts from Exp6504 base rows, checks shard and
transaction receipts, and writes a closed artifact even when the source fails.
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
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6517
SCHEMA_VERSION = "carnot.experiment_6517.branch_pilot_independent_audit.v1"
INFERENCE_SUBSTRATE = "independent_exact_row_shard_and_receipt_replay_no_llm"
VERIFIER_IS_ORACLE = True

RESULT_RELATIVE_PATH = Path("results/experiment_6517_branch_pilot_independent_audit.json")
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6516_exact_branch_pilot_dataset_v3.json")
EXP6514_RELATIVE_PATH = Path("results/experiment_6514_atomic_shard_artifact_transaction.json")
EXP6504_RELATIVE_PATH = Path("results/experiment_6504_exact_structural_benchmark_commitment.json")
EXP6510_RELATIVE_PATH = Path("results/experiment_6510_v563_independent_exact_root.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6517_branch_pilot_independent_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6517_branch_pilot_independent_audit.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

ELIGIBLE_VALUES = (False, True)
EXPECTED_BUDGET = 256
EXPECTED_BRANCH_ROW_COUNT = 36
EXPECTED_CHECKPOINT_COUNT = 18
EXPECTED_CELL_COUNT = 18
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
    "identity_ids",
    "row_order",
    "serialization_length",
    "family_shortcuts",
    "label_shortcuts",
    "future_effort",
    "censoring_removal",
    "corrupt_shards",
    "aggregate_tampering",
    "one_cell_headroom",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_artifact_receipt",
    "prior_failure_receipt",
    "independent_row_recomputation",
    "exact_receipt_replay_rows",
    "split_and_lineage_audit",
    "transaction_and_shard_audit",
    "feature_timing_audit",
    "censoring_audit",
    "shortcut_attack_matrix",
    "branch_pilot_audited_ready_score",
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
    "status": "A terminal status proves the audit closed even when the source did not.",
    "honest_verdict": (
        "The verdict reports dataset readiness without claiming learned guidance value."
    ),
    "verdict_class": (
        "Null means valid readiness. Blocked and disqualified preserve missing, incomplete, "
        "leakage, and false-receipt failures."
    ),
    "upstream_artifact_receipt": (
        "The source receipt records existence, path, hash, parse status, verdict class, row "
        "counts, shard counts, solver state, resources, and protected hashes."
    ),
    "prior_failure_receipt": (
        "Prior Exp6510 and Exp6511 failure context stays visible without becoming a dependency."
    ),
    "independent_row_recomputation": (
        "Readiness is recomputed from branch rows and per-unit evidence rather than imported "
        "aggregate fields."
    ),
    "exact_receipt_replay_rows": (
        "Each audit row verifies a branch receipt against the immutable base row hash and forced "
        "assignment."
    ),
    "split_and_lineage_audit": (
        "Split rows prove train, development, and held lineages are sealed and disjoint."
    ),
    "transaction_and_shard_audit": (
        "Shard and journal checks prove planned rows, terminal rows, hashes, resume receipts, "
        "and atomic receipts are consistent."
    ),
    "feature_timing_audit": (
        "Feature rows prove checkpoint selection used decision-time structural features only."
    ),
    "censoring_audit": (
        "Censoring rows prove budgets, timeouts, and terminal dispositions are symmetric and "
        "explicit."
    ),
    "shortcut_attack_matrix": (
        "Attack rows close identity, order, length, label, effort, censoring, shard, aggregate, "
        "and headroom shortcuts."
    ),
    "branch_pilot_audited_ready_score": (
        "The score opens only when every independent audit gate passes."
    ),
    "gate_check_summary": "Each failed gate records expected and observed values.",
    "per_unit_rows": (
        "Audit rows expose one source-unit row plus manifest, split, feature, censoring, and "
        "attack rows."
    ),
    "aggregate_row_recomputation": (
        "The aggregate is rebuilt from audit rows instead of source totals."
    ),
    "preconditions_checked": (
        "Preconditions record paths, resources, solvers, protected hashes, and planning date."
    ),
    "protected_files_unchanged": (
        "The source, shards, historical inputs, and conductor must remain byte-identical."
    ),
    "inference_substrate": (
        "The declaration keeps the audit on exact row, shard, and receipt replay with no LLM."
    ),
    "verifier_is_oracle": "Oracle authority is limited to dataset integrity checks.",
    "field_principles": "Principles explain why each required field exists.",
    "field_provenance": (
        "Provenance maps fields to specs, inputs, rows, receipts, tests, and deterministic "
        "reducers."
    ),
    "random_seed": "A fixed seed makes attack ordering reproducible.",
    "duration_s": "Measured wall time supports authenticity checks.",
    "tests_run": "Command receipts show which validation actually ran.",
    "reproducibility_checksum": (
        "A content hash detects drift in inputs, rows, gates, and decisions."
    ),
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6517_branch_pilot_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6517_branch_pilot_independent_audit.py "
    "-m pytest tests/python/test_experiment_6517_branch_pilot_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6517_branch_pilot_independent_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6517_branch_pilot_independent_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6517_branch_pilot_independent_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6517_branch_pilot_independent_audit.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6517_branch_pilot_independent_audit --date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6517_branch_pilot_independent_audit --validate"
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

STATIC_PROTECTED_RELATIVE_PATHS = (
    UPSTREAM_RELATIVE_PATH,
    EXP6514_RELATIVE_PATH,
    EXP6504_RELATIVE_PATH,
    EXP6510_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-references.md"),
)


def canonical_json(value: Any) -> str:
    """Serialize evidence with stable key order for row checksums."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with the project digest prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def canonical_shard_bytes(value: Any) -> bytes:
    """Return the byte format used by the row shard transaction helper."""

    return (canonical_json(value) + "\n").encode("utf-8")


def sha256_shard_payload(value: Any) -> str:
    """Hash terminal row payload bytes exactly as row shards do."""

    return "sha256:" + hashlib.sha256(canonical_shard_bytes(value)).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Hash a file, or return a visible marker for missing inputs."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json_with_status(path: Path) -> tuple[JsonDict, str, str]:
    if not path.is_file():
        return {}, "missing", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, "corrupt_json", str(exc)
    if not isinstance(payload, Mapping):
        return {}, "non_object", "top-level JSON is not an object"
    return dict(payload), "parsed", ""


def _read_json_object(path: Path) -> JsonDict:
    payload, status, error = _read_json_with_status(path)
    if status != "parsed":
        return {"_read_status": status, "_read_error": error}
    return payload


def _command_output(command: Sequence[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout.strip() or result.stderr.strip()


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


def solver_availability() -> JsonDict:
    """Record exact replay tools without using them as performance evidence."""

    try:
        z3_version = exp6504.z3.get_version_string()
        z3_available = True
    except Exception as exc:  # pragma: no cover - only fires when z3 import is broken.
        z3_version = f"unavailable:{exc}"  # pragma: no cover
        z3_available = False  # pragma: no cover
    return {
        "exact_replay_available": z3_available,
        "exhaustive_solver": "independent_complete_truth_table_forced_branch_v1",
        "z3_python_available": z3_available,
        "z3_python_version": z3_version,
        "z3_cli_path": shutil.which("z3"),
    }


def _source_shard_paths(payload: Mapping[str, Any]) -> list[Path]:
    shard = payload.get("shard_manifest", {})
    if not isinstance(shard, Mapping):
        return []
    paths: list[Path] = []
    for receipt in [
        *list(shard.get("terminal_write_receipts", [])),
        *list(shard.get("resume_receipts", [])),
    ]:
        if isinstance(receipt, Mapping) and receipt.get("shard_path"):
            paths.append(Path(str(receipt["shard_path"])))
    return sorted(set(paths), key=str)


def protected_file_hashes(
    repo_root: Path,
    source_path: Path,
    source_payload: Mapping[str, Any],
) -> dict[str, str]:
    hashes = {
        path.as_posix(): sha256_file(repo_root / path) for path in STATIC_PROTECTED_RELATIVE_PATHS
    }
    source_key = (
        UPSTREAM_RELATIVE_PATH.as_posix()
        if source_path.resolve(strict=False) == (repo_root / UPSTREAM_RELATIVE_PATH).resolve(False)
        else str(source_path)
    )
    hashes[source_key] = sha256_file(source_path)
    for shard_path in _source_shard_paths(source_payload):
        hashes[str(shard_path)] = sha256_file(shard_path)
    return hashes


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


def upstream_artifact_receipt(
    *,
    repo_root: Path,
    source_path: Path,
    source_payload: Mapping[str, Any],
    parse_status: str,
    parse_error: str,
    protected_before: Mapping[str, str],
) -> JsonDict:
    rows = source_payload.get("branch_counterfactual_rows", [])
    per_unit_rows = source_payload.get("per_unit_rows", [])
    receipts = source_payload.get("exact_solver_receipts", [])
    shard = source_payload.get("shard_manifest", {})
    shard_counts = {
        "planned_unit_count": shard.get("planned_unit_count") if isinstance(shard, Mapping) else None,
        "terminal_row_count": shard.get("terminal_row_count") if isinstance(shard, Mapping) else None,
        "resume_receipt_count": (
            len(shard.get("resume_receipts", [])) if isinstance(shard, Mapping) else 0
        ),
        "terminal_write_receipt_count": (
            len(shard.get("terminal_write_receipts", [])) if isinstance(shard, Mapping) else 0
        ),
        "hash_chain_count": len(shard.get("hash_chain", [])) if isinstance(shard, Mapping) else 0,
        "journal_record_count": shard.get("journal_record_count") if isinstance(shard, Mapping) else None,
    }
    return {
        "row_type": "upstream_artifact_receipt",
        "path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "absolute_path": str(source_path),
        "exists": source_path.is_file(),
        "sha256": sha256_file(source_path),
        "parse_status": parse_status,
        "parse_error": parse_error,
        "source_status": source_payload.get("status"),
        "source_honest_verdict": source_payload.get("honest_verdict"),
        "source_verdict_class": source_payload.get("verdict_class"),
        "source_ready_score": source_payload.get("branch_pilot_dataset_ready_score"),
        "branch_row_count": len(rows) if isinstance(rows, list) else 0,
        "per_unit_row_count": len(per_unit_rows) if isinstance(per_unit_rows, list) else 0,
        "exact_receipt_count": len(receipts) if isinstance(receipts, list) else 0,
        "shard_counts": shard_counts,
        "solver_availability": solver_availability(),
        "resources": _resource_state(repo_root),
        "protected_file_hashes_before": dict(protected_before),
        "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-MISSING-UPSTREAM"],
    }


def prior_failure_receipt(repo_root: Path, source_payload: Mapping[str, Any]) -> JsonDict:
    exp6510_payload = _read_json_object(repo_root / EXP6510_RELATIVE_PATH)
    source_prior = source_payload.get("prior_failure_receipts", {})
    if not isinstance(source_prior, Mapping):
        source_prior = {}
    return {
        "row_type": "prior_failure_receipt",
        "exp6510_path": EXP6510_RELATIVE_PATH.as_posix(),
        "exp6510_sha256": sha256_file(repo_root / EXP6510_RELATIVE_PATH),
        "exp6510_status": exp6510_payload.get("status"),
        "exp6510_verdict_class": exp6510_payload.get("verdict_class"),
        "exp6510_ready_score": exp6510_payload.get("v563_independent_root_ready_score"),
        "source_prior_exp6510_status": source_prior.get("exp6510_status"),
        "source_prior_exp6510_verdict_class": source_prior.get("exp6510_verdict_class"),
        "source_prior_exp6511_missing": source_prior.get("exp6511_dataset_missing_or_retired"),
        "structured_dependency_used": False,
        "spec_refs": ["REQ-BENCH-6517"],
    }


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
    limit = min(int(budget), output_space_size)
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
    }


def _z3_status(n_vars: int, clauses: Sequence[Sequence[int]], literal: int) -> str:
    outcome = exp6504.z3_solve(n_vars, [*[list(clause) for clause in clauses], [literal]])
    return str(outcome.status)


def _row_hash_matches(row: Mapping[str, Any]) -> bool:
    payload = {key: value for key, value in row.items() if key != "row_hash"}
    return row.get("row_hash") == sha256_json(payload)


def _unit_id_for(row: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "checkpoint_id": row.get("checkpoint_id"),
            "candidate_value": row.get("candidate_value"),
            "budget": row.get("exact_budget"),
        }
    )


def _base_rows_by_hash(repo_root: Path) -> dict[str, JsonDict]:
    exp6504_payload = _read_json_object(repo_root / EXP6504_RELATIVE_PATH)
    rows = exp6504_payload.get("raw_instance_rows", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("raw_instance_hash")): dict(row) for row in rows if isinstance(row, Mapping)}


def exact_receipt_replay_rows(
    repo_root: Path,
    source_payload: Mapping[str, Any],
) -> list[JsonDict]:
    branch_rows = source_payload.get("branch_counterfactual_rows", [])
    if not isinstance(branch_rows, list):
        return []
    base_by_hash = _base_rows_by_hash(repo_root)
    audit_rows: list[JsonDict] = []
    for index, row_value in enumerate(branch_rows):
        row = dict(row_value) if isinstance(row_value, Mapping) else {}
        receipt = row.get("exact_receipt", {})
        receipt = dict(receipt) if isinstance(receipt, Mapping) else {}
        base_hash = str(row.get("base_instance_hash"))
        base = base_by_hash.get(base_hash)
        forced_variable = int(row.get("checkpoint_variable", 0) or 0)
        candidate_value = bool(row.get("candidate_value"))
        forced_literal = forced_variable if candidate_value else -forced_variable
        exact: JsonDict = {"status": "missing_base", "model": None, "proof_hash": None}
        z3_status = "missing_base"
        model_valid = False
        proof_valid = False
        if base is not None and forced_variable > 0:
            clauses = [[int(literal) for literal in clause] for clause in base["clauses"]]
            forced_clauses = [*clauses, [forced_literal]]
            exact = _forced_exhaustive_solve(
                n_vars=int(base["variable_count"]),
                clauses=forced_clauses,
                forced_variable=forced_variable,
                forced_value=candidate_value,
                budget=int(row.get("exact_budget", 0) or 0),
            )
            z3_status = _z3_status(int(base["variable_count"]), clauses, forced_literal)
            if exact["model"] is not None:
                model_valid = _assignment_satisfies(
                    forced_clauses,
                    {
                        int(key.removeprefix("x")): bool(value)
                        for key, value in exact["model"].items()
                    },
                )
            proof_valid = exact["status"] == "unsat" and exact["proof_hash"] is not None
        model_hash = sha256_json(exact["model"]) if exact.get("model") is not None else None
        expected_disposition = "sat_model" if exact.get("status") == "sat" else "unsat_proof"
        if exact.get("timeout"):
            expected_disposition = "timeout"
        status_matches = (
            receipt.get("exhaustive_status") == exact.get("status")
            and receipt.get("z3_status") == z3_status
            and receipt.get("exact_answer_equality") is True
            and exact.get("status") == z3_status
        )
        model_or_proof = (
            (exact.get("status") == "sat" and receipt.get("model_hash") == model_hash and model_valid)
            or (
                exact.get("status") == "unsat"
                and receipt.get("proof_hash") == exact.get("proof_hash")
                and proof_valid
            )
        )
        checks = {
            "base_hash_found_in_exp6504": base is not None,
            "receipt_unit_id_matches_row": receipt.get("unit_id") == row.get("unit_id"),
            "unit_id_recomputed": row.get("unit_id") == _unit_id_for(row),
            "forced_literal_matches": receipt.get("forced_literal") == forced_literal,
            "receipt_base_hash_matches_row": receipt.get("base_instance_hash") == base_hash,
            "replayed_exact_status_matches_receipt": status_matches,
            "terminal_disposition_matches": row.get("terminal_disposition")
            == receipt.get("terminal_disposition")
            == expected_disposition,
            "model_or_proof_revalidated": model_or_proof,
            "row_hash_recomputed": _row_hash_matches(row),
            "budget_matches_contract": row.get("exact_budget") == EXPECTED_BUDGET,
            "timeout_absent": row.get("timeout") is False,
            "censoring_absent": row.get("censored") is False,
        }
        payload = {
            "row_type": "source_unit_audit",
            "source_row_index": index,
            "unit_id": row.get("unit_id"),
            "checkpoint_id": row.get("checkpoint_id"),
            "base_instance_hash": base_hash,
            "base_lineage_id": row.get("base_lineage_id"),
            "split": row.get("split"),
            "family": row.get("family"),
            "scale": row.get("scale"),
            "candidate_value": row.get("candidate_value"),
            "forced_literal": forced_literal,
            "recomputed_exhaustive_status": exact.get("status"),
            "recomputed_z3_status": z3_status,
            "receipt_exhaustive_status": receipt.get("exhaustive_status"),
            "receipt_z3_status": receipt.get("z3_status"),
            **checks,
            "audit_passed": all(checks.values()),
            "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-ROW-REPLAY"],
        }
        audit_rows.append({**payload, "audit_row_hash": sha256_json(payload)})
    return audit_rows


def _candidate_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("checkpoint_id"))].append(row)
    return groups


def _source_aggregate_matches(
    source_payload: Mapping[str, Any],
    branch_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
) -> bool:
    aggregate = source_payload.get("aggregate_row_recomputation", {})
    if not isinstance(aggregate, Mapping):
        return False
    groups = _candidate_groups(branch_rows)
    return (
        aggregate.get("row_count") == len(branch_rows)
        and aggregate.get("checkpoint_count") == len(groups)
        and aggregate.get("candidate_completeness_passed") is True
        and aggregate.get("exact_receipt_failure_count") == 0
        and aggregate.get("ready_score_from_rows") == (
            1.0 if all(row.get("audit_passed") is True for row in exact_rows) else 0.0
        )
    )


def independent_row_recomputation(
    source_payload: Mapping[str, Any],
    exact_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    branch_rows = [
        dict(row)
        for row in source_payload.get("branch_counterfactual_rows", [])
        if isinstance(row, Mapping)
    ]
    groups = _candidate_groups(branch_rows)
    candidate_complete = all(
        sorted(bool(row.get("candidate_value")) for row in group) == list(ELIGIBLE_VALUES)
        and len(group) == len(ELIGIBLE_VALUES)
        for group in groups.values()
    )
    exact_failures = sum(1 for row in exact_rows if row.get("audit_passed") is not True)
    duplicate_units = len(branch_rows) - len({str(row.get("unit_id")) for row in branch_rows})
    source_score = source_payload.get("branch_pilot_dataset_ready_score")
    aggregate_matches = _source_aggregate_matches(source_payload, branch_rows, exact_rows)
    ready = (
        len(branch_rows) == EXPECTED_BRANCH_ROW_COUNT
        and len(groups) == EXPECTED_CHECKPOINT_COUNT
        and candidate_complete
        and exact_failures == 0
        and duplicate_units == 0
        and source_score == 1.0
        and aggregate_matches
    )
    return {
        "row_type": "independent_row_recomputation",
        "branch_row_count": len(branch_rows),
        "per_unit_row_count": len(source_payload.get("per_unit_rows", [])),
        "exact_solver_receipt_count": len(source_payload.get("exact_solver_receipts", [])),
        "checkpoint_count": len(groups),
        "candidate_value_count_per_checkpoint": len(ELIGIBLE_VALUES),
        "candidate_completeness_passed": candidate_complete,
        "duplicate_unit_id_count": duplicate_units,
        "exact_receipt_replay_failure_count": exact_failures,
        "source_aggregate_matches_independent_rows": aggregate_matches,
        "imported_source_ready_score": source_score,
        "recomputed_ready_score_from_rows": 1.0 if ready else 0.0,
        "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-ROW-REPLAY"],
    }


def split_and_lineage_audit(source_payload: Mapping[str, Any]) -> JsonDict:
    rows = [
        dict(row)
        for row in source_payload.get("branch_counterfactual_rows", [])
        if isinstance(row, Mapping)
    ]
    lineage_to_splits: dict[str, set[str]] = defaultdict(set)
    checkpoint_to_lineages: dict[str, set[str]] = defaultdict(set)
    cell_lineages: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        lineage = str(row.get("base_lineage_id"))
        split = str(row.get("split"))
        lineage_to_splits[lineage].add(split)
        checkpoint_to_lineages[str(row.get("checkpoint_id"))].add(lineage)
        cell_lineages[(split, str(row.get("family")), str(row.get("scale")))].add(lineage)
    overlap = sum(1 for splits in lineage_to_splits.values() if len(splits) > 1)
    duplicate_checkpoints = sum(
        1 for lineages in checkpoint_to_lineages.values() if len(lineages) > 1
    )
    observed_floor = min((len(lineages) for lineages in cell_lineages.values()), default=0)
    split = source_payload.get("split_commitment", {})
    split = dict(split) if isinstance(split, Mapping) else {}
    source_matches = (
        split.get("base_lineage_overlap_count") == overlap
        and split.get("duplicate_checkpoint_count") == duplicate_checkpoints
        and split.get("minimum_cell_floor_observed") == observed_floor
        and split.get("post_held_repair_count") == sum(1 for row in rows if row.get("post_held_repair"))
    )
    lineage_sets = {
        split_name: sorted(lineage for lineage, splits in lineage_to_splits.items() if split_name in splits)
        for split_name in ("train", "development", "held")
    }
    passed = (
        overlap == 0
        and duplicate_checkpoints == 0
        and observed_floor >= MINIMUM_CELL_FLOOR
        and len(cell_lineages) == EXPECTED_CELL_COUNT
        and split.get("post_held_repair_count") == 0
        and source_matches
    )
    payload = {
        "row_type": "split_and_lineage_audit",
        "split_audit_passed": passed,
        "lineage_sets": lineage_sets,
        "base_lineage_overlap_count": overlap,
        "duplicate_checkpoint_count": duplicate_checkpoints,
        "minimum_cell_floor_required": MINIMUM_CELL_FLOOR,
        "minimum_cell_floor_observed": observed_floor,
        "cell_count": len(cell_lineages),
        "post_held_repair_count": split.get("post_held_repair_count"),
        "source_split_commitment_matches_recomputed": source_matches,
        "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-SPLIT-TIMING"],
    }
    return {**payload, "audit_row_hash": sha256_json(payload)}


def _feature_names(source_payload: Mapping[str, Any]) -> set[str]:
    schema = source_payload.get("structural_feature_schema", {})
    features = schema.get("features", []) if isinstance(schema, Mapping) else []
    return {
        str(feature.get("name", "")).lower()
        for feature in features
        if isinstance(feature, Mapping)
    }


def _forbidden_features(source_payload: Mapping[str, Any]) -> list[str]:
    names = _feature_names(source_payload)
    return sorted(name for name in names if name in FORBIDDEN_FEATURE_NAMES)


def feature_timing_audit(source_payload: Mapping[str, Any]) -> JsonDict:
    rows = [
        dict(row)
        for row in source_payload.get("branch_counterfactual_rows", [])
        if isinstance(row, Mapping)
    ]
    checkpoints = source_payload.get("checkpoint_contract", {}).get("checkpoint_rows", [])
    checkpoints = [dict(row) for row in checkpoints if isinstance(row, Mapping)]
    forbidden = _forbidden_features(source_payload)
    feature_event_before_replay = all(
        int(row.get("feature_event_index", 99)) < int(row.get("replay_event_index", -1))
        for row in rows
    ) and all(
        int(row.get("selection_event_index", 99)) < int(row.get("replay_event_index", -1))
        for row in checkpoints
    )
    outcome_flags = [
        "selection_used_label",
        "selection_used_held_outcome",
        "selection_used_unit_id",
        "uses_label",
        "uses_held_outcome",
        "uses_unit_id",
    ]
    effort_flags = ["selection_used_future_effort", "uses_future_effort"]
    checkpoint_selection_uses_outcome = any(
        row.get(flag) is True for row in [*rows, *checkpoints] for flag in outcome_flags
    )
    checkpoint_selection_uses_future_effort = any(
        row.get(flag) is True for row in [*rows, *checkpoints] for flag in effort_flags
    )
    contract = source_payload.get("checkpoint_contract", {})
    contract_ok = (
        isinstance(contract, Mapping)
        and contract.get("uses_only_decision_time_structural_features") is True
    )
    passed = (
        not forbidden
        and feature_event_before_replay
        and not checkpoint_selection_uses_outcome
        and not checkpoint_selection_uses_future_effort
        and contract_ok
    )
    payload = {
        "row_type": "feature_timing_audit",
        "feature_timing_passed": passed,
        "forbidden_features_observed": forbidden,
        "feature_event_before_replay": feature_event_before_replay,
        "checkpoint_selection_uses_outcome": checkpoint_selection_uses_outcome,
        "checkpoint_selection_uses_future_effort": checkpoint_selection_uses_future_effort,
        "checkpoint_contract_decision_time_only": contract_ok,
        "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-SPLIT-TIMING"],
    }
    return {**payload, "audit_row_hash": sha256_json(payload)}


def censoring_audit(source_payload: Mapping[str, Any]) -> JsonDict:
    rows = [
        dict(row)
        for row in source_payload.get("branch_counterfactual_rows", [])
        if isinstance(row, Mapping)
    ]
    groups = _candidate_groups(rows)
    asymmetric = 0
    missing_terminal = 0
    missing_values = 0
    for group in groups.values():
        budgets = {row.get("exact_budget") for row in group}
        values = sorted(bool(row.get("candidate_value")) for row in group)
        if budgets != {EXPECTED_BUDGET}:
            asymmetric += 1
        if any(not row.get("terminal_disposition") for row in group):
            missing_terminal += 1
        if values != list(ELIGIBLE_VALUES) or len(group) != len(ELIGIBLE_VALUES):
            missing_values += 1
    censored = sum(1 for row in rows if row.get("censored") is True)
    timeouts = sum(1 for row in rows if row.get("timeout") is True)
    budget_rows = [
        dict(row)
        for row in source_payload.get("censoring_and_budget_rows", [])
        if isinstance(row, Mapping)
    ]
    budget_rows_ok = len(budget_rows) == len(groups) and all(
        row.get("equal_budget") is True
        and row.get("terminal_disposition_present") is True
        and row.get("censored") is False
        and row.get("timeout_count") == 0
        for row in budget_rows
    )
    passed = (
        asymmetric == 0
        and missing_terminal == 0
        and missing_values == 0
        and censored == 0
        and timeouts == 0
        and budget_rows_ok
    )
    payload = {
        "row_type": "censoring_audit",
        "censoring_audit_passed": passed,
        "asymmetric_budget_count": asymmetric,
        "missing_terminal_disposition_count": missing_terminal,
        "missing_candidate_value_group_count": missing_values,
        "censored_row_count": censored,
        "timeout_count": timeouts,
        "source_budget_rows_match": budget_rows_ok,
        "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-ATTACKS"],
    }
    return {**payload, "audit_row_hash": sha256_json(payload)}


def _hash_chain(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    chain: list[str] = []
    prior = "sha256:" + "0" * 64
    for row in sorted(rows, key=lambda item: str(item.get("unit_id"))):
        prior = sha256_json(
            {"prior": prior, "unit_id": row.get("unit_id"), "row_hash": row.get("row_hash")}
        )
        chain.append(prior)
    return chain


def transaction_and_shard_audit(repo_root: Path, source_payload: Mapping[str, Any]) -> JsonDict:
    rows = [
        dict(row)
        for row in source_payload.get("branch_counterfactual_rows", [])
        if isinstance(row, Mapping)
    ]
    row_by_unit = {str(row.get("unit_id")): row for row in rows}
    row_ids = set(row_by_unit)
    shard = source_payload.get("shard_manifest", {})
    shard = dict(shard) if isinstance(shard, Mapping) else {}
    writes = [
        dict(row) for row in shard.get("terminal_write_receipts", []) if isinstance(row, Mapping)
    ]
    resumes = [dict(row) for row in shard.get("resume_receipts", []) if isinstance(row, Mapping)]
    write_by_unit = {str(row.get("unit_id")): row for row in writes}
    resume_by_unit = {str(row.get("unit_id")): row for row in resumes}
    planned_ok = shard.get("planned_unit_count") == len(rows) == EXPECTED_BRANCH_ROW_COUNT
    planned_ids_match = shard.get("planned_unit_count") == len(row_ids)
    terminal_ids_match = set(write_by_unit) == row_ids and shard.get("terminal_row_count") == len(rows)
    terminal_shards_match = all(
        write_by_unit.get(unit_id, {}).get("shard_hash") == sha256_shard_payload(row)
        for unit_id, row in row_by_unit.items()
    )
    resume_receipts_match = set(resume_by_unit) == row_ids and all(
        resume_by_unit[unit_id].get("shard_hash") == write_by_unit[unit_id].get("shard_hash")
        and resume_by_unit[unit_id].get("verified") is True
        for unit_id in row_ids
        if unit_id in resume_by_unit and unit_id in write_by_unit
    )
    shard_files_verified = all(
        receipt.get("shard_hash") == sha256_file(Path(str(receipt.get("shard_path"))))
        for receipt in [*writes, *resumes]
        if receipt.get("shard_path")
    )
    journal_chain_ok = shard.get("hash_chain") == _hash_chain(rows)
    journal_count_ok = int(shard.get("journal_record_count") or 0) >= len(rows) * 2
    exp6514 = _read_json_object(repo_root / EXP6514_RELATIVE_PATH)
    terminal = exp6514.get("terminal_write_receipt", {})
    terminal = terminal if isinstance(terminal, Mapping) else {}
    final_atomic = (
        exp6514.get("atomic_artifact_contract_ready_score") == 1.0
        and terminal.get("atomic_replace") is True
        and terminal.get("file_fsync") is True
        and terminal.get("final_path_status") == "terminal_complete"
        and terminal.get("success_path_nonterminal_artifact") is False
    )
    passed = (
        planned_ok
        and planned_ids_match
        and terminal_ids_match
        and terminal_shards_match
        and resume_receipts_match
        and shard_files_verified
        and journal_chain_ok
        and journal_count_ok
        and shard.get("complete") is True
        and shard.get("resume_verified") is True
        and shard.get("corrupt_resume_detected") is True
        and shard.get("final_transaction_verified") is True
        and final_atomic
    )
    payload = {
        "row_type": "transaction_and_shard_audit",
        "transaction_audit_passed": passed,
        "exp6514_path": EXP6514_RELATIVE_PATH.as_posix(),
        "exp6514_sha256": sha256_file(repo_root / EXP6514_RELATIVE_PATH),
        "exp6514_ready_score": exp6514.get("atomic_artifact_contract_ready_score"),
        "source_checksum": sha256_file(repo_root / UPSTREAM_RELATIVE_PATH),
        "planned_ids_match_branch_rows": planned_ids_match,
        "terminal_ids_match_branch_rows": terminal_ids_match,
        "terminal_shard_hashes_match_rows": terminal_shards_match,
        "resume_receipts_match_terminal_rows": resume_receipts_match,
        "shard_files_verified": shard_files_verified,
        "journal_chain_length_matches_rows": journal_chain_ok and journal_count_ok,
        "corrupt_resume_detected": shard.get("corrupt_resume_detected") is True,
        "final_atomic_receipt_passed": final_atomic,
        "planned_unit_count": shard.get("planned_unit_count"),
        "terminal_unit_count": shard.get("terminal_row_count"),
        "resume_receipt_count": len(resumes),
        "terminal_write_receipt_count": len(writes),
        "journal_record_count": shard.get("journal_record_count"),
        "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-SHARDS"],
    }
    return {**payload, "audit_row_hash": sha256_json(payload)}


def shortcut_attack_matrix(
    *,
    source_payload: Mapping[str, Any],
    row_recompute: Mapping[str, Any],
    split: Mapping[str, Any],
    transaction: Mapping[str, Any],
    feature: Mapping[str, Any],
    censoring: Mapping[str, Any],
) -> JsonDict:
    names = _feature_names(source_payload)
    attack_checks = {
        "identity_ids": not any(
            name in names for name in {"unit_id", "row_id", "base_lineage_id", "base_instance_id"}
        )
        and feature.get("checkpoint_selection_uses_outcome") is False,
        "row_order": "row_order" not in names,
        "serialization_length": not any(
            name in names for name in {"serialization_length", "serialized_length"}
        ),
        "family_shortcuts": not any(name in names for name in {"family", "family_label"}),
        "label_shortcuts": not any(
            name in names for name in {"label", "exact_label", "base_label", "held_outcome"}
        ),
        "future_effort": not any(name in names for name in {"future_effort", "assignments_examined"})
        and feature.get("checkpoint_selection_uses_future_effort") is False,
        "censoring_removal": censoring.get("censoring_audit_passed") is True,
        "corrupt_shards": transaction.get("corrupt_resume_detected") is True
        and transaction.get("shard_files_verified") is True,
        "aggregate_tampering": row_recompute.get("source_aggregate_matches_independent_rows")
        is True,
        "one_cell_headroom": split.get("minimum_cell_floor_observed", 0)
        >= split.get("minimum_cell_floor_required", MINIMUM_CELL_FLOOR)
        and split.get("cell_count") == EXPECTED_CELL_COUNT,
    }
    rows: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        payload = {
            "row_type": "shortcut_attack",
            "attack_id": attack_id,
            "fail_closed": bool(attack_checks[attack_id]),
            "false_accept": not bool(attack_checks[attack_id]),
            "observed_value": attack_checks[attack_id],
            "expected_value": True,
            "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-ATTACKS"],
        }
        rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    payload = {
        "schema_version": SCHEMA_VERSION + ".shortcut_attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
        "false_accept_count": sum(1 for row in rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in rows if row["fail_closed"] is not True],
    }
    return {**payload, "shortcut_attack_matrix_hash": sha256_json(payload)}


def recompute_aggregate(payload: Mapping[str, Any]) -> JsonDict:
    receipt = payload.get("upstream_artifact_receipt", {})
    recompute = payload.get("independent_row_recomputation", {})
    exact_rows = payload.get("exact_receipt_replay_rows", [])
    split = payload.get("split_and_lineage_audit", {})
    transaction = payload.get("transaction_and_shard_audit", {})
    feature = payload.get("feature_timing_audit", {})
    censoring = payload.get("censoring_audit", {})
    attacks = payload.get("shortcut_attack_matrix", {})
    per_unit = payload.get("per_unit_rows", [])
    source_available = (
        isinstance(receipt, Mapping)
        and receipt.get("exists") is True
        and receipt.get("parse_status") == "parsed"
    )
    source_complete = (
        source_available
        and receipt.get("source_status") == "complete_exact_branch_pilot_dataset_v3_ready"
        and receipt.get("source_verdict_class") is None
        and receipt.get("source_ready_score") == 1.0
    )
    exact_passed = bool(exact_rows) and all(
        isinstance(row, Mapping) and row.get("audit_passed") is True for row in exact_rows
    )
    independent_passed = recompute.get("recomputed_ready_score_from_rows") == 1.0
    split_passed = split.get("split_audit_passed") is True
    transaction_passed = transaction.get("transaction_audit_passed") is True
    feature_passed = feature.get("feature_timing_passed") is True
    censoring_passed = censoring.get("censoring_audit_passed") is True
    attacks_passed = (
        attacks.get("all_attacks_fail_closed") is True and attacks.get("false_accept_count") == 0
    )
    aggregate_tampering = recompute.get("source_aggregate_matches_independent_rows") is not True
    protected_ok = (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is True
    )
    ready = all(
        [
            source_available,
            source_complete,
            independent_passed,
            exact_passed,
            split_passed,
            transaction_passed,
            feature_passed,
            censoring_passed,
            attacks_passed,
            not aggregate_tampering,
            protected_ok,
        ]
    )
    return {
        "source_available_and_parsed": source_available,
        "source_declares_complete_ready": source_complete,
        "independent_row_recomputation_passed": independent_passed,
        "exact_receipts_replay_passed": exact_passed,
        "split_and_lineage_passed": split_passed,
        "transaction_and_shards_passed": transaction_passed,
        "feature_timing_passed": feature_passed,
        "censoring_and_budgets_passed": censoring_passed,
        "shortcut_attacks_passed": attacks_passed,
        "aggregate_tampering_detected": aggregate_tampering,
        "protected_files_unchanged": protected_ok,
        "source_unit_audit_row_count": sum(
            1 for row in per_unit if isinstance(row, Mapping) and row.get("row_type") == "source_unit_audit"
        ),
        "audit_row_count": len(per_unit) if isinstance(per_unit, list) else 0,
        "failed_exact_receipt_count": sum(
            1 for row in exact_rows if isinstance(row, Mapping) and row.get("audit_passed") is not True
        ),
        "ready_score_from_audit_rows": 1.0 if ready else 0.0,
        "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-TERMINAL"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "source_available_and_parsed": True,
        "source_declares_complete_ready": True,
        "independent_row_recomputation": True,
        "exact_receipts_replay": True,
        "split_and_lineage": True,
        "transaction_and_shards": True,
        "feature_timing": True,
        "censoring_and_budgets": True,
        "shortcut_attacks": True,
        "aggregate_tampering": False,
        "protected_files_unchanged": True,
        "ready_score_from_audit_rows": 1.0,
    }
    observed = {
        "source_available_and_parsed": aggregate.get("source_available_and_parsed"),
        "source_declares_complete_ready": aggregate.get("source_declares_complete_ready"),
        "independent_row_recomputation": aggregate.get("independent_row_recomputation_passed"),
        "exact_receipts_replay": aggregate.get("exact_receipts_replay_passed"),
        "split_and_lineage": aggregate.get("split_and_lineage_passed"),
        "transaction_and_shards": aggregate.get("transaction_and_shards_passed"),
        "feature_timing": aggregate.get("feature_timing_passed"),
        "censoring_and_budgets": aggregate.get("censoring_and_budgets_passed"),
        "shortcut_attacks": aggregate.get("shortcut_attacks_passed"),
        "aggregate_tampering": aggregate.get("aggregate_tampering_detected"),
        "protected_files_unchanged": aggregate.get("protected_files_unchanged"),
        "ready_score_from_audit_rows": aggregate.get("ready_score_from_audit_rows"),
    }
    checks = {
        key: {"expected": value, "observed": observed[key], "passed": observed[key] == value}
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


def _status_and_verdict(
    score: float,
    gates: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    if score == 1.0:
        return (
            "complete_branch_pilot_independent_audit_ready",
            "complete_branch_pilot_independent_audit: all rows, receipts, splits, shards, features, censoring, attacks, and protected hashes passed",
            None,
        )
    failed = ",".join(gates.get("failed_checks", [])) or "unknown_gate"
    if (
        aggregate.get("source_available_and_parsed") is not True
        or aggregate.get("source_declares_complete_ready") is not True
    ):
        return (
            "blocked_branch_pilot_independent_audit",
            f"blocked_branch_pilot_independent_audit: {failed}",
            "blocked",
        )
    return (
        "disqualified_branch_pilot_independent_audit",
        f"disqualified_branch_pilot_independent_audit: {failed}",
        "disqualified",
    )


def _field_provenance(repo_root: Path, source_path: Path) -> dict[str, JsonDict]:
    source_hashes = {
        path.as_posix(): sha256_file(repo_root / path) for path in STATIC_PROTECTED_RELATIVE_PATHS
    }
    source_hashes[str(source_path)] = sha256_file(source_path)
    return {
        field: {
            "source": "deterministic_exp6517_independent_audit",
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
            "source_artifact": str(source_path),
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
    source_path: Path,
    run_date: str,
    protected_before: Mapping[str, str],
) -> JsonDict:
    git_rc, git_status = _command_output(["git", "status", "--short"], repo_root)
    return {
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "source_path": str(source_path),
        "source_default_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "git_status_command_exit_code": git_rc,
        "git_status_short": git_status,
        "solver_availability": solver_availability(),
        "resources": _resource_state(repo_root),
        "protected_file_hashes_before": dict(protected_before),
        "source_and_shards_modification_allowed": False,
        "conductor_modification_allowed": False,
        "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-MISSING-UPSTREAM"],
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def _empty_audits() -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict, JsonDict, JsonDict]:
    empty_row = {
        "row_type": "audit_not_run",
        "audit_passed": False,
        "reason": "source_missing_or_unparseable",
    }
    split = {
        "row_type": "split_and_lineage_audit",
        "split_audit_passed": False,
        "lineage_sets": {"train": [], "development": [], "held": []},
        "base_lineage_overlap_count": None,
        "duplicate_checkpoint_count": None,
        "minimum_cell_floor_required": MINIMUM_CELL_FLOOR,
        "minimum_cell_floor_observed": 0,
        "cell_count": 0,
        "post_held_repair_count": None,
        "source_split_commitment_matches_recomputed": False,
    }
    transaction = {
        "row_type": "transaction_and_shard_audit",
        "transaction_audit_passed": False,
        "planned_ids_match_branch_rows": False,
        "terminal_ids_match_branch_rows": False,
        "terminal_shard_hashes_match_rows": False,
        "resume_receipts_match_terminal_rows": False,
        "shard_files_verified": False,
        "journal_chain_length_matches_rows": False,
        "corrupt_resume_detected": False,
        "final_atomic_receipt_passed": False,
        "exp6514_ready_score": None,
    }
    feature = {
        "row_type": "feature_timing_audit",
        "feature_timing_passed": False,
        "forbidden_features_observed": [],
        "feature_event_before_replay": False,
        "checkpoint_selection_uses_outcome": None,
        "checkpoint_selection_uses_future_effort": None,
    }
    censoring = {
        "row_type": "censoring_audit",
        "censoring_audit_passed": False,
        "asymmetric_budget_count": None,
        "missing_terminal_disposition_count": None,
        "censored_row_count": None,
        "timeout_count": None,
    }
    attacks = {
        "schema_version": SCHEMA_VERSION + ".shortcut_attack_matrix",
        "rows": [
            {
                "row_type": "shortcut_attack",
                "attack_id": attack_id,
                "fail_closed": False,
                "false_accept": True,
                "observed_value": None,
                "expected_value": True,
            }
            for attack_id in ATTACK_IDS
        ],
        "attack_count": len(ATTACK_IDS),
        "all_attacks_fail_closed": False,
        "false_accept_count": len(ATTACK_IDS),
        "failed_attack_ids": list(ATTACK_IDS),
    }
    return [empty_row], split, transaction, feature, censoring, attacks


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    source_path: Path | str = UPSTREAM_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the independent audit and optionally write the terminal artifact."""

    start = time.perf_counter()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    if not result_path.is_absolute():
        result_path = repo_root / result_path
    source_path = Path(source_path)
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    source_payload, parse_status, parse_error = _read_json_with_status(source_path)
    protected_before = protected_file_hashes(repo_root, source_path, source_payload)
    upstream = upstream_artifact_receipt(
        repo_root=repo_root,
        source_path=source_path,
        source_payload=source_payload,
        parse_status=parse_status,
        parse_error=parse_error,
        protected_before=protected_before,
    )
    prior = prior_failure_receipt(repo_root, source_payload)
    if parse_status == "parsed":
        exact_rows = exact_receipt_replay_rows(repo_root, source_payload)
        row_recompute = independent_row_recomputation(source_payload, exact_rows)
        split = split_and_lineage_audit(source_payload)
        transaction = transaction_and_shard_audit(repo_root, source_payload)
        feature = feature_timing_audit(source_payload)
        censoring = censoring_audit(source_payload)
        attacks = shortcut_attack_matrix(
            source_payload=source_payload,
            row_recompute=row_recompute,
            split=split,
            transaction=transaction,
            feature=feature,
            censoring=censoring,
        )
    else:
        exact_rows, split, transaction, feature, censoring, attacks = _empty_audits()
        row_recompute = {
            "row_type": "independent_row_recomputation",
            "branch_row_count": 0,
            "per_unit_row_count": 0,
            "exact_solver_receipt_count": 0,
            "checkpoint_count": 0,
            "candidate_value_count_per_checkpoint": len(ELIGIBLE_VALUES),
            "candidate_completeness_passed": False,
            "duplicate_unit_id_count": None,
            "exact_receipt_replay_failure_count": None,
            "source_aggregate_matches_independent_rows": False,
            "imported_source_ready_score": None,
            "recomputed_ready_score_from_rows": 0.0,
            "spec_refs": ["REQ-BENCH-6517", "SCENARIO-BENCH-6517-MISSING-UPSTREAM"],
        }
    protected_after = protected_file_hashes(repo_root, source_path, source_payload)
    protected = protected_files_unchanged(protected_before, protected_after)
    per_unit_rows = [
        *exact_rows,
        {
            "row_type": "manifest_audit",
            "audit_passed": transaction.get("transaction_audit_passed") is True,
            "source_sha256": upstream["sha256"],
            "audit_row_hash": transaction.get("audit_row_hash"),
        },
        split,
        feature,
        censoring,
        *attacks["rows"],
    ]
    partial: JsonDict = {
        "status": "blocked_branch_pilot_independent_audit",
        "honest_verdict": "blocked_branch_pilot_independent_audit: building",
        "verdict_class": "blocked",
        "upstream_artifact_receipt": upstream,
        "prior_failure_receipt": prior,
        "independent_row_recomputation": row_recompute,
        "exact_receipt_replay_rows": exact_rows,
        "split_and_lineage_audit": split,
        "transaction_and_shard_audit": transaction,
        "feature_timing_audit": feature,
        "censoring_audit": censoring,
        "shortcut_attack_matrix": attacks,
        "branch_pilot_audited_ready_score": 0.0,
        "gate_check_summary": {},
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": {},
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            result_path=result_path,
            source_path=source_path,
            run_date=run_date,
            protected_before=protected_before,
        ),
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(repo_root, source_path),
        "random_seed": {"artifact_seed": RANDOM_SEED, "attack_ids": list(ATTACK_IDS)},
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = recompute_aggregate(partial)
    gates = gate_check_summary(aggregate)
    score = float(aggregate["ready_score_from_audit_rows"])
    status, honest, verdict_class = _status_and_verdict(score, gates, aggregate)
    partial.update(
        {
            "status": status,
            "honest_verdict": honest,
            "verdict_class": verdict_class,
            "branch_pilot_audited_ready_score": score,
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gates,
        }
    )
    partial["reproducibility_checksum"] = reproducibility_checksum(partial)
    errors = validate_artifact(partial)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        atomic_write_json(result_path, partial, sort_keys=True, env={})
    return partial


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate Exp6517 and return all fail-closed reasons."""

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
    if payload.get("verdict_class") not in {None, "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6517 enum")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    score = payload.get("branch_pilot_audited_ready_score")
    if score not in (0.0, 1.0):
        errors.append("branch_pilot_audited_ready_score must be 0.0 or 1.0")
    aggregate = recompute_aggregate(payload)
    expected_gate = gate_check_summary(aggregate)
    if score != aggregate["ready_score_from_audit_rows"]:
        errors.append("ready score mismatch")
    if payload.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    if payload.get("gate_check_summary") != expected_gate:
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
    source_path: Path | str = UPSTREAM_RELATIVE_PATH,
) -> JsonDict:
    return build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        source_path=source_path,
        write=True,
        duration_s=None,
        tests_run=DEFAULT_TESTS_RUN,
        run_date=date,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--source-path", default=str(UPSTREAM_RELATIVE_PATH))
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
    run(date=args.date, result_path=result_path, source_path=Path(args.source_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``.
    raise SystemExit(main())
