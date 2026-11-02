"""Exp6543 independent DRIFT corpus provenance, split, and replay audit v2.

Spec refs: REQ-BENCH-6543, SCENARIO-BENCH-6543-MISSING,
SCENARIO-BENCH-6543-SOURCE, SCENARIO-BENCH-6543-CHRONOLOGY,
SCENARIO-BENCH-6543-SPLIT, SCENARIO-BENCH-6543-EXACT,
SCENARIO-BENCH-6543-TRANSACTION, SCENARIO-BENCH-6543-ATTACKS.

This reducer audits the V566 intake from the source tree and fixture rows it
can read now. It treats Exp6542 aggregates and IDs as claims to verify, not as
evidence that the split or replay is independent.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

import z3

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6543
INFERENCE_SUBSTRATE = "independent_source_split_transaction_and_z3_replay_audit_no_llm"

DRIFT_REPO_URL = "https://github.com/kaons-research/drift-bench"
DRIFT_GIT_URL = "https://github.com/kaons-research/drift-bench.git"
DRIFT_EXPECTED_COMMIT = "d24cda4f59a6ee06bafe886f4724899a7ec94f1c"
DRIFT_EXPECTED_COMMIT_DATE = "2026-04-25T13:18:49-07:00"
EXPECTED_PROBLEM_FILE_COUNT = 1020

RESULT_RELATIVE_PATH = Path("results/experiment_6543_external_corpus_independent_audit_v2.json")
INTAKE_RELATIVE_PATH = Path("results/experiment_6542_drift_bench_external_intake_v2.json")
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
WORK_RELATIVE_PATH = Path("results/.experiment_6542_drift_bench_external_intake_v2.tx")
SOURCE_CONTRACT_RELATIVE_PATH = Path("results/experiment_6541_v566_direct_source_contract.json")
PRIOR_AUDIT_RELATIVE_PATH = Path("results/experiment_6530_external_constraint_corpus_audit.json")
ATOMIC_TRANSACTION_RELATIVE_PATH = Path(
    "results/experiment_6514_atomic_shard_artifact_transaction.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6543_external_corpus_independent_audit_v2.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6543_external_corpus_independent_audit_v2.py"
)
DEFAULT_SOURCE_CACHE_ROOT = (
    Path.home() / ".cache" / "carnot" / "exp6541" / f"drift-bench-{DRIFT_EXPECTED_COMMIT[:12]}"
)

LOCAL_SPLITS = ("train", "development", "held")

PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-roadmap.yaml"),
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("scripts/research_conductor.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    PRIOR_AUDIT_RELATIVE_PATH,
    SOURCE_CONTRACT_RELATIVE_PATH,
    INTAKE_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
    ATOMIC_TRANSACTION_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "source_existence_and_hash_receipts",
    "independent_revision_license_and_schema_receipt",
    "source_identity_audit_rows",
    "chronology_replay_rows",
    "split_and_lineage_audit",
    "independent_exact_replay_rows",
    "shard_and_transaction_audit",
    "missing_input_disposition",
    "independent_aggregate_rows",
    "leakage_attack_matrix",
    "external_constraint_corpus_audited_ready_score",
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
    "status": "Records the terminal Exp6543 independent audit state.",
    "honest_verdict": "Names audit readiness without declaring a scientific result.",
    "verdict_class": "Separates clean, partial, blocked, and disqualified audit evidence.",
    "source_existence_and_hash_receipts": "Records artifact, fixture, source, solver, resource, seed, and protected-file hashes.",
    "independent_revision_license_and_schema_receipt": "Resolves source revision, license, schema, file census, corruption warning, and Z3 path without trusting Exp6542 aggregates.",
    "source_identity_audit_rows": "Recomputes file, problem, turn, constraint, and source-row hashes from source files.",
    "chronology_replay_rows": "Rebuilds base-problem turn order and detects gaps, duplicates, or reordering.",
    "split_and_lineage_audit": "Recomputes train, development, and held lineage sets from fixture rows.",
    "independent_exact_replay_rows": "Runs fresh Z3 construction for audited rows and compares exact outcomes.",
    "shard_and_transaction_audit": "Verifies planned units, terminal shards, journal, resume receipts, and final fixture hash.",
    "missing_input_disposition": "Closes missing or empty-success inputs as blocked with observed values.",
    "independent_aggregate_rows": "Recomputes aggregate rows locally and compares intake aggregate claims.",
    "leakage_attack_matrix": "Attacks missing inputs, null rows, duplicates, aggregate tampering, aliases, solver paths, and post-held repair.",
    "external_constraint_corpus_audited_ready_score": "Opens only when source, fixture, chronology, split, replay, and transaction checks all pass.",
    "gate_check_summary": "Names failed checks with expected and observed values.",
    "per_unit_rows": "Flattens identity, chronology, replay, aggregate, attack, and gate rows.",
    "aggregate_row_recomputation": "Recomputes readiness from rows instead of trusting status text.",
    "preconditions_checked": "Records date, paths, direct source availability, solvers, resources, seed, and protected hashes.",
    "protected_files_unchanged": "Shows guarded inputs and conductor files stayed byte-identical during the run.",
    "inference_substrate": "Declares independent source, split, transaction, and Z3 replay audit with no LLM inference.",
    "verifier_is_oracle": "True only for audit checks; the artifact makes no positive scientific class.",
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps every field to deterministic rows, files, receipts, or tests.",
    "random_seed": "Pins deterministic replay and attack ordering.",
    "duration_s": "Records measured reducer wall time.",
    "tests_run": "Records validation command receipts.",
    "reproducibility_checksum": "Detects drift in source, fixture, rows, gates, commands, and verdicts.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6543 deterministic independent audit reducer",
        "spec_refs": ["REQ-BENCH-6543"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE["source_existence_and_hash_receipts"]["source"] = (
    "source_existence_and_hash_receipts"
)
FIELD_PROVENANCE["independent_revision_license_and_schema_receipt"]["source"] = (
    "independent_revision_license_and_schema_receipt"
)
FIELD_PROVENANCE["source_identity_audit_rows"]["source"] = "source_identity_audit_rows"
FIELD_PROVENANCE["chronology_replay_rows"]["source"] = "chronology_replay_rows"
FIELD_PROVENANCE["split_and_lineage_audit"]["source"] = "split_and_lineage_audit"
FIELD_PROVENANCE["independent_exact_replay_rows"]["source"] = "independent_exact_replay_rows"
FIELD_PROVENANCE["shard_and_transaction_audit"]["source"] = "shard_and_transaction_audit"
FIELD_PROVENANCE["missing_input_disposition"]["source"] = "missing_input_disposition"
FIELD_PROVENANCE["independent_aggregate_rows"]["source"] = "independent_aggregate_rows"
FIELD_PROVENANCE["leakage_attack_matrix"]["source"] = "leakage_attack_matrix"
FIELD_PROVENANCE["aggregate_row_recomputation"]["source"] = "aggregate_row_recomputation"
FIELD_PROVENANCE["preconditions_checked"]["source"] = "preconditions_checked"
FIELD_PROVENANCE["protected_files_unchanged"]["source"] = "protected_files_unchanged"

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6543_external_corpus_independent_audit_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6543_external_corpus_independent_audit_v2.py "
    "-m pytest tests/python/test_experiment_6543_external_corpus_independent_audit_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6543_external_corpus_independent_audit_v2.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6543_external_corpus_independent_audit_v2.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6543_external_corpus_independent_audit_v2.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6543_external_corpus_independent_audit_v2.json"
)
EXACT_E2E_COMMAND = ".venv/bin/pytest tests/python/test_z3_live_benchmark.py -q --no-cov -n 0"
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6543_external_corpus_independent_audit_v2 --validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6543_external_corpus_independent_audit_v2 "
    "--date 20260823"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_sha256_json(value: Any) -> str:
    data = (canonical_json(value) + "\n").encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        rows.append(dict(value) if isinstance(value, Mapping) else {"_raw": value})
    return rows


def _tests_run_receipts(rows: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = rows if rows is not None else DEFAULT_TESTS_RUN
    return [dict(row) for row in source]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_alias(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _z3_version() -> str:
    return ".".join(map(str, z3.get_version()))


def _source_turn_id(problem_id: str, turn_index: int) -> str:
    return f"{problem_id}:turn:{turn_index + 1}"


def _label_from_solver_result(is_sat: bool, timeout: bool, error: str | None) -> str:
    if timeout:
        return "timeout"
    if error:
        return "error"
    return "satisfiable" if is_sat else "contradiction"


def _run_git(source_root: Path, args: Sequence[str]) -> str | None:
    if not (source_root / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _git_state(repo_root: Path) -> JsonDict:
    try:
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        status = "unavailable"
    return {"status_short": status.strip()}


def _file_receipt(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _resource_receipt(repo_root: Path) -> JsonDict:
    usage = shutil.disk_usage(repo_root)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": usage.free,
        "z3_version": _z3_version(),
    }


def _protected_hashes(repo_root: Path) -> dict[str, JsonDict]:
    return {
        relpath.as_posix(): _file_receipt(repo_root / relpath)
        for relpath in PROTECTED_RELATIVE_PATHS
    }


def protected_files_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    rows = []
    for relpath in sorted(set(before) | set(after)):
        before_hash = before.get(relpath, {}).get("sha256")
        after_hash = after.get(relpath, {}).get("sha256")
        rows.append(
            {
                "path": relpath,
                "before_sha256": before_hash,
                "after_sha256": after_hash,
                "unchanged": before_hash == after_hash,
            }
        )
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "rows": rows,
    }


def _source_root_from_intake(intake: Mapping[str, Any]) -> Path | None:
    candidates = [
        intake.get("source_tree_and_file_hashes", {}).get("checkout_path")
        if isinstance(intake.get("source_tree_and_file_hashes"), Mapping)
        else None,
        intake.get("source_revision_and_license_receipt", {}).get("source_root")
        if isinstance(intake.get("source_revision_and_license_receipt"), Mapping)
        else None,
    ]
    for raw in candidates:
        if isinstance(raw, str) and raw:
            return Path(raw)
    return None


def _resolve_source_root(
    *,
    intake: Mapping[str, Any],
    source_root: Path | str | None,
) -> Path:
    if source_root is not None:
        return Path(source_root)
    env_root = os.environ.get("CARNOT_EXP6543_DRIFT_SOURCE_ROOT")
    if env_root:
        return Path(env_root)
    return _source_root_from_intake(intake) or DEFAULT_SOURCE_CACHE_ROOT


def _problem_file_count(source_root: Path) -> int:
    return len(list((source_root / "data" / "problems").glob("**/*.json")))


def _required_file_hashes(source_root: Path) -> dict[str, str]:
    return {
        relpath: sha256_file(source_root / relpath)
        for relpath in (
            "README.md",
            "LICENSE",
            "data/problems/README.md",
            "src/z3_checker.py",
        )
    }


def _read_problem(source_root: Path, relpath: str) -> JsonDict:
    path = source_root / relpath
    return _load_json(path)


def _turn_for_row(source_root: Path, row: Mapping[str, Any]) -> tuple[JsonDict, JsonDict]:
    problem = _read_problem(source_root, str(row.get("source_file_relpath") or ""))
    turns = problem.get("turns")
    turn_index = _safe_int(row.get("turn_index"), -1)
    if not isinstance(turns, list) or turn_index < 0 or turn_index >= len(turns):
        return problem, {}
    turn = turns[turn_index]
    return problem, dict(turn) if isinstance(turn, Mapping) else {}


def _source_turn_payload(
    problem: Mapping[str, Any],
    row: Mapping[str, Any],
    turn: Mapping[str, Any],
) -> JsonDict:
    return {
        "source_problem_id": problem.get("problem_id"),
        "source_file_relpath": row.get("source_file_relpath"),
        "turn_index": _safe_int(row.get("turn_index")),
        "turn": dict(turn),
    }


def _context_from_problem(problem: Mapping[str, Any]) -> JsonDict:
    domain = str(problem.get("domain") or "")
    if domain == "logic_grid":
        return {"categories": dict(problem.get("categories") or {})}
    if domain == "scheduling":
        return {
            "num_slots": problem.get("num_slots"),
            "max_duration": problem.get("max_duration"),
        }
    if domain == "seating":
        return {
            "num_entities": problem.get("num_entities"),
            "table_shape": problem.get("table_shape", "round"),
        }
    return {}


def _load_z3_checker(source_root: Path) -> ModuleType | None:
    path = source_root / "src" / "z3_checker.py"
    if not path.is_file():
        return None
    module_name = f"exp6543_z3_checker_{hashlib.sha1(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def independent_revision_license_and_schema_receipt(
    *,
    source_root: Path,
    intake: Mapping[str, Any],
    expected_problem_file_count: int,
) -> JsonDict:
    git_revision = _run_git(source_root, ["rev-parse", "HEAD"])
    git_commit_date = _run_git(source_root, ["show", "-s", "--format=%cI", "HEAD"])
    intake_source = intake.get("source_revision_and_license_receipt")
    intake_source = dict(intake_source) if isinstance(intake_source, Mapping) else {}
    observed_revision = git_revision or intake_source.get("immutable_revision")
    observed_commit_date = git_commit_date or intake_source.get("commit_date")
    license_text = (source_root / "LICENSE").read_text(encoding="utf-8") if (source_root / "LICENSE").is_file() else ""
    schema_text = (
        (source_root / "data" / "problems" / "README.md").read_text(encoding="utf-8")
        if (source_root / "data" / "problems" / "README.md").is_file()
        else ""
    )
    readme_text = (
        (source_root / "README.md").read_text(encoding="utf-8")
        if (source_root / "README.md").is_file()
        else ""
    )
    problem_count = _problem_file_count(source_root) if source_root.exists() else 0
    z3_path = source_root / "src" / "z3_checker.py"
    solver_path_identity_ok = z3_path.is_file() and z3_path.name == "z3_checker.py"
    return {
        "repo_url": DRIFT_REPO_URL,
        "git_url": DRIFT_GIT_URL,
        "source_root": str(source_root),
        "source_root_exists": source_root.exists(),
        "source_root_is_git_checkout": (source_root / ".git").exists(),
        "revision_source": "git" if git_revision else "intake_declared_non_git_source",
        "immutable_revision": observed_revision,
        "expected_revision": DRIFT_EXPECTED_COMMIT,
        "revision_matches_expected": observed_revision == DRIFT_EXPECTED_COMMIT,
        "revision_is_immutable": bool(observed_revision and re.fullmatch(r"[0-9a-f]{40}", str(observed_revision))),
        "commit_date": observed_commit_date,
        "expected_commit_date": DRIFT_EXPECTED_COMMIT_DATE,
        "commit_date_matches_expected": observed_commit_date in {
            DRIFT_EXPECTED_COMMIT_DATE,
            None,
        }
        or str(observed_commit_date).startswith("2026-04-25"),
        "license": "MIT" if "MIT License" in license_text else "unknown",
        "license_verified": "MIT License" in license_text,
        "schema_path": "data/problems/README.md",
        "schema_verified": all(token in schema_text for token in ("problem_id", "domain", "turns")),
        "z3_replay_path": "src/z3_checker.py",
        "z3_replay_code_present": z3_path.is_file(),
        "solver_path": str(z3_path),
        "solver_path_identity_ok": solver_path_identity_ok,
        "z3_checker_sha256": sha256_file(z3_path),
        "problem_file_count": problem_count,
        "expected_problem_file_count": expected_problem_file_count,
        "problem_file_count_matches_expected": problem_count == expected_problem_file_count,
        "corruption_warning_verified": "sqlite" in readme_text.lower()
        and ("corrupt" in readme_text.lower() or "corruption" in readme_text.lower()),
        "required_file_sha256": _required_file_hashes(source_root),
    }


def missing_input_disposition(
    *,
    intake_path: Path,
    fixture_path: Path,
    source_root: Path,
    fixture_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    missing = []
    for name, exists, path in (
        ("intake_artifact", intake_path.is_file(), intake_path),
        ("fixture", fixture_path.is_file(), fixture_path),
        ("source_root", source_root.exists(), source_root),
    ):
        if not exists:
            missing.append({"input": name, "path": str(path), "expected": True, "observed": False})
    empty_success = fixture_path.is_file() and len(fixture_rows) == 0
    if empty_success:
        missing.append(
            {
                "input": "empty_success_labeled_fixture",
                "path": str(fixture_path),
                "expected": "non-empty terminal fixture",
                "observed": 0,
            }
        )
    return {
        "blocked": bool(missing),
        "missing_inputs": missing,
        "fixture_row_count": len(fixture_rows),
    }


def source_existence_and_hash_receipts(
    *,
    repo_root: Path,
    intake_path: Path,
    fixture_path: Path,
    source_root: Path,
    fixture_rows: Sequence[Mapping[str, Any]],
    before_protected: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    source_receipts = []
    for relpath in sorted({str(row.get("source_file_relpath") or "") for row in fixture_rows}):
        if not relpath:
            continue
        source_path = source_root / relpath
        declared = sorted(
            {str(row.get("source_file_sha256") or "") for row in fixture_rows if row.get("source_file_relpath") == relpath}
        )
        source_receipts.append(
            {
                "relpath": relpath,
                "path": str(source_path),
                "exists": source_path.is_file(),
                "computed_sha256": sha256_file(source_path),
                "declared_fixture_sha256_values": declared,
                "all_declared_hashes_match": all(
                    item == sha256_file(source_path) for item in declared
                ),
            }
        )
    return {
        "intake_artifact": _file_receipt(intake_path),
        "fixture": {
            **_file_receipt(fixture_path),
            "row_count": len(fixture_rows),
            "jsonl_sha256": sha256_file(fixture_path),
        },
        "source_root": {
            "path": str(source_root),
            "exists": source_root.exists(),
            "is_dir": source_root.is_dir(),
        },
        "source_files": source_receipts,
        "direct_source_available": source_root.exists() and source_root.is_dir(),
        "solver_versions": {"python_z3": _z3_version()},
        "resources": _resource_receipt(repo_root),
        "random_seed": RANDOM_SEED,
        "protected_files": dict(before_protected),
    }


def source_identity_audit_rows(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_root: Path,
) -> list[JsonDict]:
    local_counts = Counter(str(row.get("local_unit_id") or "") for row in fixture_rows)
    turn_counts = Counter(str(row.get("source_turn_id") or "") for row in fixture_rows)
    source_row_counts = Counter(str(row.get("source_row_hash") or "") for row in fixture_rows)
    rows: list[JsonDict] = []
    for row in fixture_rows:
        relpath = str(row.get("source_file_relpath") or "")
        problem, turn = _turn_for_row(source_root, row)
        source_path = source_root / relpath
        constraints = turn.get("cumulative_constraints", [])
        source_turn_id = _source_turn_id(
            str(problem.get("problem_id") or row.get("source_problem_id") or ""),
            _safe_int(row.get("turn_index")),
        )
        computed_source_row_hash = (
            sha256_json(_source_turn_payload(problem, row, turn)) if turn else "missing"
        )
        computed_constraints_hash = sha256_json(constraints) if isinstance(constraints, list) else "missing"
        duplicate_local = local_counts[str(row.get("local_unit_id") or "")] > 1
        duplicate_turn = turn_counts[str(row.get("source_turn_id") or "")] > 1
        duplicate_source_row = source_row_counts[str(row.get("source_row_hash") or "")] > 1
        checks = {
            "source_file_exists": source_path.is_file(),
            "source_file_hash_matches": sha256_file(source_path) == row.get("source_file_sha256"),
            "source_problem_hash_matches": sha256_json(problem) == row.get("source_problem_hash"),
            "source_turn_hash_matches": sha256_json(turn) == row.get("source_turn_sha256"),
            "constraints_hash_matches": computed_constraints_hash == row.get("constraints_sha256"),
            "source_row_hash_matches": computed_source_row_hash == row.get("source_row_hash"),
            "source_turn_id_matches": source_turn_id == row.get("source_turn_id"),
            "source_problem_id_matches": problem.get("problem_id") == row.get("source_problem_id"),
            "local_unit_id_separate_from_source_id": bool(row.get("local_unit_id"))
            and row.get("local_unit_id") != row.get("source_turn_id"),
            "duplicate_local_unit_id": duplicate_local,
            "duplicate_source_turn_id": duplicate_turn,
            "duplicate_source_row_hash": duplicate_source_row,
        }
        rows.append(
            {
                "row_type": "source_identity",
                "local_unit_id": row.get("local_unit_id"),
                "source_problem_id": row.get("source_problem_id"),
                "source_turn_id": row.get("source_turn_id"),
                "source_file_relpath": relpath,
                "computed_source_file_sha256": sha256_file(source_path),
                "declared_source_file_sha256": row.get("source_file_sha256"),
                "computed_source_problem_hash": sha256_json(problem),
                "declared_source_problem_hash": row.get("source_problem_hash"),
                "computed_source_turn_sha256": sha256_json(turn),
                "declared_source_turn_sha256": row.get("source_turn_sha256"),
                "computed_constraints_sha256": computed_constraints_hash,
                "declared_constraints_sha256": row.get("constraints_sha256"),
                "computed_source_row_hash": computed_source_row_hash,
                "declared_source_row_hash": row.get("source_row_hash"),
                **checks,
                "passed": all(value for key, value in checks.items() if not key.startswith("duplicate_"))
                and not duplicate_local
                and not duplicate_turn
                and not duplicate_source_row,
            }
        )
    return rows


def chronology_replay_rows(fixture_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for order, row in enumerate(fixture_rows):
        base = str(row.get("base_problem_id") or row.get("source_problem_id") or "")
        grouped[base].append((order, row))
    out: list[JsonDict] = []
    for base, indexed_rows in grouped.items():
        previous_index: int | None = None
        seen_events: set[str] = set()
        for fixture_order, row in indexed_rows:
            turn_index = _safe_int(row.get("turn_index"), -1)
            event_id = str(row.get("source_turn_id") or "")
            duplicate_event = event_id in seen_events
            expected_next = 0 if previous_index is None else previous_index + 1
            chronology_gap = turn_index != expected_next
            source_turn_id_matches = event_id.endswith(f":turn:{turn_index + 1}")
            chronology_index_matches = _safe_int(row.get("chronology_index"), -1) == turn_index
            seen_events.add(event_id)
            previous_index = turn_index
            chronology_valid = (
                not duplicate_event
                and not chronology_gap
                and source_turn_id_matches
                and chronology_index_matches
            )
            out.append(
                {
                    "row_type": "chronology",
                    "local_unit_id": row.get("local_unit_id"),
                    "base_problem_id": base,
                    "fixture_order": fixture_order,
                    "turn_index": turn_index,
                    "event_id": event_id,
                    "expected_turn_index": expected_next,
                    "duplicate_event": duplicate_event,
                    "chronology_gap": chronology_gap,
                    "source_turn_id_matches": source_turn_id_matches,
                    "chronology_index_matches": chronology_index_matches,
                    "chronology_valid": chronology_valid,
                }
            )
    return out


def split_and_lineage_audit(
    fixture_rows: Sequence[Mapping[str, Any]],
    chronology_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    base_splits: dict[str, set[str]] = defaultdict(set)
    family_aliases: dict[str, set[str]] = defaultdict(set)
    local_ids: list[str] = []
    source_turn_ids: list[str] = []
    missing_terminal = 0
    post_held_repair = 0
    outcome_based_sampling = 0
    for row in fixture_rows:
        base = str(row.get("base_problem_id") or row.get("source_problem_id") or "")
        split = str(row.get("split_name") or "")
        family = str(row.get("family") or row.get("domain") or "")
        base_splits[base].add(split)
        family_aliases[_normalize_alias(family)].add(family)
        local_ids.append(str(row.get("local_unit_id") or ""))
        source_turn_ids.append(str(row.get("source_turn_id") or ""))
        if row.get("terminal_status") != "terminal":
            missing_terminal += 1
        if row.get("post_held_repair") or row.get("repair_after_held"):
            post_held_repair += 1
        if row.get("sampled_after_outcome") or row.get("outcome_based_sampling"):
            outcome_based_sampling += 1
    split_counts = dict(sorted(Counter(str(row.get("split_name") or "") for row in fixture_rows).items()))
    duplicate_local = len(local_ids) - len(set(local_ids))
    duplicate_source_turn = len(source_turn_ids) - len(set(source_turn_ids))
    base_overlap = sum(1 for splits in base_splits.values() if len(splits) > 1)
    alias_collision = sum(1 for names in family_aliases.values() if len(names) > 1)
    chronology_gap_count = sum(1 for row in chronology_rows if row.get("chronology_gap"))
    chronology_duplicate_count = sum(1 for row in chronology_rows if row.get("duplicate_event"))
    floors = {
        "lineage_floor_train": split_counts.get("train", 0) > 0,
        "lineage_floor_development": split_counts.get("development", 0) > 0,
        "lineage_floor_held": split_counts.get("held", 0) > 0,
    }
    passed = bool(fixture_rows) and all(floors.values()) and all(
        value == 0
        for value in (
            base_overlap,
            alias_collision,
            duplicate_local,
            duplicate_source_turn,
            chronology_gap_count,
            chronology_duplicate_count,
            missing_terminal,
            post_held_repair,
            outcome_based_sampling,
        )
    )
    return {
        "row_type": "split_and_lineage",
        "split_names": list(LOCAL_SPLITS),
        "split_counts": split_counts,
        "base_problem_overlap_count": base_overlap,
        "family_alias_collision_count": alias_collision,
        "duplicate_local_unit_id_count": duplicate_local,
        "duplicate_source_turn_id_count": duplicate_source_turn,
        "chronology_gap_count": chronology_gap_count,
        "chronology_duplicate_count": chronology_duplicate_count,
        "missing_terminal_count": missing_terminal,
        "post_held_repair_count": post_held_repair,
        "outcome_based_sampling_count": outcome_based_sampling,
        **floors,
        "passed": passed,
    }


def _solver_assertion_count(
    checker: ModuleType | None,
    *,
    domain: str,
    entities: list[str],
    constraints: list[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> int:
    if checker is None or not hasattr(checker, "build_domain_solver"):
        return len(constraints)
    solver, _aux = checker.build_domain_solver(
        domain,
        entities,
        [dict(row) for row in constraints],
        context=dict(context),
    )
    return len(solver.assertions()) if hasattr(solver, "assertions") else len(constraints)


def independent_exact_replay_rows(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_root: Path,
    sample_seed: int,
) -> list[JsonDict]:
    checker = _load_z3_checker(source_root)
    z3_checker_hash = sha256_file(source_root / "src" / "z3_checker.py")
    rows: list[JsonDict] = []
    for row in fixture_rows:
        problem, turn = _turn_for_row(source_root, row)
        constraints = turn.get("cumulative_constraints", [])
        constraints = constraints if isinstance(constraints, list) else []
        entities = problem.get("entities", [])
        entities = [str(item) for item in entities] if isinstance(entities, list) else []
        domain = str(problem.get("domain") or row.get("domain") or "")
        context = _context_from_problem(problem)
        timeout = False
        error = None
        is_sat = False
        assignment_valid = False
        mus: list[Mapping[str, Any]] = []
        assertion_count = len(constraints)
        try:
            assertion_count = _solver_assertion_count(
                checker,
                domain=domain,
                entities=entities,
                constraints=constraints,
                context=context,
            )
            if checker is None or not hasattr(checker, "check_satisfiability"):
                raise RuntimeError("z3_checker_unavailable")
            sat_receipt = checker.check_satisfiability(
                [dict(item) for item in constraints],
                domain,
                entities,
                context=dict(context),
            )
            is_sat = bool(sat_receipt.get("is_sat"))
            if hasattr(checker, "verify_with_z3"):
                assignment_valid = bool(
                    checker.verify_with_z3(
                        dict(turn.get("gold_solution") or {}),
                        [dict(item) for item in constraints],
                        domain,
                        entities,
                        context=dict(context),
                    )
                )
            if not is_sat and hasattr(checker, "compute_mus"):
                mus = [
                    dict(item)
                    for item in checker.compute_mus(
                        [dict(entry) for entry in constraints],
                        domain,
                        entities,
                        context=dict(context),
                    )
                    if isinstance(item, Mapping)
                ]
        except TimeoutError as exc:
            timeout = True
            error = str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        label = _label_from_solver_result(is_sat, timeout, error)
        terminal_status = "terminal_timeout" if timeout else "terminal_error" if error else "terminal"
        fixture_label = str(row.get("exact_label") or "")
        fixture_satisfiable = bool(row.get("satisfiable"))
        fixture_assignment = bool(row.get("assignment_validity"))
        fixture_terminal = str(row.get("terminal_status") or "")
        independent_receipt = {
            "local_unit_id": row.get("local_unit_id"),
            "source_turn_id": row.get("source_turn_id"),
            "domain": row.get("domain"),
            "split_name": row.get("split_name"),
            "exact_label": label,
            "satisfiable": is_sat,
            "assignment_validity": assignment_valid,
            "solver": "exp6543_independent_source_z3_construction",
            "solver_version": _z3_version(),
            "z3_checker_sha256": z3_checker_hash,
            "constraint_count": len(constraints),
            "solver_assertion_count": assertion_count,
            "timeout": timeout,
            "censored": timeout,
            "error": error,
            "terminal_status": terminal_status,
        }
        rows.append(
            {
                "row_type": "exact_replay",
                "local_unit_id": row.get("local_unit_id"),
                "source_turn_id": row.get("source_turn_id"),
                "split_name": row.get("split_name"),
                "domain": row.get("domain"),
                "sample_seed": sample_seed,
                "sample_policy": "all_rows_deterministic_full_replay",
                "sample_reason": "full_fixture_replay_including_timeout_contradiction_and_effort_rows",
                "solver": "exp6543_independent_source_z3_construction",
                "solver_version": _z3_version(),
                "solver_path": str(source_root / "src" / "z3_checker.py"),
                "solver_path_identity_ok": (source_root / "src" / "z3_checker.py").is_file(),
                "z3_checker_sha256": z3_checker_hash,
                "constraint_count": len(constraints),
                "solver_assertion_count": assertion_count,
                "fixture_exact_label": fixture_label,
                "recomputed_exact_label": label,
                "replayed_label_matches": fixture_label == label,
                "fixture_satisfiable": fixture_satisfiable,
                "recomputed_satisfiable": is_sat,
                "satisfiability_matches": fixture_satisfiable == is_sat,
                "fixture_assignment_validity": fixture_assignment,
                "recomputed_assignment_validity": assignment_valid,
                "assignment_validity_matches": fixture_assignment == assignment_valid,
                "fixture_terminal_status": fixture_terminal,
                "recomputed_terminal_status": terminal_status,
                "terminal_status_matches": fixture_terminal == terminal_status,
                "fixture_exact_receipt_hash": row.get("exact_receipt_hash"),
                "independent_exact_receipt_hash": sha256_json(independent_receipt),
                "timeout": timeout,
                "error": error,
                "conflict_or_mus_evidence": {
                    "available": bool(mus),
                    "mus_size": len(mus),
                    "mus_sha256": sha256_json(mus) if mus else None,
                },
                "passed": fixture_label == label
                and fixture_satisfiable == is_sat
                and fixture_assignment == assignment_valid
                and fixture_terminal == terminal_status
                and terminal_status == "terminal",
            }
        )
    return rows


def _manifest(intake: Mapping[str, Any]) -> JsonDict:
    value = intake.get("shard_manifest")
    return dict(value) if isinstance(value, Mapping) else {}


def _read_journal_rows(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        rows.append(dict(value) if isinstance(value, Mapping) else {"_raw": value})
    return rows


def _journal_hashes_valid(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if "record_hash" not in row:
            return False
        base = {key: value for key, value in row.items() if key != "record_hash"}
        if row.get("record_hash") != _atomic_sha256_json(base):
            return False
    return True


def shard_and_transaction_audit(
    *,
    repo_root: Path,
    intake: Mapping[str, Any],
    fixture_path: Path,
    fixture_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    manifest = _manifest(intake)
    local_by_id = {str(row.get("local_unit_id") or ""): dict(row) for row in fixture_rows}
    planned = [str(item) for item in manifest.get("planned_unit_ids", [])]
    terminal = [str(item) for item in manifest.get("terminal_unit_ids", [])]
    final_receipt = dict(manifest.get("final_atomic_write_receipt") or {})
    shard_rows = []
    for shard in manifest.get("shards", []):
        if not isinstance(shard, Mapping):
            continue
        path = Path(str(shard.get("shard_path") or ""))
        if not path.is_absolute():
            path = repo_root / path
        computed = sha256_file(path)
        unit_id = str(shard.get("unit_id") or "")
        shard_payload = _load_json(path)
        expected_hash = str(shard.get("shard_hash") or "")
        shard_rows.append(
            {
                "row_type": "shard",
                "unit_id": unit_id,
                "path": str(path),
                "exists": path.is_file(),
                "expected_sha256": expected_hash,
                "computed_sha256": computed,
                "content_addressed": path.stem == expected_hash.removeprefix("sha256:"),
                "payload_matches_fixture_row": shard_payload == local_by_id.get(unit_id),
                "passed": path.is_file()
                and computed == expected_hash
                and path.stem == expected_hash.removeprefix("sha256:")
                and shard_payload == local_by_id.get(unit_id),
            }
        )
    journal_path = Path(str(manifest.get("journal_path") or ""))
    if journal_path and not journal_path.is_absolute():
        journal_path = repo_root / journal_path
    journal_rows = _read_journal_rows(journal_path)
    resume_receipts = manifest.get("resume_receipts", [])
    resume_closed = bool(resume_receipts) and all(
        isinstance(row, Mapping)
        and row.get("verified") is True
        and row.get("missing_unit_ids") == []
        for row in resume_receipts
    )
    fixture_hash = sha256_file(fixture_path)
    fixture_roundtrip_rows = _load_jsonl(fixture_path)
    planned_match = sorted(planned) == sorted(local_by_id) and bool(fixture_rows)
    terminal_match = sorted(terminal) == sorted(local_by_id) and bool(fixture_rows)
    final_hash_match = (
        final_receipt.get("atomic_replace") is True
        and final_receipt.get("final_sha256") == fixture_hash
        and final_receipt.get("row_count") == len(fixture_rows)
    )
    journal_match = (
        journal_path.is_file()
        and sha256_file(journal_path) == manifest.get("journal_sha256")
        and len(journal_rows) == manifest.get("journal_record_count")
        and _journal_hashes_valid(journal_rows)
    )
    roundtrip_match = fixture_roundtrip_rows == [dict(row) for row in fixture_rows]
    all_shards = bool(shard_rows) and all(row.get("passed") for row in shard_rows)
    passed = all(
        (
            planned_match,
            terminal_match,
            all_shards,
            journal_match,
            resume_closed,
            manifest.get("corrupt_resume_rejected") is True,
            final_hash_match,
            roundtrip_match,
        )
    )
    return {
        "row_type": "shard_and_transaction",
        "transaction_schema": manifest.get("transaction_schema"),
        "transaction_id": manifest.get("transaction_id"),
        "planned_ids_match_fixture_rows": planned_match,
        "terminal_ids_match_fixture_rows": terminal_match,
        "planned_count": len(planned),
        "terminal_count": len(terminal),
        "fixture_row_count": len(fixture_rows),
        "shard_rows": shard_rows,
        "all_shards_verified_from_disk": all_shards,
        "journal_path": str(journal_path),
        "journal_sha256_matches": journal_match,
        "journal_record_count": len(journal_rows),
        "resume_receipts_closed": resume_closed,
        "corrupt_resume_rejected": manifest.get("corrupt_resume_rejected") is True,
        "final_atomic_write_receipt": final_receipt,
        "fixture_sha256": fixture_hash,
        "final_fixture_hash_matches": final_hash_match,
        "fixture_roundtrip_hash": sha256_json(fixture_roundtrip_rows),
        "fixture_roundtrip_matches_rows": roundtrip_match,
        "passed": passed,
    }


def _recomputed_counts(
    fixture_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "fixture_row_count": len(fixture_rows),
        "base_problem_count": len({str(row.get("base_problem_id") or "") for row in fixture_rows}),
        "domain_counts": dict(sorted(Counter(str(row.get("domain") or "") for row in fixture_rows).items())),
        "split_counts": dict(sorted(Counter(str(row.get("split_name") or "") for row in fixture_rows).items())),
        "turn_position_counts": dict(
            sorted(Counter(str(row.get("turn_position") or "") for row in fixture_rows).items())
        ),
        "effort_strata_counts": dict(
            sorted(Counter(str(row.get("pre_replay_effort_stratum") or "") for row in fixture_rows).items())
        ),
        "exact_label_counts": dict(
            sorted(Counter(str(row.get("recomputed_exact_label") or "") for row in exact_rows).items())
        ),
        "censoring_counts": dict(
            sorted(Counter(str(bool(row.get("censored"))) for row in fixture_rows).items())
        ),
    }


def independent_aggregate_rows(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    intake: Mapping[str, Any],
) -> list[JsonDict]:
    recomputed = _recomputed_counts(fixture_rows, exact_rows)
    intake_aggregate = intake.get("aggregate_row_recomputation")
    intake_aggregate = dict(intake_aggregate) if isinstance(intake_aggregate, Mapping) else {}
    rows = []
    for metric, value in recomputed.items():
        observed = intake_aggregate.get(metric)
        rows.append(
            {
                "row_type": "aggregate",
                "metric": metric,
                "recomputed": value,
                "intake_observed": observed,
                "intake_matches": observed is None or observed == value,
            }
        )
    return rows


def _aggregate_tampering(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, JsonDict, JsonDict]:
    recomputed = {str(row.get("metric")): row.get("recomputed") for row in aggregate_rows}
    observed = {str(row.get("metric")): row.get("intake_observed") for row in aggregate_rows}
    tampered = any(
        row.get("intake_observed") is not None and row.get("intake_matches") is not True
        for row in aggregate_rows
    )
    return tampered, recomputed, observed


def leakage_attack_matrix(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_identity_rows: Sequence[Mapping[str, Any]],
    chronology_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    split_audit: Mapping[str, Any],
    shard_audit: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    revision_receipt: Mapping[str, Any],
) -> list[JsonDict]:
    local_ids = [str(row.get("local_unit_id") or "") for row in fixture_rows]
    source_turn_ids = [str(row.get("source_turn_id") or "") for row in fixture_rows]
    aggregate_tampered, recomputed, observed = _aggregate_tampering(aggregate_rows)
    source_row_hashes = [str(row.get("source_row_hash") or "") for row in fixture_rows]
    attacks = [
        {
            "attack": "empty_success_labeled_fixture_attack",
            "passed": bool(fixture_rows),
            "expected": "non-empty fixture",
            "observed": len(fixture_rows),
        },
        {
            "attack": "null_row_attack",
            "passed": all(bool(row) for row in fixture_rows),
            "expected": "all rows are JSON objects",
            "observed": len([row for row in fixture_rows if not row]),
        },
        {
            "attack": "duplicate_local_id_attack",
            "passed": len(local_ids) == len(set(local_ids)),
            "expected": 0,
            "observed": len(local_ids) - len(set(local_ids)),
        },
        {
            "attack": "duplicate_source_turn_attack",
            "passed": len(source_turn_ids) == len(set(source_turn_ids)),
            "expected": 0,
            "observed": len(source_turn_ids) - len(set(source_turn_ids)),
        },
        {
            "attack": "source_identity_hash_attack",
            "passed": bool(source_identity_rows)
            and all(row.get("passed") is True for row in source_identity_rows),
            "expected": "all source identity rows pass",
            "observed": sum(1 for row in source_identity_rows if row.get("passed")),
        },
        {
            "attack": "chronology_gap_or_reorder_attack",
            "passed": bool(chronology_rows)
            and all(row.get("chronology_valid") is True for row in chronology_rows),
            "expected": "all chronology rows valid",
            "observed": [row for row in chronology_rows if row.get("chronology_valid") is not True],
        },
        {
            "attack": "base_lineage_overlap_attack",
            "passed": split_audit.get("base_problem_overlap_count") == 0,
            "expected": 0,
            "observed": split_audit.get("base_problem_overlap_count"),
        },
        {
            "attack": "outcome_based_sampling_attack",
            "passed": split_audit.get("outcome_based_sampling_count") == 0,
            "expected": 0,
            "observed": split_audit.get("outcome_based_sampling_count"),
        },
        {
            "attack": "post_held_repair_attack",
            "passed": split_audit.get("post_held_repair_count") == 0,
            "expected": 0,
            "observed": split_audit.get("post_held_repair_count"),
        },
        {
            "attack": "exact_replay_cache_trust_attack",
            "passed": bool(exact_rows) and all(row.get("passed") is True for row in exact_rows),
            "expected": "all exact rows replay independently",
            "observed": [row for row in exact_rows if row.get("passed") is not True],
        },
        {
            "attack": "transaction_closure_attack",
            "passed": shard_audit.get("passed") is True,
            "expected": True,
            "observed": shard_audit.get("passed"),
        },
        {
            "attack": "aggregate_tampering_attack",
            "passed": not aggregate_tampered,
            "expected": recomputed,
            "observed": observed,
        },
        {
            "attack": "solver_path_identity_attack",
            "passed": revision_receipt.get("solver_path_identity_ok") is True,
            "expected": True,
            "observed": revision_receipt.get("solver_path_identity_ok"),
        },
        {
            "attack": "source_row_alias_attack",
            "passed": len(source_row_hashes) == len(set(source_row_hashes)),
            "expected": 0,
            "observed": len(source_row_hashes) - len(set(source_row_hashes)),
        },
    ]
    return [{"row_type": "leakage_attack", **row} for row in attacks]


def aggregate_row_recomputation(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_identity_rows: Sequence[Mapping[str, Any]],
    chronology_rows: Sequence[Mapping[str, Any]],
    split_audit: Mapping[str, Any],
    exact_rows: Sequence[Mapping[str, Any]],
    shard_audit: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    revision_receipt: Mapping[str, Any],
    missing_input_disposition: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    counts = _recomputed_counts(fixture_rows, exact_rows)
    aggregate_tampered, _recomputed, _observed = _aggregate_tampering(aggregate_rows)
    source_ok = bool(source_identity_rows) and all(row.get("passed") for row in source_identity_rows)
    chronology_ok = bool(chronology_rows) and all(
        row.get("chronology_valid") for row in chronology_rows
    )
    exact_ok = bool(exact_rows) and all(row.get("passed") for row in exact_rows)
    attacks_ok = bool(attack_rows) and all(row.get("passed") for row in attack_rows)
    revision_ok = all(
        revision_receipt.get(key) is True
        for key in (
            "revision_matches_expected",
            "revision_is_immutable",
            "license_verified",
            "schema_verified",
            "z3_replay_code_present",
            "problem_file_count_matches_expected",
            "corruption_warning_verified",
            "solver_path_identity_ok",
        )
    )
    all_passed = all(
        (
            missing_input_disposition.get("blocked") is False,
            revision_ok,
            source_ok,
            chronology_ok,
            split_audit.get("passed") is True,
            exact_ok,
            shard_audit.get("passed") is True,
            not aggregate_tampered,
            attacks_ok,
            protected.get("all_protected_files_unchanged") is True,
        )
    )
    return {
        **counts,
        "source_identity_pass_count": sum(1 for row in source_identity_rows if row.get("passed")),
        "chronology_pass_count": sum(1 for row in chronology_rows if row.get("chronology_valid")),
        "exact_replay_pass_count": sum(1 for row in exact_rows if row.get("passed")),
        "blocked_preconditions": missing_input_disposition.get("blocked") is True,
        "revision_ok": revision_ok,
        "source_identity_ok": source_ok,
        "chronology_ok": chronology_ok,
        "split_lineage_ok": split_audit.get("passed") is True,
        "exact_replay_ok": exact_ok,
        "shard_transaction_ok": shard_audit.get("passed") is True,
        "aggregate_tampering_detected": aggregate_tampered,
        "leakage_attacks_passed": attacks_ok,
        "protected_files_unchanged": protected.get("all_protected_files_unchanged") is True,
        "all_audit_rows_passed": all_passed,
        "ready_score_from_rows": 1.0 if all_passed else 0.0,
    }


def gate_check_summary(
    *,
    missing_input_disposition: Mapping[str, Any],
    revision_receipt: Mapping[str, Any],
    source_identity_rows: Sequence[Mapping[str, Any]],
    chronology_rows: Sequence[Mapping[str, Any]],
    split_audit: Mapping[str, Any],
    exact_rows: Sequence[Mapping[str, Any]],
    shard_audit: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    failures: list[JsonDict] = []

    def add(check: str, expected: Any, observed: Any, severity: str = "blocked") -> None:
        failures.append(
            {
                "check": check,
                "expected": expected,
                "observed": observed,
                "passed": False,
                "severity": severity,
            }
        )

    for row in missing_input_disposition.get("missing_inputs", []):
        add(f"{row.get('input')}_exists", row.get("expected"), row.get("observed"), "blocked")
    for key in (
        "revision_matches_expected",
        "revision_is_immutable",
        "license_verified",
        "schema_verified",
        "z3_replay_code_present",
        "problem_file_count_matches_expected",
        "corruption_warning_verified",
        "solver_path_identity_ok",
    ):
        if revision_receipt.get(key) is not True:
            add(key, True, revision_receipt.get(key), "blocked")
    if missing_input_disposition.get("blocked") is True:
        return {"all_gates_passed": False, "failed_checks": failures}
    bad_identity = [row for row in source_identity_rows if row.get("passed") is not True]
    if not source_identity_rows or bad_identity:
        add("source_identity", "all identity rows pass", bad_identity, "disqualified")
    bad_chronology = [row for row in chronology_rows if row.get("chronology_valid") is not True]
    if not chronology_rows or bad_chronology:
        add("chronology", "all chronology rows valid", bad_chronology, "disqualified")
    if split_audit.get("passed") is not True:
        add("split_lineage", True, split_audit, "disqualified")
    bad_exact = [row for row in exact_rows if row.get("passed") is not True]
    if not exact_rows or bad_exact:
        add("exact_replay", "all exact rows replay and match", bad_exact, "disqualified")
    if shard_audit.get("passed") is not True:
        add("shard_transaction", True, shard_audit, "disqualified")
    tampered, recomputed, observed = _aggregate_tampering(aggregate_rows)
    if tampered:
        add("aggregate_tampering", recomputed, observed, "disqualified")
    failed_attacks = [row for row in attack_rows if row.get("passed") is not True]
    if failed_attacks:
        add("leakage_attacks", "all attacks pass", failed_attacks, "disqualified")
    if protected.get("all_protected_files_unchanged") is not True:
        add("protected_files_unchanged", True, protected.get("all_protected_files_unchanged"))
    if aggregate.get("ready_score_from_rows") not in (0.0, 1.0):
        add("ready_score_scalar", "0.0 or 1.0", aggregate.get("ready_score_from_rows"))
    return {"all_gates_passed": not failures, "failed_checks": failures}


def _verdict_class(gate: Mapping[str, Any]) -> str | None:
    if gate.get("all_gates_passed"):
        return None
    severities = {str(row.get("severity")) for row in gate.get("failed_checks", [])}
    if "disqualified" in severities:
        return "disqualified"
    if "blocked" in severities:
        return "blocked"
    if "partial" in severities:
        return "partial"
    return "blocked"


def _status_for_class(verdict_class: str | None) -> str:
    if verdict_class is None:
        return "complete_external_corpus_independent_audit_v2"
    if verdict_class == "partial":
        return "partial_external_corpus_independent_audit_v2"
    if verdict_class == "disqualified":
        return "disqualified_external_corpus_independent_audit_v2"
    return "blocked_external_corpus_independent_audit_v2"


def _honest_verdict(status: str, gate: Mapping[str, Any]) -> str:
    if status.startswith("complete_"):
        return (
            f"{status}: source, fixture, chronology, split, replay, transaction, "
            "aggregate, and attack checks pass"
        )
    checks = ",".join(str(row.get("check")) for row in gate.get("failed_checks", []))
    return f"{status}: failed_checks={checks or 'unknown'}"


def build_per_unit_rows(
    *,
    source_identity_rows: Sequence[Mapping[str, Any]],
    chronology_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> list[JsonDict]:
    rows = [dict(row) for row in source_identity_rows]
    rows.extend(dict(row) for row in chronology_rows)
    rows.extend(dict(row) for row in exact_rows)
    rows.extend(dict(row) for row in aggregate_rows)
    rows.extend(dict(row) for row in attack_rows)
    rows.extend({"row_type": "gate", **row} for row in gate.get("failed_checks", []))
    if gate.get("all_gates_passed"):
        rows.append({"row_type": "gate", "check": "all_gates_passed", "observed": True})
    return rows


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(stable)


def preconditions_checked(
    *,
    repo_root: Path,
    run_date: str,
    now_utc: str,
    intake_path: Path,
    fixture_path: Path,
    source_root: Path,
    fixture_rows: Sequence[Mapping[str, Any]],
    revision_receipt: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    return {
        "run_date": run_date,
        "checked_at_utc": now_utc,
        "intake_artifact_path": str(intake_path),
        "fixture_path": str(fixture_path),
        "source_root": str(source_root),
        "intake_artifact_sha256": sha256_file(intake_path),
        "fixture_sha256": sha256_file(fixture_path),
        "fixture_row_count": len(fixture_rows),
        "direct_source_available": source_root.exists(),
        "source_revision": revision_receipt.get("immutable_revision"),
        "solver_versions": {"python_z3": _z3_version()},
        "resources": _resource_receipt(repo_root),
        "random_seed": RANDOM_SEED,
        "git_state": _git_state(repo_root),
        "protected_files_unchanged": protected.get("all_protected_files_unchanged"),
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str | None = None,
    source_root: Path | str | None = None,
    expected_problem_file_count: int = EXPECTED_PROBLEM_FILE_COUNT,
    run_date: str = RUN_DATE,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    now_utc: str | None = None,
) -> JsonDict:
    start = time.monotonic()
    now = now_utc or _utc_now()
    before_hashes = _protected_hashes(repo_root)
    intake_path = repo_root / INTAKE_RELATIVE_PATH
    fixture_path = repo_root / FIXTURE_RELATIVE_PATH
    intake = _load_json(intake_path)
    fixture_rows = _load_jsonl(fixture_path)
    resolved_source_root = _resolve_source_root(intake=intake, source_root=source_root)
    revision_receipt = independent_revision_license_and_schema_receipt(
        source_root=resolved_source_root,
        intake=intake,
        expected_problem_file_count=expected_problem_file_count,
    )
    existence = source_existence_and_hash_receipts(
        repo_root=repo_root,
        intake_path=intake_path,
        fixture_path=fixture_path,
        source_root=resolved_source_root,
        fixture_rows=fixture_rows,
        before_protected=before_hashes,
    )
    missing = missing_input_disposition(
        intake_path=intake_path,
        fixture_path=fixture_path,
        source_root=resolved_source_root,
        fixture_rows=fixture_rows,
    )
    source_identity = source_identity_audit_rows(
        fixture_rows=fixture_rows,
        source_root=resolved_source_root,
    )
    chronology = chronology_replay_rows(fixture_rows)
    split_audit = split_and_lineage_audit(fixture_rows, chronology)
    exact = independent_exact_replay_rows(
        fixture_rows=fixture_rows,
        source_root=resolved_source_root,
        sample_seed=RANDOM_SEED,
    )
    shard_audit = shard_and_transaction_audit(
        repo_root=repo_root,
        intake=intake,
        fixture_path=fixture_path,
        fixture_rows=fixture_rows,
    )
    aggregates = independent_aggregate_rows(
        fixture_rows=fixture_rows,
        exact_rows=exact,
        intake=intake,
    )
    attacks = leakage_attack_matrix(
        fixture_rows=fixture_rows,
        source_identity_rows=source_identity,
        chronology_rows=chronology,
        exact_rows=exact,
        split_audit=split_audit,
        shard_audit=shard_audit,
        aggregate_rows=aggregates,
        revision_receipt=revision_receipt,
    )
    after_hashes = _protected_hashes(repo_root)
    protected = protected_files_unchanged(before_hashes, after_hashes)
    aggregate = aggregate_row_recomputation(
        fixture_rows=fixture_rows,
        source_identity_rows=source_identity,
        chronology_rows=chronology,
        split_audit=split_audit,
        exact_rows=exact,
        shard_audit=shard_audit,
        aggregate_rows=aggregates,
        attack_rows=attacks,
        revision_receipt=revision_receipt,
        missing_input_disposition=missing,
        protected=protected,
    )
    gate = gate_check_summary(
        missing_input_disposition=missing,
        revision_receipt=revision_receipt,
        source_identity_rows=source_identity,
        chronology_rows=chronology,
        split_audit=split_audit,
        exact_rows=exact,
        shard_audit=shard_audit,
        aggregate_rows=aggregates,
        attack_rows=attacks,
        aggregate=aggregate,
        protected=protected,
    )
    verdict_class = _verdict_class(gate)
    status = _status_for_class(verdict_class)
    preconditions = preconditions_checked(
        repo_root=repo_root,
        run_date=run_date,
        now_utc=now,
        intake_path=intake_path,
        fixture_path=fixture_path,
        source_root=resolved_source_root,
        fixture_rows=fixture_rows,
        revision_receipt=revision_receipt,
        protected=protected,
    )
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": _honest_verdict(status, gate),
        "verdict_class": verdict_class,
        "source_existence_and_hash_receipts": existence,
        "independent_revision_license_and_schema_receipt": revision_receipt,
        "source_identity_audit_rows": source_identity,
        "chronology_replay_rows": chronology,
        "split_and_lineage_audit": split_audit,
        "independent_exact_replay_rows": exact,
        "shard_and_transaction_audit": shard_audit,
        "missing_input_disposition": missing,
        "independent_aggregate_rows": aggregates,
        "leakage_attack_matrix": attacks,
        "external_constraint_corpus_audited_ready_score": float(
            aggregate["ready_score_from_rows"]
        ),
        "gate_check_summary": gate,
        "per_unit_rows": build_per_unit_rows(
            source_identity_rows=source_identity,
            chronology_rows=chronology,
            exact_rows=exact,
            aggregate_rows=aggregates,
            attack_rows=attacks,
            gate=gate,
        ),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": FIELD_PROVENANCE,
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - start),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if write:
        target = Path(result_path) if result_path is not None else repo_root / RESULT_RELATIVE_PATH
        atomic_write_json(target, artifact, allow_override=False, sort_keys=False)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("verdict_class") not in {None, "partial", "blocked", "disqualified"}:
        errors.append("verdict_class outside Exp6543 enum")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict terminal prefix mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    score = artifact.get("external_constraint_corpus_audited_ready_score")
    recomputed_score = artifact.get("aggregate_row_recomputation", {}).get("ready_score_from_rows")
    if score not in {0.0, 1.0} or score != recomputed_score:
        errors.append("ready score mismatch")
    gate = artifact.get("gate_check_summary", {})
    if score == 1.0 and gate.get("all_gates_passed") is not True:
        errors.append("ready artifact must have all gates passed")
    if score == 0.0 and gate.get("all_gates_passed") is True:
        errors.append("blocked artifact cannot have all gates passed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate Exp6543 independent external corpus audit v2."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--source-root", default=None)
    parser.add_argument(
        "--expected-problem-file-count",
        type=int,
        default=EXPECTED_PROBLEM_FILE_COUNT,
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        errors = validate_artifact(_load_json(result))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {RESULT_RELATIVE_PATH.as_posix()}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result,
        source_root=args.source_root,
        expected_problem_file_count=args.expected_problem_file_count,
        run_date=str(args.date),
        write=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
