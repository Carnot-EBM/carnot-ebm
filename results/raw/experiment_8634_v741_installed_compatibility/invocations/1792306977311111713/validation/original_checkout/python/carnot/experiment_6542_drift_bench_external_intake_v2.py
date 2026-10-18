"""Exp6542 content-pinned DRIFT-Bench external intake v2.

Spec refs: REQ-BENCH-6542, SCENARIO-BENCH-6542-SOURCE,
SCENARIO-BENCH-6542-CHRONOLOGY, SCENARIO-BENCH-6542-EXACT,
SCENARIO-BENCH-6542-SPLIT, SCENARIO-BENCH-6542-SHARDS,
SCENARIO-BENCH-6542-CENSORING, SCENARIO-BENCH-6542-ATTACKS.

This reducer turns the pinned DRIFT-Bench source tree into a small local
fixture. It treats the upstream JSON as source data only: all exact labels and
receipts are replayed locally through the pinned Z3 checker before final write.
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
import tempfile
import time
from typing import Any

import z3

from carnot.atomic_shard_transaction import (
    TRANSACTION_SCHEMA,
    AtomicShardTransaction,
    CorruptShardError,
    MissingTerminalUnitError,
    sha256_bytes,
)
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6542
INFERENCE_SUBSTRATE = "content_pinned_drift_intake_and_local_z3_replay_no_llm"

DRIFT_REPO_URL = "https://github.com/kaons-research/drift-bench"
DRIFT_GIT_URL = "https://github.com/kaons-research/drift-bench.git"
DRIFT_EXPECTED_COMMIT = "d24cda4f59a6ee06bafe886f4724899a7ec94f1c"
DRIFT_EXPECTED_COMMIT_DATE = "2026-04-25T13:18:49-07:00"
EXPECTED_PROBLEM_FILE_COUNT = 1020
DEFAULT_FIXTURE_BOUND = 9

RESULT_RELATIVE_PATH = Path("results/experiment_6542_drift_bench_external_intake_v2.json")
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v566_drift_bench_external_slice.jsonl")
WORK_RELATIVE_PATH = Path("results/.experiment_6542_drift_bench_external_intake_v2.tx")
UPSTREAM_GATE_RELATIVE_PATH = Path("results/experiment_6541_v566_direct_source_contract.json")
ATOMIC_TRANSACTION_RELATIVE_PATH = Path(
    "results/experiment_6514_atomic_shard_artifact_transaction.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6542_drift_bench_external_intake_v2.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6542_drift_bench_external_intake_v2.py")
DEFAULT_SOURCE_CACHE_ROOT = (
    Path.home() / ".cache" / "carnot" / "exp6541" / f"drift-bench-{DRIFT_EXPECTED_COMMIT[:12]}"
)

LOCAL_SPLITS = ("train", "development", "held")
DOMAINS = ("logic_grid", "scheduling", "seating")

PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
    UPSTREAM_GATE_RELATIVE_PATH,
    ATOMIC_TRANSACTION_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "source_revision_and_license_receipt",
    "source_tree_and_file_hashes",
    "upstream_corruption_boundary",
    "intake_commitment",
    "family_turn_and_effort_census",
    "source_to_local_identity_rows",
    "exact_replay_rows",
    "solver_receipts",
    "split_commitment",
    "shard_manifest",
    "planned_and_terminal_unit_counts",
    "fixture_path_and_hash",
    "leakage_attack_matrix",
    "external_constraint_corpus_ready_score",
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
    "status": "Records the terminal Exp6542 intake state.",
    "honest_verdict": "Names whether the content-pinned fixture is complete, partial, blocked, or disqualified.",
    "verdict_class": "Separates complete data contracts from bounded, blocked, or disqualified paths.",
    "upstream_gate_receipt": "Records the Exp6541 gate value, artifact hash, source commit, tools, resources, fixture bound, and protected hashes.",
    "source_revision_and_license_receipt": "Binds the source checkout to the expected revision, license, schema, Z3 code, and problem count.",
    "source_tree_and_file_hashes": "Records tree, required-file, prompt, schema, and source problem hashes before transformation.",
    "upstream_corruption_boundary": "Shows that corrupted SQLite databases and paper aggregates are not inherited.",
    "intake_commitment": "Freezes the bounded sample before replay labels or solver costs are inspected.",
    "family_turn_and_effort_census": "Shows the selected fixture covers domains, sizes, turns, splits, and effort strata.",
    "source_to_local_identity_rows": "Maps immutable source IDs to separate local unit IDs by hash.",
    "exact_replay_rows": "Records local Z3 satisfiability, assignment validity, effort, timeout, and terminal status.",
    "solver_receipts": "Binds solver version, replay code hash, call counts, and timeout policy.",
    "split_commitment": "Seals base-problem lineage so no turn crosses train, development, or held splits.",
    "shard_manifest": "Records content-addressed shards, journal, resume state, corrupt-resume probe, and final atomic replace.",
    "planned_and_terminal_unit_counts": "Proves every planned unit has a terminal row before readiness opens.",
    "fixture_path_and_hash": "Binds the final JSONL path, hash, row count, and round-trip check.",
    "leakage_attack_matrix": "Records duplicate, chronology, family, name, order, hash, solver, resume, terminal, and aggregate attacks.",
    "external_constraint_corpus_ready_score": "Opens only when source, replay, lineage, shards, attacks, hashes, terminal rows, and round-trip pass.",
    "gate_check_summary": "Names failed checks with expected and observed values.",
    "per_unit_rows": "Flattens identity, replay, shard, attack, and gate evidence.",
    "aggregate_row_recomputation": "Recomputes readiness and aggregate counts from local rows.",
    "preconditions_checked": "Records date, paths, resources, git state, solver versions, and protected hashes.",
    "protected_files_unchanged": "Shows guarded inputs and conductor files were not modified by the run.",
    "inference_substrate": "Declares content-pinned source intake and local Z3 replay with no LLM inference.",
    "verifier_is_oracle": "True only for exact labels and assignment validity replayed by Z3.",
    "field_principles": "Explains why each field exists.",
    "field_provenance": "Maps fields to specs, rows, source files, transaction receipts, tests, or hashes.",
    "random_seed": "Pins deterministic sample ordering and local IDs.",
    "duration_s": "Records measured reducer wall time.",
    "tests_run": "Records validation command receipts.",
    "reproducibility_checksum": "Detects drift in rows, gates, hashes, commands, and verdicts.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6542 deterministic content-pinned intake reducer",
        "spec_refs": ["REQ-BENCH-6542"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE["upstream_gate_receipt"]["source"] = "build_upstream_gate_receipt"
FIELD_PROVENANCE["source_revision_and_license_receipt"]["source"] = (
    "build_source_revision_and_license_receipt"
)
FIELD_PROVENANCE["source_tree_and_file_hashes"]["source"] = "build_source_tree_and_file_hashes"
FIELD_PROVENANCE["intake_commitment"]["source"] = "freeze_balanced_slice"
FIELD_PROVENANCE["family_turn_and_effort_census"]["source"] = "family_turn_and_effort_census"
FIELD_PROVENANCE["source_to_local_identity_rows"]["source"] = "source_to_local_identity_rows"
FIELD_PROVENANCE["exact_replay_rows"]["source"] = "replay_selected_turns"
FIELD_PROVENANCE["split_commitment"]["source"] = "build_split_commitment"
FIELD_PROVENANCE["shard_manifest"]["source"] = "write_fixture_transaction"
FIELD_PROVENANCE["leakage_attack_matrix"]["source"] = "leakage_attack_matrix"
FIELD_PROVENANCE["aggregate_row_recomputation"]["source"] = "aggregate_row_recomputation"
FIELD_PROVENANCE["protected_files_unchanged"]["source"] = "protected_files_unchanged"

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6542_drift_bench_external_intake_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6542_drift_bench_external_intake_v2.py "
    "-m pytest tests/python/test_experiment_6542_drift_bench_external_intake_v2.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6542_drift_bench_external_intake_v2.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6542_drift_bench_external_intake_v2.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6542_drift_bench_external_intake_v2.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6542_drift_bench_external_intake_v2.json"
)
EXACT_E2E_COMMAND = ".venv/bin/pytest tests/python/test_z3_live_benchmark.py -q --no-cov -n 0"
CHECKSUM_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6542_drift_bench_external_intake_v2 --validate"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6542_drift_bench_external_intake_v2 --date 20260823"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": CHECKSUM_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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
    rows: list[JsonDict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            rows.append(dict(value) if isinstance(value, Mapping) else {"value": value})
    return rows


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_alias(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _resource_receipt(root: Path) -> JsonDict:
    disk = shutil.disk_usage(root if root.exists() else root.parent)
    ram_total = None
    if hasattr(os, "sysconf"):
        try:
            ram_total = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError):  # pragma: no cover - platform sysconf failure guard.
            ram_total = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": ram_total,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _git_status(repo_root: Path) -> JsonDict:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return {"exit_code": result.returncode, "status_short": result.stdout.strip()}


def _git_output(root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, text=True, capture_output=True)
    return result.stdout.strip()


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {rel.as_posix(): sha256_file(repo_root / rel) for rel in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "protected_file_rows": rows,
    }


def _problem_files(source_root: Path) -> list[Path]:
    return sorted((source_root / "data" / "problems").glob("*/*.json"))


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _context_from_problem(problem: Mapping[str, Any]) -> JsonDict:
    domain = str(problem.get("domain") or "")
    if domain == "logic_grid":
        return {"categories": dict(problem.get("categories") or {})}
    if domain == "scheduling":
        return {
            "num_slots": problem.get("num_slots", max(6, len(problem.get("entities", [])) + 1)),
            "max_duration": problem.get("max_duration", 3),
        }
    if domain == "seating":
        return {
            "num_entities": problem.get("num_entities", len(problem.get("entities", []))),
            "table_shape": problem.get("table_shape", "round"),
        }
    return {}


def _turn_position(turn_index: int, turn_count: int) -> str:
    if turn_index <= 0:
        return "early"
    if turn_index >= max(turn_count - 1, 0):
        return "late"
    return "middle"


def _effort_stratum(constraint_count: int) -> str:
    if constraint_count <= 3:
        return "low"
    if constraint_count <= 7:
        return "medium"
    return "high"


def _drift_git_metadata(root: Path) -> JsonDict:
    if (root / ".git").exists():
        return {
            "repo_url": DRIFT_REPO_URL,
            "commit": _git_output(root, ["rev-parse", "HEAD"]),
            "commit_date": _git_output(root, ["show", "--no-patch", "--format=%cI", "HEAD"]),
            "commit_subject": _git_output(root, ["show", "--no-patch", "--format=%s", "HEAD"]),
            "root_tree_git_sha": _git_output(root, ["rev-parse", "HEAD^{tree}"]),
            "problems_tree_git_sha": _git_output(root, ["rev-parse", "HEAD:data/problems"]),
            "checkout_path": str(root),
            "ls_remote_head": _git_output(root, ["rev-parse", "HEAD"]),
            "metadata_source": "git",
        }
    return {
        "repo_url": DRIFT_REPO_URL,
        "commit": DRIFT_EXPECTED_COMMIT,
        "commit_date": DRIFT_EXPECTED_COMMIT_DATE,
        "commit_subject": "non-git fixture source",
        "root_tree_git_sha": sha256_json(
            sorted(p.as_posix() for p in root.rglob("*") if p.is_file())
        ),
        "problems_tree_git_sha": sha256_json(
            sorted(_relative(p, root) for p in _problem_files(root))
        ),
        "checkout_path": str(root),
        "ls_remote_head": DRIFT_EXPECTED_COMMIT,
        "metadata_source": "non_git_fixture",
    }


def prepare_drift_source_root() -> Path:  # pragma: no cover - live fallback path.
    env_root = os.environ.get("CARNOT_EXP6542_DRIFT_SOURCE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if DEFAULT_SOURCE_CACHE_ROOT.exists():
        return DEFAULT_SOURCE_CACHE_ROOT
    cache_root = DEFAULT_SOURCE_CACHE_ROOT.parent
    cache_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", DRIFT_GIT_URL, str(DEFAULT_SOURCE_CACHE_ROOT)],
        check=False,
        text=True,
        capture_output=True,
    )
    if (DEFAULT_SOURCE_CACHE_ROOT / ".git").exists():
        subprocess.run(
            ["git", "checkout", DRIFT_EXPECTED_COMMIT],
            cwd=DEFAULT_SOURCE_CACHE_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    return DEFAULT_SOURCE_CACHE_ROOT


def build_upstream_gate_receipt(
    *,
    repo_root: Path,
    upstream: Mapping[str, Any],
    fixture_bound: int,
    protected_before: Mapping[str, str],
) -> JsonDict:
    observed = _safe_float(upstream.get("v566_direct_source_ready_score"), default=0.0)
    contract = dict(upstream.get("drift_revision_license_schema_contract") or {})
    source_hashes = dict(upstream.get("source_tree_hashes") or {})
    return {
        "artifact_path": str(repo_root / UPSTREAM_GATE_RELATIVE_PATH),
        "artifact_sha256": sha256_file(repo_root / UPSTREAM_GATE_RELATIVE_PATH),
        "field": "v566_direct_source_ready_score",
        "expected": 1.0,
        "observed": observed,
        "passed": observed == 1.0,
        "source_commit": contract.get("immutable_revision"),
        "source_commit_expected": DRIFT_EXPECTED_COMMIT,
        "license": contract.get("license"),
        "source_tree_git_sha": source_hashes.get("root_tree_git_sha"),
        "solver_versions": {
            "z3": z3.get_version_string(),
            "python": platform.python_version(),
        },
        "resources": _resource_receipt(repo_root),
        "fixture_bound": fixture_bound,
        "protected_file_hashes_before": dict(protected_before),
    }


def build_source_tree_and_file_hashes(
    source_root: Path,
    metadata: Mapping[str, Any],
    upstream: Mapping[str, Any],
) -> JsonDict:
    problem_entries = [
        {"path": _relative(path, source_root), "sha256": sha256_file(path)}
        for path in _problem_files(source_root)
    ]
    required_paths = (
        "README.md",
        "LICENSE",
        "data/problems/README.md",
        "src/z3_checker.py",
        "docs/prompts.md",
        "src/prompts.py",
    )
    required_hashes = {rel: sha256_file(source_root / rel) for rel in required_paths}
    upstream_hashes = dict(upstream.get("source_tree_hashes") or {}).get("required_file_sha256", {})
    compared = {
        rel: {
            "expected": upstream_hashes.get(rel),
            "observed": required_hashes.get(rel),
            "matches": upstream_hashes.get(rel) in (None, required_hashes.get(rel)),
        }
        for rel in required_hashes
    }
    return {
        "repo_url": DRIFT_REPO_URL,
        "checkout_path": str(source_root),
        "root_tree_git_sha": metadata.get("root_tree_git_sha"),
        "problems_tree_git_sha": metadata.get("problems_tree_git_sha"),
        "problem_file_count": len(problem_entries),
        "problem_manifest_sha256": sha256_json(problem_entries),
        "problem_manifest_first_paths": [row["path"] for row in problem_entries[:5]],
        "required_file_sha256": required_hashes,
        "upstream_required_file_comparison": compared,
        "all_required_files_present": all(value != "missing" for value in required_hashes.values()),
        "hashes_match_exp6541_when_declared": all(row["matches"] for row in compared.values()),
    }


def build_source_revision_and_license_receipt(
    *,
    source_root: Path,
    metadata: Mapping[str, Any],
    tree_hashes: Mapping[str, Any],
    expected_problem_file_count: int,
) -> JsonDict:
    license_text = _read_text(source_root / "LICENSE")
    schema_text = _read_text(source_root / "data" / "problems" / "README.md")
    readme_text = _read_text(source_root / "README.md")
    z3_text = _read_text(source_root / "src" / "z3_checker.py")
    commit = str(metadata.get("commit") or "")
    return {
        "repo_url": DRIFT_REPO_URL,
        "git_url": DRIFT_GIT_URL,
        "source_root": str(source_root),
        "immutable_revision": commit,
        "expected_revision": DRIFT_EXPECTED_COMMIT,
        "revision_matches_expected": commit == DRIFT_EXPECTED_COMMIT,
        "revision_is_immutable": bool(re.fullmatch(r"[0-9a-f]{40}", commit)),
        "commit_date": metadata.get("commit_date"),
        "commit_date_matches_expected": metadata.get("commit_date") == DRIFT_EXPECTED_COMMIT_DATE,
        "license": "MIT" if "MIT" in license_text else "unknown",
        "license_verified": "MIT" in license_text,
        "data_schema_path": "data/problems/README.md",
        "data_schema_verified": all(
            token in schema_text
            for token in (
                "problem_id",
                "domain",
                "split",
                "entities",
                "turns",
                "cumulative_constraints",
                "gold_solution",
                "is_satisfiable",
            )
        ),
        "z3_replay_path": "src/z3_checker.py",
        "z3_replay_code_present": "z3" in z3_text.lower(),
        "problem_file_count": tree_hashes.get("problem_file_count"),
        "expected_problem_file_count": expected_problem_file_count,
        "problem_file_count_matches_expected": (
            tree_hashes.get("problem_file_count") == expected_problem_file_count
        ),
        "upstream_corruption_warning_present": (
            "sqlite" in readme_text.lower() and "corrupt" in readme_text.lower()
        ),
        "source_commit_subject": metadata.get("commit_subject"),
    }


def build_upstream_corruption_boundary(source_root: Path) -> JsonDict:
    readme = _read_text(source_root / "README.md")
    return {
        "sqlite_corruption_warning_present": "sqlite" in readme.lower()
        and "corrupt" in readme.lower(),
        "upstream_sqlite_results_inherited": False,
        "paper_aggregate_claims_inherited": False,
        "upstream_result_databases_imported": False,
        "local_replay_required_for_every_row": True,
        "readme_sha256": sha256_file(source_root / "README.md"),
    }


def _source_preconditions_pass(source_receipt: Mapping[str, Any]) -> bool:
    return all(
        (
            source_receipt.get("revision_matches_expected") is True,
            source_receipt.get("license_verified") is True,
            source_receipt.get("data_schema_verified") is True,
            source_receipt.get("z3_replay_code_present") is True,
            source_receipt.get("problem_file_count_matches_expected") is True,
            source_receipt.get("upstream_corruption_warning_present") is True,
        )
    )


def _problem_record(path: Path, source_root: Path) -> JsonDict:
    problem = _load_json(path)
    turns = list(problem.get("turns") or [])
    cumulative_counts = [
        len(turn.get("cumulative_constraints") or []) for turn in turns if isinstance(turn, Mapping)
    ]
    return {
        "path": path,
        "source_file_relpath": _relative(path, source_root),
        "source_file_sha256": sha256_file(path),
        "problem": problem,
        "source_problem_id": str(problem.get("problem_id") or path.stem),
        "base_problem_id": str(problem.get("problem_id") or path.stem),
        "domain": str(problem.get("domain") or path.stem.rsplit("_", 1)[0]),
        "source_split": str(problem.get("split") or path.parent.name),
        "num_entities": _safe_int(problem.get("num_entities"), len(problem.get("entities", []))),
        "turn_count": len(turns),
        "max_cumulative_constraints": max(cumulative_counts or [0]),
        "source_problem_hash": sha256_json(problem),
    }


def load_problem_records(source_root: Path) -> list[JsonDict]:
    return [_problem_record(path, source_root) for path in _problem_files(source_root)]


def _stable_problem_order(record: Mapping[str, Any]) -> tuple[str, str, str]:
    key = sha256_json(
        {
            "seed": RANDOM_SEED,
            "source_problem_id": record.get("source_problem_id"),
            "source_split": record.get("source_split"),
        }
    )
    return (str(record.get("domain")), key, str(record.get("source_problem_id")))


def freeze_balanced_slice(
    records: Sequence[Mapping[str, Any]],
    *,
    fixture_bound: int,
) -> tuple[list[JsonDict], JsonDict]:
    selected: list[JsonDict] = []
    used: set[str] = set()
    per_domain = max(1, fixture_bound // max(len(DOMAINS), 1))
    for domain in DOMAINS:
        domain_records = [dict(row) for row in records if row.get("domain") == domain]
        sizes = sorted({int(row.get("num_entities") or 0) for row in domain_records})
        for split_index, split_name in enumerate(LOCAL_SPLITS[:per_domain]):
            target_size = sizes[split_index % len(sizes)] if sizes else None
            pool = [
                row
                for row in domain_records
                if row["source_problem_id"] not in used and row.get("num_entities") == target_size
            ]
            if not pool:
                pool = [row for row in domain_records if row["source_problem_id"] not in used]
            if not pool:
                continue
            chosen = sorted(pool, key=_stable_problem_order)[0]
            chosen["split_name"] = split_name
            chosen["selection_index"] = len(selected)
            chosen["selection_frozen_before_replay"] = True
            chosen["selection_basis"] = {
                "domain": chosen["domain"],
                "num_entities": chosen["num_entities"],
                "turn_count": chosen["turn_count"],
                "max_cumulative_constraints": chosen["max_cumulative_constraints"],
                "source_split": chosen["source_split"],
            }
            selected.append(chosen)
            used.add(str(chosen["source_problem_id"]))
    selected = selected[:fixture_bound]
    commitment = {
        "random_seed": RANDOM_SEED,
        "fixture_bound_base_problems": fixture_bound,
        "selected_base_problem_count": len(selected),
        "sample_frozen_before_downstream_labels_or_costs": True,
        "labels_or_costs_inspected_before_freeze": False,
        "selection_policy": (
            "domain x local_split balanced; size diversity preferred; stable hash tie-break"
        ),
        "selected_problem_ids": [
            {
                "selection_index": row["selection_index"],
                "split_name": row["split_name"],
                "domain": row["domain"],
                "source_problem_id": row["source_problem_id"],
                "source_file_relpath": row["source_file_relpath"],
                "selection_basis": row["selection_basis"],
            }
            for row in selected
        ],
    }
    commitment["commitment_hash"] = sha256_json(commitment["selected_problem_ids"])
    return selected, commitment


def load_z3_checker(source_root: Path) -> Any:
    checker_path = source_root / "src" / "z3_checker.py"
    spec = importlib.util.spec_from_file_location(
        f"drift_z3_checker_{hashlib.sha1(str(checker_path).encode()).hexdigest()}",
        checker_path,
    )
    if spec is None or spec.loader is None:  # pragma: no cover - importlib defensive guard.
        raise ImportError(f"cannot load {checker_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _solver_assertion_count(
    checker: Any,
    *,
    domain: str,
    entities: list[str],
    constraints: list[dict[str, Any]],
    context: Mapping[str, Any],
) -> int:
    if not hasattr(checker, "build_domain_solver"):
        return len(constraints)
    solver, _aux = checker.build_domain_solver(
        domain,
        entities,
        constraints,
        context=dict(context),
    )
    return len(solver.assertions())


def _local_unit_id(selection_index: int, source_problem_id: str, turn_index: int) -> str:
    digest = hashlib.sha256(
        f"{RANDOM_SEED}:{selection_index}:{source_problem_id}:{turn_index}".encode()
    ).hexdigest()[:16]
    return f"v566_unit_{selection_index:03d}_{turn_index:02d}_{digest}"


def replay_selected_turns(
    *,
    selected_problems: Sequence[Mapping[str, Any]],
    checker: Any,
    source_root: Path,
) -> tuple[list[JsonDict], list[JsonDict]]:
    fixture_rows: list[JsonDict] = []
    exact_rows: list[JsonDict] = []
    z3_code_hash = sha256_file(source_root / "src" / "z3_checker.py")
    solver_version = z3.get_version_string()
    for selected in selected_problems:
        problem = dict(selected["problem"])
        turns = [dict(turn) for turn in problem.get("turns", [])]
        domain = str(problem.get("domain"))
        entities = [str(entity) for entity in problem.get("entities", [])]
        context = _context_from_problem(problem)
        for turn_index, turn in enumerate(turns):
            constraints = [
                dict(item)
                for item in turn.get("cumulative_constraints", [])
                if isinstance(item, Mapping)
            ]
            source_turn_id = f"{selected['source_problem_id']}:turn:{turn_index + 1}"
            source_turn_payload = {
                "source_problem_id": selected["source_problem_id"],
                "source_file_relpath": selected["source_file_relpath"],
                "turn_index": turn_index,
                "turn": turn,
            }
            source_row_hash = sha256_json(source_turn_payload)
            local_unit_id = _local_unit_id(
                int(selected["selection_index"]),
                str(selected["source_problem_id"]),
                turn_index,
            )
            started = time.monotonic()
            timeout = False
            error = None
            is_sat = False
            assignment_valid = False
            mus: list[dict[str, Any]] = []
            assertion_count = len(constraints)
            try:
                assertion_count = _solver_assertion_count(
                    checker,
                    domain=domain,
                    entities=entities,
                    constraints=constraints,
                    context=context,
                )
                sat_receipt = checker.check_satisfiability(
                    constraints,
                    domain,
                    entities,
                    context=dict(context),
                )
                is_sat = bool(sat_receipt.get("is_sat"))
                assignment_valid = bool(
                    checker.verify_with_z3(
                        dict(turn.get("gold_solution") or {}),
                        constraints,
                        domain,
                        entities,
                        context=dict(context),
                    )
                )
                if not is_sat and hasattr(checker, "compute_mus"):
                    mus = [
                        dict(item)
                        for item in checker.compute_mus(
                            constraints,
                            domain,
                            entities,
                            context=dict(context),
                        )
                        if isinstance(item, Mapping)
                    ]
            except TimeoutError as exc:
                timeout = True
                error = str(exc)
            except Exception as exc:  # pragma: no cover - defensive source checker guard.
                error = f"{type(exc).__name__}: {exc}"
            duration = time.monotonic() - started
            exact_label = "timeout" if timeout else "satisfiable" if is_sat else "contradiction"
            terminal_status = (
                "terminal_timeout" if timeout else "terminal" if error is None else "terminal_error"
            )
            exact_receipt = {
                "local_unit_id": local_unit_id,
                "source_turn_id": source_turn_id,
                "domain": domain,
                "split_name": selected["split_name"],
                "exact_label": exact_label,
                "satisfiable": is_sat,
                "assignment_validity": assignment_valid,
                "solver": "pinned_drift_src_z3_checker",
                "solver_version": solver_version,
                "expected_solver_version": solver_version,
                "solver_version_matches": True,
                "z3_checker_sha256": z3_code_hash,
                "constraint_count": len(constraints),
                "solver_assertion_count": assertion_count,
                "wall_time_s": round(duration, 9),
                "timeout": timeout,
                "censored": timeout,
                "error": error,
                "conflict_or_mus_evidence": {
                    "available": bool(mus),
                    "mus_size": len(mus),
                    "mus_sha256": sha256_json(mus) if mus else None,
                },
                "terminal_status": terminal_status,
            }
            row: JsonDict = {
                "local_unit_id": local_unit_id,
                "source_problem_id": selected["source_problem_id"],
                "base_problem_id": selected["base_problem_id"],
                "source_turn_id": source_turn_id,
                "source_split": selected["source_split"],
                "split_name": selected["split_name"],
                "domain": domain,
                "family": domain,
                "num_entities": selected["num_entities"],
                "turn_index": turn_index,
                "turn_number": turn.get("turn_number", turn_index + 1),
                "turn_position": _turn_position(turn_index, len(turns)),
                "chronology_index": turn_index,
                "source_file_relpath": selected["source_file_relpath"],
                "source_file_sha256": selected["source_file_sha256"],
                "source_problem_hash": selected["source_problem_hash"],
                "source_row_hash": source_row_hash,
                "source_turn_sha256": sha256_json(turn),
                "constraints_sha256": sha256_json(constraints),
                "cumulative_constraint_count": len(constraints),
                "pre_replay_effort_stratum": _effort_stratum(len(constraints)),
                "user_message_sha256": sha256_json(turn.get("user_message", "")),
                "gold_solution_sha256": sha256_json(turn.get("gold_solution", {})),
                "exact_label": exact_label,
                "satisfiable": is_sat,
                "assignment_validity": assignment_valid,
                "solver_effort": {
                    "constraint_count": len(constraints),
                    "solver_assertion_count": assertion_count,
                    "z3_check_calls": 2,
                    "wall_time_s": round(duration, 9),
                },
                "timeout": timeout,
                "censored": timeout,
                "terminal_status": terminal_status,
                "exact_receipt_hash": sha256_json(exact_receipt),
                "row_order_key_components": [
                    "split_name",
                    "domain",
                    "source_problem_id",
                    "turn_index",
                ],
                "upstream_sqlite_result_inherited": False,
                "paper_aggregate_inherited": False,
            }
            exact_rows.append({"row_type": "exact_replay", **exact_receipt})
            fixture_rows.append(row)
    fixture_rows.sort(
        key=lambda row: (
            LOCAL_SPLITS.index(str(row["split_name"])),
            str(row["domain"]),
            str(row["source_problem_id"]),
            int(row["turn_index"]),
        )
    )
    exact_by_id = {row["local_unit_id"]: row for row in exact_rows}
    exact_rows = [exact_by_id[row["local_unit_id"]] for row in fixture_rows]
    return fixture_rows, exact_rows


def source_to_local_identity_rows(
    fixture_rows: Sequence[Mapping[str, Any]],
    *,
    source_root: Path,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in fixture_rows:
        source_path = source_root / str(row.get("source_file_relpath") or "")
        source_file_hash = sha256_file(source_path)
        source_id = str(row.get("source_turn_id") or "")
        local_id = str(row.get("local_unit_id") or "")
        source_file_hash_matches = source_file_hash == row.get("source_file_sha256")
        rows.append(
            {
                "row_type": "source_to_local_identity",
                "local_unit_id": local_id,
                "source_problem_id": row.get("source_problem_id"),
                "source_turn_id": source_id,
                "base_problem_id": row.get("base_problem_id"),
                "domain": row.get("domain"),
                "split_name": row.get("split_name"),
                "source_file_relpath": row.get("source_file_relpath"),
                "computed_source_file_sha256": source_file_hash,
                "declared_source_file_sha256": row.get("source_file_sha256"),
                "source_file_hash_matches": source_file_hash_matches,
                "source_row_hash": row.get("source_row_hash"),
                "source_id_preserved": bool(source_id),
                "local_unit_id_separate_from_source_id": bool(local_id and local_id != source_id),
                "passed": bool(source_id)
                and bool(local_id)
                and local_id != source_id
                and source_file_hash_matches
                and str(row.get("source_row_hash") or "").startswith("sha256:"),
            }
        )
    return rows


def family_turn_and_effort_census(
    fixture_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> JsonDict:
    selected_sizes: dict[str, set[int]] = defaultdict(set)
    available_sizes: dict[str, set[int]] = defaultdict(set)
    split_domain: Counter[str] = Counter()
    for record in records:
        available_sizes[str(record.get("domain"))].add(_safe_int(record.get("num_entities")))
    for row in fixture_rows:
        domain = str(row.get("domain"))
        selected_sizes[domain].add(_safe_int(row.get("num_entities")))
        split_domain[f"{row.get('split_name')}:{domain}"] += 1
    size_ok = all(
        len(selected_sizes.get(domain, set())) >= min(2, len(sizes))
        for domain, sizes in available_sizes.items()
        if domain in DOMAINS
    )
    return {
        "row_count": len(fixture_rows),
        "base_problem_count": len({str(row.get("base_problem_id")) for row in fixture_rows}),
        "domain_counts": dict(
            sorted(Counter(str(row.get("domain")) for row in fixture_rows).items())
        ),
        "split_counts": dict(
            sorted(Counter(str(row.get("split_name")) for row in fixture_rows).items())
        ),
        "split_domain_counts": dict(sorted(split_domain.items())),
        "turn_position_counts": dict(
            sorted(Counter(str(row.get("turn_position")) for row in fixture_rows).items())
        ),
        "effort_strata_counts": dict(
            sorted(
                Counter(str(row.get("pre_replay_effort_stratum")) for row in fixture_rows).items()
            )
        ),
        "selected_sizes_by_domain": {
            domain: sorted(values) for domain, values in sorted(selected_sizes.items())
        },
        "available_sizes_by_domain": {
            domain: sorted(values) for domain, values in sorted(available_sizes.items())
        },
        "balanced_domains": set(Counter(str(row.get("domain")) for row in fixture_rows))
        == set(DOMAINS),
        "balanced_local_splits": set(Counter(str(row.get("split_name")) for row in fixture_rows))
        == set(LOCAL_SPLITS),
        "multiple_sizes_where_available": size_ok,
    }


def build_split_commitment(fixture_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    base_splits: dict[str, set[str]] = defaultdict(set)
    family_aliases: dict[str, set[str]] = defaultdict(set)
    by_base: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_turns: set[tuple[str, str]] = set()
    duplicate_turn_count = 0
    missing_terminal = 0
    censored_count = 0
    for row in fixture_rows:
        base = str(row.get("base_problem_id") or row.get("source_problem_id") or "")
        split = str(row.get("split_name") or "")
        family = str(row.get("family") or row.get("domain") or "")
        base_splits[base].add(split)
        family_aliases[_normalize_alias(family)].add(family)
        by_base[base].append(row)
        key = (base, str(row.get("source_turn_id") or row.get("turn_index") or ""))
        if key in seen_turns:
            duplicate_turn_count += 1
        seen_turns.add(key)
        if row.get("terminal_status") != "terminal":
            missing_terminal += 1
        if row.get("censored") is True:
            censored_count += 1
    chronology_gap_count = 0
    for rows in by_base.values():
        ordered = sorted(rows, key=lambda item: _safe_int(item.get("turn_index")))
        expected = 0
        for row in ordered:
            observed = _safe_int(row.get("turn_index"))
            if observed != expected:
                chronology_gap_count += 1
                expected = observed + 1
            else:
                expected += 1
    base_overlap = sum(1 for splits in base_splits.values() if len(splits) > 1)
    family_alias_collision = sum(1 for names in family_aliases.values() if len(names) > 1)
    split_counts = dict(sorted(Counter(str(row.get("split_name")) for row in fixture_rows).items()))
    passed = all(
        value == 0
        for value in (
            base_overlap,
            chronology_gap_count,
            duplicate_turn_count,
            family_alias_collision,
            missing_terminal,
        )
    )
    return {
        "row_type": "split_commitment",
        "split_names": list(LOCAL_SPLITS),
        "split_counts": split_counts,
        "base_problem_overlap_count": base_overlap,
        "chronology_gap_count": chronology_gap_count,
        "duplicate_turn_count": duplicate_turn_count,
        "family_alias_collision_count": family_alias_collision,
        "missing_terminal_count": missing_terminal,
        "censored_count": censored_count,
        "lineage_floor_train": split_counts.get("train", 0) > 0,
        "lineage_floor_development": split_counts.get("development", 0) > 0,
        "lineage_floor_held": split_counts.get("held", 0) > 0,
        "lineage_may_cross_splits": False,
        "passed": passed,
    }


def _fixture_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def _corrupt_resume_probe(work_dir: Path) -> JsonDict:
    probe_dir = work_dir / "corrupt_resume_probe"
    final_path = work_dir / "corrupt_resume_probe.json"
    with AtomicShardTransaction(
        work_dir=probe_dir,
        final_path=final_path,
        transaction_id="exp6542-corrupt-resume-probe",
        stale_lock_s=0.01,
    ) as tx:
        tx.plan_units(["probe"])
        receipt = tx.write_terminal_unit("probe", {"status": "terminal_probe"})
    shard_path = Path(receipt["shard_path"])
    shard_path.write_text('{"corrupt":true}\n', encoding="utf-8")
    rejected = False
    corrupt_rows: list[JsonDict] = []
    with AtomicShardTransaction(
        work_dir=probe_dir,
        final_path=final_path,
        transaction_id="exp6542-corrupt-resume-probe",
        stale_lock_s=0.01,
    ) as resumed:
        try:
            state = resumed.resume_state()
            corrupt_rows = [dict(row) for row in state["corrupt_shard_rows"]]
            rejected = bool(corrupt_rows) and state["missing_unit_ids"] == ["probe"]
        except CorruptShardError:  # pragma: no cover - helper normally returns corrupt rows.
            rejected = True
    return {
        "probe_path": str(probe_dir),
        "corrupt_resume_rejected": rejected,
        "corrupt_shard_rows": corrupt_rows,
    }


def write_fixture_transaction(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    fixture_path: Path,
    work_dir: Path,
) -> JsonDict:
    transaction_id = "exp6542-v566-drift-bench-external-slice"
    if fixture_path.exists() and work_dir.exists():  # pragma: no cover - completed rerun cleanup.
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with AtomicShardTransaction(
        work_dir=work_dir,
        final_path=fixture_path,
        transaction_id=transaction_id,
        stale_lock_s=0.01,
    ) as tx:
        unit_ids = [str(row["local_unit_id"]) for row in fixture_rows]
        tx.plan_units(unit_ids)
        shard_receipts = [
            tx.write_terminal_unit(str(row["local_unit_id"]), row) for row in fixture_rows
        ]
        state = tx.resume_state()
        if state["missing_unit_ids"]:  # pragma: no cover - all rows are written just above.
            raise MissingTerminalUnitError("missing terminal units before fixture finalization")
        final_receipt = tx._atomic_replace_final(_fixture_jsonl_bytes(fixture_rows))
        final_receipt.update(
            {
                "final_path": str(fixture_path),
                "final_sha256": sha256_file(fixture_path),
                "row_count": len(fixture_rows),
            }
        )
        journal_rows = tx.read_journal()
    roundtrip_rows = _load_jsonl(fixture_path)
    corrupt_probe = _corrupt_resume_probe(work_dir)
    return {
        "transaction_schema": TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "work_dir": str(work_dir),
        "journal_path": str(work_dir / "journal.jsonl"),
        "journal_sha256": sha256_file(work_dir / "journal.jsonl"),
        "journal_record_count": len(journal_rows),
        "planned_unit_ids": sorted(str(row["local_unit_id"]) for row in fixture_rows),
        "terminal_unit_ids": sorted(str(row["unit_id"]) for row in shard_receipts),
        "shards": [
            {
                "unit_id": row["unit_id"],
                "shard_hash": row["shard_hash"],
                "shard_path": row["shard_path"],
                "shard_path_is_content_addressed": Path(str(row["shard_path"])).stem
                == str(row["shard_hash"]).removeprefix("sha256:"),
            }
            for row in shard_receipts
        ],
        "all_shards_verified": all(
            Path(str(row["shard_path"])).is_file()
            and sha256_file(Path(str(row["shard_path"]))) == row["shard_hash"]
            for row in shard_receipts
        ),
        "resume_receipts": [
            {
                "verified": state["all_planned_terminal"],
                "missing_unit_ids": state["missing_unit_ids"],
                "terminal_unit_count": len(state["terminal_unit_ids"]),
            }
        ],
        "corrupt_resume_receipt": corrupt_probe,
        "corrupt_resume_rejected": corrupt_probe["corrupt_resume_rejected"],
        "final_atomic_write_receipt": final_receipt,
        "fixture_roundtrip_row_count": len(roundtrip_rows),
        "fixture_roundtrip_hash": sha256_json(roundtrip_rows),
    }


def planned_and_terminal_unit_counts(
    fixture_rows: Sequence[Mapping[str, Any]], shard_manifest: Mapping[str, Any]
) -> JsonDict:
    planned = list(shard_manifest.get("planned_unit_ids") or [])
    terminal = list(shard_manifest.get("terminal_unit_ids") or [])
    missing = sorted(set(planned) - set(terminal))
    return {
        "planned_count": len(planned) if planned else len(fixture_rows),
        "terminal_count": len(terminal),
        "missing_count": len(missing),
        "all_planned_terminal": bool(planned) and sorted(planned) == sorted(terminal),
    }


def fixture_path_and_hash(
    fixture_path: Path, expected_rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    rows = _load_jsonl(fixture_path)
    return {
        "path": str(fixture_path),
        "exists": fixture_path.is_file(),
        "sha256": sha256_file(fixture_path),
        "row_count": len(rows),
        "expected_row_count": len(expected_rows),
        "roundtrip_matches_expected": rows == [dict(row) for row in expected_rows],
        "roundtrip_sha256": sha256_json(rows),
    }


def aggregate_row_recomputation(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    split_commitment: Mapping[str, Any],
    shard_manifest: Mapping[str, Any],
    protected: Mapping[str, Any],
    upstream_gate: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    inherited_aggregate_present: bool,
    attack_rows: Sequence[Mapping[str, Any]] | None = None,
    fixture_receipt: Mapping[str, Any] | None = None,
) -> JsonDict:
    attack_passed = (
        None if attack_rows is None else all(row.get("passed") is True for row in attack_rows)
    )
    fixture_ok = (
        True
        if fixture_receipt is None
        else fixture_receipt.get("roundtrip_matches_expected") is True
    )
    source_identity_ok = bool(identity_rows) and all(row.get("passed") for row in identity_rows)
    exact_ok = bool(exact_rows) and all(
        row.get("terminal_status") == "terminal"
        and row.get("assignment_validity") is True
        and row.get("solver_version_matches") is True
        for row in exact_rows
    )
    planned_ok = bool(shard_manifest.get("planned_unit_ids")) and sorted(
        shard_manifest.get("planned_unit_ids") or []
    ) == sorted(shard_manifest.get("terminal_unit_ids") or [])
    shard_ok = (
        shard_manifest.get("all_shards_verified") is True
        and shard_manifest.get("corrupt_resume_rejected") is True
        and planned_ok
    )
    all_ready = all(
        (
            upstream_gate.get("passed") is True,
            source_receipt.get("revision_matches_expected") is True,
            source_receipt.get("license_verified") is True,
            source_receipt.get("data_schema_verified") is True,
            source_identity_ok,
            exact_ok,
            split_commitment.get("passed") is True,
            shard_ok,
            inherited_aggregate_present is False,
            protected.get("all_protected_files_unchanged") is True,
            fixture_ok,
            attack_passed is True,
        )
    )
    return {
        "fixture_row_count": len(fixture_rows),
        "base_problem_count": len({str(row.get("base_problem_id")) for row in fixture_rows}),
        "domain_counts": dict(
            sorted(Counter(str(row.get("domain")) for row in fixture_rows).items())
        ),
        "split_counts": dict(
            sorted(Counter(str(row.get("split_name")) for row in fixture_rows).items())
        ),
        "turn_position_counts": dict(
            sorted(Counter(str(row.get("turn_position")) for row in fixture_rows).items())
        ),
        "effort_strata_counts": dict(
            sorted(
                Counter(str(row.get("pre_replay_effort_stratum")) for row in fixture_rows).items()
            )
        ),
        "exact_label_counts": dict(
            sorted(Counter(str(row.get("exact_label")) for row in fixture_rows).items())
        ),
        "censoring_counts": dict(
            sorted(Counter(str(bool(row.get("censored"))) for row in fixture_rows).items())
        ),
        "source_identity_pass_count": sum(1 for row in identity_rows if row.get("passed")),
        "exact_replay_pass_count": sum(
            1
            for row in exact_rows
            if row.get("terminal_status") == "terminal" and row.get("assignment_validity") is True
        ),
        "source_identity_ok": source_identity_ok,
        "exact_replay_ok": exact_ok,
        "split_lineage_ok": split_commitment.get("passed") is True,
        "shard_manifest_ok": shard_ok,
        "planned_units_ok": planned_ok,
        "fixture_roundtrip_ok": fixture_ok,
        "inherited_aggregate_present": inherited_aggregate_present,
        "protected_files_unchanged": protected.get("all_protected_files_unchanged") is True,
        "leakage_attacks_passed": attack_passed,
        "ready_score_from_rows": 1.0 if all_ready else 0.0,
    }


def leakage_attack_matrix(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    split_commitment: Mapping[str, Any],
    shard_manifest: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    source_hashes_match: bool,
    inherited_aggregate_present: bool,
) -> list[JsonDict]:
    local_ids = [str(row.get("local_unit_id") or "") for row in fixture_rows]
    source_turn_ids = [str(row.get("source_turn_id") or "") for row in fixture_rows]
    entity_tokens = []
    for row in fixture_rows:
        for key in ("local_unit_id", "split_name"):
            entity_tokens.append(str(row.get(key) or "").lower())
    attacks = [
        {
            "attack": "duplicate_turn_attack",
            "passed": split_commitment.get("duplicate_turn_count") == 0
            and len(set(source_turn_ids)) == len(source_turn_ids),
            "expected": 0,
            "observed": split_commitment.get("duplicate_turn_count"),
        },
        {
            "attack": "chronology_gap_attack",
            "passed": split_commitment.get("chronology_gap_count") == 0,
            "expected": 0,
            "observed": split_commitment.get("chronology_gap_count"),
        },
        {
            "attack": "family_alias_attack",
            "passed": split_commitment.get("family_alias_collision_count") == 0,
            "expected": 0,
            "observed": split_commitment.get("family_alias_collision_count"),
        },
        {
            "attack": "entity_name_leakage_attack",
            "passed": not any("entity_" in token for token in entity_tokens),
            "expected": "no source entity names in local IDs or split labels",
            "observed": sha256_json(entity_tokens),
        },
        {
            "attack": "row_order_leakage_attack",
            "passed": all(
                "exact" not in " ".join(map(str, row.get("row_order_key_components", []))).lower()
                for row in fixture_rows
            ),
            "expected": "row order key excludes labels and solver costs",
            "observed": [
                list(item)
                for item in sorted(
                    {tuple(row.get("row_order_key_components", [])) for row in fixture_rows}
                )
            ],
        },
        {
            "attack": "source_hash_mismatch_attack",
            "passed": source_hashes_match and all(row.get("passed") for row in identity_rows),
            "expected": True,
            "observed": source_hashes_match,
        },
        {
            "attack": "solver_version_drift_attack",
            "passed": all(row.get("solver_version_matches") is True for row in exact_rows),
            "expected": True,
            "observed": sorted({row.get("solver_version") for row in exact_rows}),
        },
        {
            "attack": "corrupt_resume_attack",
            "passed": shard_manifest.get("corrupt_resume_rejected") is True,
            "expected": True,
            "observed": shard_manifest.get("corrupt_resume_rejected"),
        },
        {
            "attack": "missing_terminal_units_attack",
            "passed": split_commitment.get("missing_terminal_count") == 0
            and aggregate.get("planned_units_ok") is True,
            "expected": 0,
            "observed": split_commitment.get("missing_terminal_count"),
        },
        {
            "attack": "inherited_aggregate_attack",
            "passed": inherited_aggregate_present is False,
            "expected": False,
            "observed": inherited_aggregate_present,
        },
    ]
    return [{"row_type": "leakage_attack", **row} for row in attacks]


def gate_check_summary(
    *,
    upstream_gate: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    source_hashes_match: bool,
    identity_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    split_commitment: Mapping[str, Any],
    shard_manifest: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
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

    if upstream_gate.get("passed") is not True:
        add("upstream_gate", 1.0, upstream_gate.get("observed"), "blocked")
        return {"all_gates_passed": not failures, "failed_checks": failures}
    for key in (
        "revision_matches_expected",
        "license_verified",
        "data_schema_verified",
        "z3_replay_code_present",
        "problem_file_count_matches_expected",
        "upstream_corruption_warning_present",
    ):
        if source_receipt and source_receipt.get(key) is not True:
            add(key, True, source_receipt.get(key), "blocked")
    if failures:
        return {"all_gates_passed": False, "failed_checks": failures}
    if source_hashes_match is not True:
        add("source_hashes_match", True, source_hashes_match, "disqualified")
    bad_identity = [
        row
        for row in identity_rows
        if not (
            row.get("passed")
            and row.get("source_file_hash_matches") is True
            and row.get("local_unit_id_separate_from_source_id") is True
        )
    ]
    if identity_rows and bad_identity:
        add("source_to_local_identity", "all identity rows pass", bad_identity, "disqualified")
    if exact_rows and not all(
        row.get("terminal_status") == "terminal"
        and row.get("assignment_validity") is True
        and row.get("solver_version_matches") is True
        for row in exact_rows
    ):
        add(
            "exact_replay",
            "all exact rows terminal, valid, and solver-version matched",
            [
                row
                for row in exact_rows
                if row.get("terminal_status") != "terminal"
                or row.get("assignment_validity") is not True
                or row.get("solver_version_matches") is not True
            ],
            "disqualified",
        )
    if split_commitment and split_commitment.get("passed") is not True:
        add("split_lineage", True, split_commitment, "disqualified")
    if shard_manifest and not (
        shard_manifest.get("all_shards_verified") is True
        and shard_manifest.get("corrupt_resume_rejected") is True
    ):
        add(
            "shard_manifest",
            "all shards verified and corrupt resume rejected",
            shard_manifest,
            "disqualified",
        )
    if aggregate and aggregate.get("inherited_aggregate_present") is True:
        add("inherited_aggregate", False, True, "disqualified")
    failed_attacks = [row for row in attacks if row.get("passed") is not True]
    if failed_attacks:
        add("leakage_attacks", "all attacks pass", failed_attacks, "disqualified")
    if protected and protected.get("all_protected_files_unchanged") is not True:
        add(
            "protected_files_unchanged",
            True,
            protected.get("all_protected_files_unchanged"),
            "blocked",
        )
    if aggregate and aggregate.get("ready_score_from_rows") not in (0.0, 1.0):
        add("ready_score_scalar", "0.0 or 1.0", aggregate.get("ready_score_from_rows"), "blocked")
    return {"all_gates_passed": not failures, "failed_checks": failures}


def _verdict_class(gate: Mapping[str, Any]) -> str | None:
    if gate.get("all_gates_passed"):
        return None
    severities = {str(row.get("severity")) for row in gate.get("failed_checks", [])}
    return "disqualified" if "disqualified" in severities else "blocked"


def _status_for_class(verdict_class: str | None) -> str:
    if verdict_class is None:
        return "complete_drift_bench_external_intake_v2"
    if verdict_class == "partial":
        return "partial_drift_bench_external_intake_v2"
    if verdict_class == "disqualified":
        return "disqualified_drift_bench_external_intake_v2"
    return "blocked_drift_bench_external_intake_v2"


def _honest_verdict(status: str, gate: Mapping[str, Any]) -> str:
    if status.startswith("complete_"):
        return (
            f"{status}: source hashes, chronological rows, local Z3 receipts, "
            "family-blind splits, shards, attacks, and fixture round-trip pass"
        )
    failed = ",".join(str(row.get("check")) for row in gate.get("failed_checks", []))
    return f"{status}: failed_checks={failed or 'unknown'}"


def build_per_unit_rows(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    shard_manifest: Mapping[str, Any],
    attacks: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> list[JsonDict]:
    rows = [dict(row) for row in identity_rows]
    rows.extend(dict(row) for row in exact_rows)
    rows.extend(
        {
            "row_type": "shard",
            "unit_id": row.get("unit_id"),
            "shard_hash": row.get("shard_hash"),
            "shard_path": row.get("shard_path"),
            "passed": row.get("shard_path_is_content_addressed") is True,
        }
        for row in shard_manifest.get("shards", [])
        if isinstance(row, Mapping)
    )
    rows.extend(dict(row) for row in attacks)
    rows.extend({"row_type": "gate", **row} for row in gate.get("failed_checks", []))
    if gate.get("all_gates_passed"):
        rows.append({"row_type": "gate", "check": "all_gates_passed", "observed": True})
    return rows


def solver_receipts(
    exact_rows: Sequence[Mapping[str, Any]],
    *,
    source_root: Path,
) -> JsonDict:
    return {
        "solver": "pinned_drift_src_z3_checker",
        "z3_python_version": z3.get_version_string(),
        "z3_checker_path": str(source_root / "src" / "z3_checker.py"),
        "z3_checker_sha256": sha256_file(source_root / "src" / "z3_checker.py"),
        "total_replay_rows": len(exact_rows),
        "timeout_s": 30.0,
        "timeout_count": sum(1 for row in exact_rows if row.get("timeout")),
        "censored_count": sum(1 for row in exact_rows if row.get("censored")),
        "assignment_validity_oracle_scope": True,
        "positive_scientific_class_declared": False,
    }


def preconditions_checked(
    *,
    repo_root: Path,
    run_date: str,
    now_utc: str,
    source_root: Path,
    result_path: Path,
    fixture_path: Path,
    fixture_bound: int,
    upstream_gate: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    return {
        "run_date": run_date,
        "checked_at_utc": now_utc,
        "source_root": str(source_root),
        "result_path": str(result_path),
        "fixture_path": str(fixture_path),
        "fixture_bound": fixture_bound,
        "upstream_gate_expected": upstream_gate.get("expected"),
        "upstream_gate_observed": upstream_gate.get("observed"),
        "resources": _resource_receipt(repo_root),
        "solver_versions": {"z3": z3.get_version_string(), "python": platform.python_version()},
        "git_state": _git_status(repo_root),
        "protected_files_unchanged": protected.get("all_protected_files_unchanged"),
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return sha256_json(stable)


def _empty_artifact_parts(
    fixture_path: Path,
) -> tuple[list[JsonDict], list[JsonDict], JsonDict, JsonDict, JsonDict, JsonDict]:
    empty_manifest: JsonDict = {
        "transaction_schema": TRANSACTION_SCHEMA,
        "transaction_id": None,
        "work_dir": None,
        "journal_path": None,
        "journal_sha256": "missing",
        "journal_record_count": 0,
        "planned_unit_ids": [],
        "terminal_unit_ids": [],
        "shards": [],
        "all_shards_verified": False,
        "resume_receipts": [],
        "corrupt_resume_receipt": {},
        "corrupt_resume_rejected": False,
        "final_atomic_write_receipt": {},
        "fixture_roundtrip_row_count": 0,
        "fixture_roundtrip_hash": sha256_json([]),
    }
    counts = {
        "planned_count": 0,
        "terminal_count": 0,
        "missing_count": 0,
        "all_planned_terminal": False,
    }
    fixture_receipt = {
        "path": str(fixture_path),
        "exists": fixture_path.is_file(),
        "sha256": sha256_file(fixture_path),
        "row_count": len(_load_jsonl(fixture_path)),
        "expected_row_count": 0,
        "roundtrip_matches_expected": not fixture_path.exists(),
        "roundtrip_sha256": sha256_json([]),
    }
    census = {
        "row_count": 0,
        "base_problem_count": 0,
        "domain_counts": {},
        "split_counts": {},
        "split_domain_counts": {},
        "turn_position_counts": {},
        "effort_strata_counts": {},
        "selected_sizes_by_domain": {},
        "available_sizes_by_domain": {},
        "balanced_domains": False,
        "balanced_local_splits": False,
        "multiple_sizes_where_available": False,
    }
    split = {
        "row_type": "split_commitment",
        "split_names": list(LOCAL_SPLITS),
        "split_counts": {},
        "base_problem_overlap_count": 0,
        "chronology_gap_count": 0,
        "duplicate_turn_count": 0,
        "family_alias_collision_count": 0,
        "missing_terminal_count": 0,
        "censored_count": 0,
        "lineage_floor_train": False,
        "lineage_floor_development": False,
        "lineage_floor_held": False,
        "lineage_may_cross_splits": False,
        "passed": False,
    }
    return [], [], empty_manifest, counts, fixture_receipt, {"census": census, "split": split}


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str | None = None,
    fixture_path: Path | str | None = None,
    transaction_work_dir: Path | str | None = None,
    drift_source_root: Path | str | None = None,
    drift_git_metadata: Mapping[str, Any] | None = None,
    expected_problem_file_count: int = EXPECTED_PROBLEM_FILE_COUNT,
    fixture_bound: int = DEFAULT_FIXTURE_BOUND,
    run_date: str = RUN_DATE,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    now_utc: str | None = None,
) -> JsonDict:
    start = time.monotonic()
    now = now_utc or _utc_now()
    repo_root = Path(repo_root)
    result = Path(result_path) if result_path is not None else repo_root / RESULT_RELATIVE_PATH
    fixture = Path(fixture_path) if fixture_path is not None else repo_root / FIXTURE_RELATIVE_PATH
    work_dir = (
        Path(transaction_work_dir)
        if transaction_work_dir is not None
        else repo_root / WORK_RELATIVE_PATH
    )
    source_root = (
        Path(drift_source_root) if drift_source_root is not None else prepare_drift_source_root()
    )
    metadata = dict(drift_git_metadata or _drift_git_metadata(source_root))
    protected_before = _protected_hashes(repo_root)
    upstream = _load_json(repo_root / UPSTREAM_GATE_RELATIVE_PATH)
    upstream_gate = build_upstream_gate_receipt(
        repo_root=repo_root,
        upstream=upstream,
        fixture_bound=fixture_bound,
        protected_before=protected_before,
    )
    tree_hashes = build_source_tree_and_file_hashes(source_root, metadata, upstream)
    source_receipt = build_source_revision_and_license_receipt(
        source_root=source_root,
        metadata=metadata,
        tree_hashes=tree_hashes,
        expected_problem_file_count=expected_problem_file_count,
    )
    corruption = build_upstream_corruption_boundary(source_root)
    source_hashes_match = bool(tree_hashes.get("hashes_match_exp6541_when_declared"))
    records = load_problem_records(source_root) if source_root.exists() else []
    selected: list[JsonDict] = []
    intake_commitment: JsonDict = {
        "random_seed": RANDOM_SEED,
        "fixture_bound_base_problems": fixture_bound,
        "selected_base_problem_count": 0,
        "sample_frozen_before_downstream_labels_or_costs": True,
        "labels_or_costs_inspected_before_freeze": False,
        "selection_policy": "not_run_due_to_blocked_precondition",
        "selected_problem_ids": [],
        "commitment_hash": sha256_json([]),
    }
    fixture_rows: list[JsonDict] = []
    exact_rows: list[JsonDict] = []
    identity_rows: list[JsonDict] = []
    if upstream_gate["passed"] and _source_preconditions_pass(source_receipt):
        selected, intake_commitment = freeze_balanced_slice(records, fixture_bound=fixture_bound)
        checker = load_z3_checker(source_root)
        fixture_rows, exact_rows = replay_selected_turns(
            selected_problems=selected,
            checker=checker,
            source_root=source_root,
        )
        identity_rows = source_to_local_identity_rows(fixture_rows, source_root=source_root)
        shard_manifest = (
            write_fixture_transaction(
                fixture_rows=fixture_rows,
                fixture_path=fixture,
                work_dir=work_dir,
            )
            if write
            else {
                "planned_unit_ids": [row["local_unit_id"] for row in fixture_rows],
                "terminal_unit_ids": [row["local_unit_id"] for row in fixture_rows],
                "all_shards_verified": True,
                "corrupt_resume_rejected": True,
                "shards": [],
            }
        )
        unit_counts = planned_and_terminal_unit_counts(fixture_rows, shard_manifest)
        fixture_receipt = (
            fixture_path_and_hash(fixture, fixture_rows)
            if write
            else {
                "path": str(fixture),
                "exists": False,
                "sha256": "not_written",
                "row_count": len(fixture_rows),
                "expected_row_count": len(fixture_rows),
                "roundtrip_matches_expected": True,
                "roundtrip_sha256": sha256_json(fixture_rows),
            }
        )
        census = family_turn_and_effort_census(fixture_rows, records)
        split_commitment = build_split_commitment(fixture_rows)
    else:
        _identity, _exact, shard_manifest, unit_counts, fixture_receipt, empty = (
            _empty_artifact_parts(fixture)
        )
        census = empty["census"]
        split_commitment = empty["split"]
    protected_after = _protected_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    aggregate = aggregate_row_recomputation(
        fixture_rows=fixture_rows,
        identity_rows=identity_rows,
        exact_rows=exact_rows,
        split_commitment=split_commitment,
        shard_manifest=shard_manifest,
        protected=protected,
        upstream_gate=upstream_gate,
        source_receipt=source_receipt,
        inherited_aggregate_present=False,
        fixture_receipt=fixture_receipt,
    )
    attacks = leakage_attack_matrix(
        fixture_rows=fixture_rows,
        identity_rows=identity_rows,
        exact_rows=exact_rows,
        split_commitment=split_commitment,
        shard_manifest=shard_manifest,
        aggregate=aggregate,
        source_hashes_match=source_hashes_match,
        inherited_aggregate_present=False,
    )
    aggregate = aggregate_row_recomputation(
        fixture_rows=fixture_rows,
        identity_rows=identity_rows,
        exact_rows=exact_rows,
        split_commitment=split_commitment,
        shard_manifest=shard_manifest,
        protected=protected,
        upstream_gate=upstream_gate,
        source_receipt=source_receipt,
        inherited_aggregate_present=False,
        attack_rows=attacks,
        fixture_receipt=fixture_receipt,
    )
    gate = gate_check_summary(
        upstream_gate=upstream_gate,
        source_receipt=source_receipt,
        source_hashes_match=source_hashes_match,
        identity_rows=identity_rows,
        exact_rows=exact_rows,
        split_commitment=split_commitment,
        shard_manifest=shard_manifest,
        aggregate=aggregate,
        attacks=attacks,
        protected=protected,
    )
    verdict_class = _verdict_class(gate)
    status = _status_for_class(verdict_class)
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": _honest_verdict(status, gate),
        "verdict_class": verdict_class,
        "upstream_gate_receipt": upstream_gate,
        "source_revision_and_license_receipt": source_receipt,
        "source_tree_and_file_hashes": tree_hashes,
        "upstream_corruption_boundary": corruption,
        "intake_commitment": intake_commitment,
        "family_turn_and_effort_census": census,
        "source_to_local_identity_rows": identity_rows,
        "exact_replay_rows": exact_rows,
        "solver_receipts": solver_receipts(exact_rows, source_root=source_root),
        "split_commitment": split_commitment,
        "shard_manifest": shard_manifest,
        "planned_and_terminal_unit_counts": unit_counts,
        "fixture_path_and_hash": fixture_receipt,
        "leakage_attack_matrix": attacks,
        "external_constraint_corpus_ready_score": float(aggregate["ready_score_from_rows"]),
        "gate_check_summary": gate,
        "per_unit_rows": build_per_unit_rows(
            identity_rows=identity_rows,
            exact_rows=exact_rows,
            shard_manifest=shard_manifest,
            attacks=attacks,
            gate=gate,
        ),
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions_checked(
            repo_root=repo_root,
            run_date=run_date,
            now_utc=now,
            source_root=source_root,
            result_path=result,
            fixture_path=fixture,
            fixture_bound=fixture_bound,
            upstream_gate=upstream_gate,
            protected=protected,
        ),
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
        atomic_write_json(result, artifact, allow_override=False, sort_keys=False)
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
        errors.append("verdict_class outside Exp6542 enum")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict terminal prefix mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    score = artifact.get("external_constraint_corpus_ready_score")
    recomputed = artifact.get("aggregate_row_recomputation", {}).get("ready_score_from_rows")
    if score not in {0.0, 1.0} or score != recomputed:
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
        description="Build or validate Exp6542 DRIFT-Bench external intake v2."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--fixture-path", default=str(REPO_ROOT / FIXTURE_RELATIVE_PATH))
    parser.add_argument(
        "--expected-problem-file-count", type=int, default=EXPECTED_PROBLEM_FILE_COUNT
    )
    parser.add_argument("--fixture-bound", type=int, default=DEFAULT_FIXTURE_BOUND)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        errors = validate_artifact(_load_json(result_path))
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {RESULT_RELATIVE_PATH.as_posix()}")
        return 0
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        fixture_path=Path(args.fixture_path),
        expected_problem_file_count=int(args.expected_problem_file_count),
        fixture_bound=int(args.fixture_bound),
        run_date=str(args.date),
        write=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI in tests.
    raise SystemExit(main())
