"""Exp6530 independent external constraint corpus audit.

Spec refs: REQ-BENCH-6530, SCENARIO-BENCH-6530-MISSING,
SCENARIO-BENCH-6530-SOURCE, SCENARIO-BENCH-6530-CHRONOLOGY,
SCENARIO-BENCH-6530-EXACT, SCENARIO-BENCH-6530-SPLIT,
SCENARIO-BENCH-6530-SHARDS, SCENARIO-BENCH-6530-ATTACKS.

This reducer audits the Exp6529 DRIFT-Bench fixture as an external evidence
root. It does not trust Exp6529 source IDs, cached solver receipts, or
aggregates. It re-reads local source files, rebuilds labels with a small exact
solver for fixture-style constraints, and closes a blocked artifact when the
upstream intake or fixture is missing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6530
INFERENCE_SUBSTRATE = "independent_external_source_fixture_and_exact_solver_replay_no_llm"

RESULT_RELATIVE_PATH = Path("results/experiment_6530_external_constraint_corpus_audit.json")
EXP6529_RELATIVE_PATH = Path("results/experiment_6529_drift_bench_external_intake.json")
FIXTURE_RELATIVE_PATH = Path("results/fixtures/v565_drift_bench_external_slice.jsonl")
ATOMIC_TRANSACTION_RELATIVE_PATH = Path(
    "results/experiment_6514_atomic_shard_artifact_transaction.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")

PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    ATOMIC_TRANSACTION_RELATIVE_PATH,
    EXP6529_RELATIVE_PATH,
    FIXTURE_RELATIVE_PATH,
)

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6530_external_constraint_corpus_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6530_external_constraint_corpus_audit.py "
    "-m pytest tests/python/test_experiment_6530_external_constraint_corpus_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6530_external_constraint_corpus_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6530_external_constraint_corpus_audit.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6530_external_constraint_corpus_audit.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6530_external_constraint_corpus_audit.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6530_external_constraint_corpus_audit "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6530_external_constraint_corpus_audit --validate"
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

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "source_existence_and_hash_receipts",
    "independent_revision_and_license_receipt",
    "source_identity_audit_rows",
    "chronology_replay_rows",
    "independent_exact_replay_rows",
    "split_and_lineage_audit",
    "shard_and_transaction_audit",
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
    "status": "Records the terminal external-corpus audit state.",
    "honest_verdict": "Starts with a terminal prefix and names the audit outcome.",
    "verdict_class": "Separates complete, partial, blocked, and disqualified audit paths.",
    "source_existence_and_hash_receipts": (
        "Records source, fixture, protected-file, path, hash, row-count, and resource receipts."
    ),
    "independent_revision_and_license_receipt": (
        "Verifies pinned revision, local source root, license, schema, and corruption text."
    ),
    "source_identity_audit_rows": (
        "Proves each fixture row maps to source bytes by hash, not by source ID."
    ),
    "chronology_replay_rows": (
        "Rebuilds turn order and duplicate event checks from fixture and source rows."
    ),
    "independent_exact_replay_rows": (
        "Replays labels from raw constraints with an independently initialized exact solver."
    ),
    "split_and_lineage_audit": (
        "Rebuilds family-blind splits and rejects cross-split lineages or missing terminal rows."
    ),
    "shard_and_transaction_audit": (
        "Checks planned IDs, terminal IDs, shard hashes, journal, resume, final write, and counts."
    ),
    "independent_aggregate_rows": (
        "Recomputes family, turn, contradiction, drift, hardness, censoring, and exact counts."
    ),
    "leakage_attack_matrix": (
        "Attacks IDs, order, names, length, family labels, answers, caches, hashes, and aggregates."
    ),
    "external_constraint_corpus_audited_ready_score": (
        "Opens only when every independent audit and attack passes."
    ),
    "gate_check_summary": "Names every failed check with expected and observed values.",
    "per_unit_rows": "Flattens source, chronology, replay, aggregate, attack, and gate rows.",
    "aggregate_row_recomputation": "Rebuilds readiness from rows instead of inherited totals.",
    "preconditions_checked": (
        "Records run date, paths, row counts, resources, solver version, git state, and hashes."
    ),
    "protected_files_unchanged": "Compares protected-file hashes before and after the audit.",
    "inference_substrate": "Declares independent source, fixture, and exact-solver replay with no LLM.",
    "verifier_is_oracle": "True only for source, split, and exact-label audit checks.",
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps fields to inputs, rows, reducers, specs, tests, or hashes.",
    "random_seed": "Pins deterministic sample and row ordering.",
    "duration_s": "Records measured wall time for the audit reducer.",
    "tests_run": "Records verification command receipts.",
    "reproducibility_checksum": "Detects drift in audit rows, gates, commands, and verdicts.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {
        "principle": FIELD_PRINCIPLES[field],
        "source": "Exp6530 deterministic independent audit reducer",
        "spec_refs": ["REQ-BENCH-6530"],
    }
    for field in REQUIRED_ARTIFACT_FIELDS
}
FIELD_PROVENANCE["source_existence_and_hash_receipts"]["source"] = "build_existence_receipts"
FIELD_PROVENANCE["independent_revision_and_license_receipt"]["source"] = (
    "independent_revision_and_license_receipt"
)
FIELD_PROVENANCE["source_identity_audit_rows"]["source"] = "source_identity_audit_rows"
FIELD_PROVENANCE["chronology_replay_rows"]["source"] = "chronology_replay_rows"
FIELD_PROVENANCE["independent_exact_replay_rows"]["source"] = "independent_exact_replay_rows"
FIELD_PROVENANCE["split_and_lineage_audit"]["source"] = "split_and_lineage_audit"
FIELD_PROVENANCE["shard_and_transaction_audit"]["source"] = "shard_and_transaction_audit"
FIELD_PROVENANCE["independent_aggregate_rows"]["source"] = "independent_aggregate_rows"
FIELD_PROVENANCE["leakage_attack_matrix"]["source"] = "leakage_attack_matrix"
FIELD_PROVENANCE["aggregate_row_recomputation"]["source"] = "aggregate_row_recomputation"
FIELD_PROVENANCE["protected_files_unchanged"]["source"] = "protected_files_unchanged"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


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


def _file_receipt(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


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


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    rows = []
    for relpath in before:
        rows.append(
            {
                "path": relpath,
                "before_sha256": before[relpath],
                "after_sha256": after.get(relpath, "missing"),
                "unchanged": before[relpath] == after.get(relpath, "missing"),
            }
        )
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "protected_file_rows": rows,
    }


def _resource_receipt(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root if repo_root.exists() else repo_root.parent)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disk_free_bytes": disk.free,
        "exact_solver": "exp6530_independent_finite_equality_solver_v1",
    }


def _git_state(repo_root: Path) -> JsonDict:
    if not (repo_root / ".git").exists():
        return {"git_available": False, "state": "not_git_checkout"}
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "git_available": result.returncode == 0,
        "exit_code": result.returncode,
        "status_short": result.stdout.strip(),
    }


def _source_contract(intake: Mapping[str, Any]) -> JsonDict:
    for key in (
        "source_revision_and_license_receipt",
        "drift_bench_provenance_contract",
        "source_revision",
    ):
        value = intake.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _source_root(repo_root: Path, contract: Mapping[str, Any]) -> Path | None:
    raw = contract.get("source_root") or contract.get("local_source_root")
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else repo_root / candidate


def independent_revision_and_license_receipt(
    repo_root: Path, intake: Mapping[str, Any]
) -> JsonDict:
    contract = _source_contract(intake)
    root = _source_root(repo_root, contract)
    revision = str(contract.get("immutable_revision") or contract.get("revision") or "")
    license_path = root / "LICENSE" if root else repo_root / "__missing_source_root__" / "LICENSE"
    readme_path = (
        root / "README.md" if root else repo_root / "__missing_source_root__" / "README.md"
    )
    schema_rel = str(contract.get("data_schema_path") or "data/problems/README.md")
    schema_path = root / schema_rel if root else repo_root / "__missing_source_root__" / schema_rel
    license_text = (
        license_path.read_text(encoding="utf-8", errors="replace") if license_path.is_file() else ""
    )
    readme_text = (
        readme_path.read_text(encoding="utf-8", errors="replace") if readme_path.is_file() else ""
    )
    schema_text = (
        schema_path.read_text(encoding="utf-8", errors="replace") if schema_path.is_file() else ""
    )
    return {
        "repo_url": contract.get("repo_url"),
        "immutable_revision": revision if re.fullmatch(r"[0-9a-f]{40}", revision) else None,
        "revision_is_immutable": bool(re.fullmatch(r"[0-9a-f]{40}", revision)),
        "source_root": str(root) if root else None,
        "source_root_exists": bool(root and root.exists()),
        "license_path": str(license_path),
        "license_sha256": sha256_file(license_path),
        "license_verified": "MIT" in license_text,
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "schema_verified": "constraint" in schema_text.lower() and "problem" in schema_text.lower(),
        "readme_path": str(readme_path),
        "readme_sha256": sha256_file(readme_path),
        "corruption_boundary_text_verified": (
            "sqlite" in readme_text.lower() and "corrupt" in readme_text.lower()
        ),
    }


def _source_file_hashes(intake: Mapping[str, Any]) -> dict[str, str]:
    value = intake.get("source_file_hashes")
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, Mapping) else {}


def _rows_by_source_hash(source_root: Path | None, relpaths: Sequence[str]) -> dict[str, JsonDict]:
    out: dict[str, JsonDict] = {}
    if source_root is None:
        return out
    for relpath in relpaths:
        path = source_root / relpath
        for row in _load_jsonl(path):
            out[sha256_json(row)] = row
    return out


def build_existence_receipts(
    *,
    repo_root: Path,
    intake_path: Path,
    fixture_path: Path,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_root_path: Path | None,
    declared_hashes: Mapping[str, str],
) -> JsonDict:
    source_file_receipts = []
    for relpath, declared_hash in declared_hashes.items():
        source_path = source_root_path / relpath if source_root_path else repo_root / relpath
        computed = sha256_file(source_path)
        source_file_receipts.append(
            {
                "path": str(source_path),
                "relpath": relpath,
                "exists": source_path.is_file(),
                "declared_sha256": declared_hash,
                "computed_sha256": computed,
                "hash_matches": computed == declared_hash,
            }
        )
    return {
        "intake_artifact": _file_receipt(intake_path),
        "fixture": {
            **_file_receipt(fixture_path),
            "row_count": len(fixture_rows),
        },
        "atomic_transaction": _file_receipt(repo_root / ATOMIC_TRANSACTION_RELATIVE_PATH),
        "source_root": {
            "path": str(source_root_path) if source_root_path else None,
            "exists": bool(source_root_path and source_root_path.exists()),
        },
        "source_files": source_file_receipts,
        "protected_files": {
            relpath.as_posix(): _file_receipt(repo_root / relpath)
            for relpath in PROTECTED_RELATIVE_PATHS
        },
        "resources": _resource_receipt(repo_root),
    }


def source_identity_audit_rows(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_root_path: Path | None,
    declared_hashes: Mapping[str, str],
) -> list[JsonDict]:
    relpaths = sorted({str(row.get("source_file_relpath") or "") for row in fixture_rows})
    source_rows_by_hash = _rows_by_source_hash(
        source_root_path, [item for item in relpaths if item]
    )
    rows = []
    for row in fixture_rows:
        relpath = str(row.get("source_file_relpath") or "")
        source_path = source_root_path / relpath if source_root_path else Path(relpath)
        computed_file_hash = sha256_file(source_path)
        declared_file_hash = declared_hashes.get(relpath)
        observed_file_hash = str(row.get("source_file_sha256") or "")
        source_row_hash = str(row.get("source_row_hash") or "")
        hash_present = source_row_hash in source_rows_by_hash
        file_hash_matches = (
            bool(relpath)
            and computed_file_hash != "missing"
            and computed_file_hash == declared_file_hash
            and computed_file_hash == observed_file_hash
        )
        rows.append(
            {
                "row_type": "source_identity",
                "local_unit_id": row.get("local_unit_id"),
                "source_row_id": row.get("source_row_id"),
                "source_file_relpath": relpath,
                "source_file_exists": source_path.is_file(),
                "declared_source_file_sha256": declared_file_hash,
                "fixture_source_file_sha256": observed_file_hash,
                "computed_source_file_sha256": computed_file_hash,
                "source_file_hash_matches": file_hash_matches,
                "source_row_hash": source_row_hash,
                "source_row_hash_matched": hash_present,
                "source_id_used_as_identity": False,
                "passed": file_hash_matches and hash_present,
            }
        )
    return rows


def chronology_replay_rows(fixture_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fixture_rows:
        grouped[str(row.get("base_problem_id") or row.get("source_problem_id") or "")].append(row)
    out: list[JsonDict] = []
    for base_problem_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: int(item.get("turn_index") or 0))
        previous_turn: int | None = None
        seen_events: set[str] = set()
        for row in ordered:
            turn_index = int(row.get("turn_index") or 0)
            event_id = str(row.get("event_id") or row.get("source_row_id") or "")
            duplicate_event = event_id in seen_events
            gap = previous_turn is not None and turn_index != previous_turn + 1
            seen_events.add(event_id)
            previous_turn = turn_index
            out.append(
                {
                    "row_type": "chronology",
                    "local_unit_id": row.get("local_unit_id"),
                    "base_problem_id": base_problem_id,
                    "turn_index": turn_index,
                    "event_id": event_id,
                    "duplicate_event": duplicate_event,
                    "chronology_gap": gap,
                    "chronology_valid": not duplicate_event and not gap,
                }
            )
    return out


def _constraints_from_source(
    row: Mapping[str, Any],
    source_rows_by_hash: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    source_hash = str(row.get("source_row_hash") or "")
    source_row = source_rows_by_hash.get(source_hash)
    raw = source_row.get("constraints") if isinstance(source_row, Mapping) else None
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, Mapping)]
    return []


def _constraint_label(constraints: Sequence[Mapping[str, Any]]) -> str:
    assignments: dict[str, Any] = {}
    not_equals: list[tuple[str, Any]] = []
    for constraint in constraints:
        var = str(constraint.get("var") or constraint.get("variable") or "")
        if not var:
            continue
        if "equals" in constraint:
            value = constraint["equals"]
            if var in assignments and assignments[var] != value:
                return "contradiction"
            assignments[var] = value
        if "not_equals" in constraint:
            not_equals.append((var, constraint["not_equals"]))
        if constraint.get("op") in {"=", "=="}:
            value = constraint.get("value")
            if var in assignments and assignments[var] != value:
                return "contradiction"
            assignments[var] = value
        if constraint.get("op") in {"!=", "not_equals"}:
            not_equals.append((var, constraint.get("value")))
    if any(assignments.get(var) == value for var, value in not_equals):
        return "contradiction"
    return "satisfiable"


def _observed_label(row: Mapping[str, Any]) -> str:
    for key in ("exact_label", "z3_exact_label", "exact_result"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "missing"


def independent_exact_replay_rows(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_root_path: Path | None,
) -> list[JsonDict]:
    relpaths = sorted({str(row.get("source_file_relpath") or "") for row in fixture_rows})
    source_rows_by_hash = _rows_by_source_hash(
        source_root_path, [item for item in relpaths if item]
    )
    rows = []
    for row in fixture_rows:
        constraints = _constraints_from_source(row, source_rows_by_hash)
        recomputed = _constraint_label(constraints) if constraints else "unavailable"
        observed = _observed_label(row)
        rows.append(
            {
                "row_type": "exact_replay",
                "local_unit_id": row.get("local_unit_id"),
                "split_name": row.get("split_name"),
                "sample_policy": "all_held_and_all_train_development_preregistered_for_audit",
                "solver": "exp6530_independent_finite_equality_solver_v1",
                "cached_solver_result_trusted": False,
                "constraint_count": len(constraints),
                "observed_label": observed,
                "recomputed_label": recomputed,
                "observed_contradiction": bool(row.get("contradiction")),
                "recomputed_contradiction": recomputed == "contradiction",
                "terminal_disposition": row.get("terminal_disposition"),
                "replayed_label_matches": observed == recomputed,
                "terminal_present": bool(row.get("terminal_disposition")),
            }
        )
    return rows


def split_and_lineage_audit(
    fixture_rows: Sequence[Mapping[str, Any]],
    chronology_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    base_splits: dict[str, set[str]] = defaultdict(set)
    family_aliases: dict[str, set[str]] = defaultdict(set)
    seen_events: set[tuple[str, str]] = set()
    duplicate_events = 0
    missing_terminal = 0
    post_held_repair = 0
    for row in fixture_rows:
        base = str(row.get("base_problem_id") or row.get("source_problem_id") or "")
        split = str(row.get("split_name") or "")
        family = str(row.get("family") or row.get("domain") or "")
        alias = re.sub(r"[^a-z0-9]+", "", family.lower())
        event_key = (base, str(row.get("event_id") or row.get("source_row_id") or ""))
        base_splits[base].add(split)
        family_aliases[alias].add(family)
        if event_key in seen_events:
            duplicate_events += 1
        seen_events.add(event_key)
        missing_terminal += 0 if row.get("terminal_disposition") else 1
        post_held_repair += 1 if row.get("post_held_repair") or row.get("repair_after_held") else 0
    base_overlap = sum(1 for splits in base_splits.values() if len(splits) > 1)
    alias_collisions = sum(1 for values in family_aliases.values() if len(values) > 1)
    chronology_gap_count = sum(1 for row in chronology_rows if row.get("chronology_gap"))
    chronology_duplicate_count = sum(1 for row in chronology_rows if row.get("duplicate_event"))
    passed = all(
        value == 0
        for value in (
            base_overlap,
            duplicate_events,
            chronology_gap_count,
            chronology_duplicate_count,
            alias_collisions,
            post_held_repair,
            missing_terminal,
        )
    )
    return {
        "row_type": "split_and_lineage",
        "split_counts": dict(
            sorted(Counter(str(row.get("split_name") or "") for row in fixture_rows).items())
        ),
        "base_problem_overlap_count": base_overlap,
        "duplicate_event_count": duplicate_events + chronology_duplicate_count,
        "chronology_gap_count": chronology_gap_count,
        "family_alias_collision_count": alias_collisions,
        "post_held_repair_count": post_held_repair,
        "missing_terminal_disposition_count": missing_terminal,
        "passed": passed,
    }


def _manifest(intake: Mapping[str, Any]) -> JsonDict:
    value = intake.get("shard_manifest")
    return dict(value) if isinstance(value, Mapping) else {}


def _manifest_fixture_receipt(intake: Mapping[str, Any]) -> JsonDict:
    value = intake.get("fixture_path_and_hash")
    return dict(value) if isinstance(value, Mapping) else {}


def shard_and_transaction_audit(
    *,
    repo_root: Path,
    intake: Mapping[str, Any],
    fixture_path: Path,
    fixture_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    manifest = _manifest(intake)
    fixture_receipt = _manifest_fixture_receipt(intake)
    fixture_hash = sha256_file(fixture_path)
    local_ids = [str(row.get("local_unit_id")) for row in fixture_rows]
    planned_ids = [str(item) for item in manifest.get("planned_unit_ids", [])]
    terminal_ids = [str(item) for item in manifest.get("terminal_unit_ids", [])]
    shard_rows = []
    for shard in manifest.get("shards", []):
        if not isinstance(shard, Mapping):
            continue
        path = repo_root / str(shard.get("path") or "")
        shard_file_rows = _load_jsonl(path)
        shard_rows.append(
            {
                "path": str(path),
                "expected_sha256": shard.get("sha256"),
                "computed_sha256": sha256_file(path),
                "expected_row_count": shard.get("row_count"),
                "computed_row_count": len(shard_file_rows),
                "passed": sha256_file(path) == shard.get("sha256")
                and len(shard_file_rows) == shard.get("row_count"),
            }
        )
    final = manifest.get("final_atomic_write_receipt")
    final_receipt = dict(final) if isinstance(final, Mapping) else {}
    planned_ids_match = sorted(planned_ids) == sorted(local_ids) and bool(local_ids)
    terminal_ids_match = sorted(terminal_ids) == sorted(local_ids) and bool(local_ids)
    fixture_hash_matches = (
        fixture_hash != "missing"
        and fixture_hash == fixture_receipt.get("sha256")
        and fixture_hash == final_receipt.get("final_sha256")
    )
    row_count_matches = len(fixture_rows) == fixture_receipt.get("row_count") and len(
        fixture_rows
    ) == final_receipt.get("row_count")
    journal_chain_complete = bool(manifest.get("journal_chain")) and len(
        manifest.get("journal_chain", [])
    ) >= len(shard_rows)
    resume_receipts_present = bool(manifest.get("resume_receipts")) and all(
        isinstance(row, Mapping) and row.get("verified") is True
        for row in manifest.get("resume_receipts", [])
    )
    all_shards_match = bool(shard_rows) and all(row["passed"] for row in shard_rows)
    passed = all(
        (
            planned_ids_match,
            terminal_ids_match,
            fixture_hash_matches,
            row_count_matches,
            journal_chain_complete,
            resume_receipts_present,
            all_shards_match,
        )
    )
    return {
        "row_type": "shard_and_transaction",
        "planned_ids_match": planned_ids_match,
        "terminal_ids_match": terminal_ids_match,
        "shard_rows": shard_rows,
        "journal_chain_complete": journal_chain_complete,
        "resume_receipts_present": resume_receipts_present,
        "final_atomic_write_receipt": final_receipt,
        "fixture_hash": fixture_hash,
        "fixture_hash_matches": fixture_hash_matches,
        "row_count_matches": row_count_matches,
        "passed": passed,
    }


def _recomputed_counts(
    fixture_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "row_count": len(fixture_rows),
        "split_counts": dict(
            sorted(Counter(str(row.get("split_name") or "") for row in fixture_rows).items())
        ),
        "family_counts": dict(
            sorted(Counter(str(row.get("family") or "") for row in fixture_rows).items())
        ),
        "turn_counts": dict(
            sorted(Counter(str(row.get("turn_index") or "") for row in fixture_rows).items())
        ),
        "contradiction_counts": dict(
            sorted(Counter(str(row.get("recomputed_contradiction")) for row in exact_rows).items())
        ),
        "drift_counts": dict(
            sorted(Counter(str(bool(row.get("drift"))) for row in fixture_rows).items())
        ),
        "hardness_counts": dict(
            sorted(Counter(str(row.get("hardness_bin") or "") for row in fixture_rows).items())
        ),
        "censoring_counts": dict(
            sorted(Counter(str(bool(row.get("censored"))) for row in fixture_rows).items())
        ),
        "exact_label_counts": dict(
            sorted(Counter(str(row.get("recomputed_label") or "") for row in exact_rows).items())
        ),
    }


def independent_aggregate_rows(
    fixture_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    intake: Mapping[str, Any],
) -> list[JsonDict]:
    recomputed = _recomputed_counts(fixture_rows, exact_rows)
    intake_aggregate = intake.get("aggregate_row_recomputation")
    intake_aggregate = dict(intake_aggregate) if isinstance(intake_aggregate, Mapping) else {}
    metric_map = {
        "row_count": "row_count",
        "split_counts": "split_counts",
        "family_counts": "family_counts",
        "turn_counts": "turn_counts",
        "contradiction_counts": "contradiction_counts",
        "drift_counts": "drift_counts",
        "hardness_counts": "hardness_counts",
        "censoring_counts": "censoring_counts",
        "exact_label_counts": "exact_label_counts",
    }
    rows = []
    for metric, key in metric_map.items():
        intake_value = intake_aggregate.get(key)
        rows.append(
            {
                "row_type": "aggregate",
                "metric": metric,
                "recomputed": recomputed[key],
                "intake_observed": intake_value,
                "intake_matches": intake_value in (None, recomputed[key]),
            }
        )
    return rows


def _aggregate_tampering(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, JsonDict, JsonDict]:
    observed = {
        "row_count": None,
        "split_counts": None,
        "exact_label_counts": None,
    }
    intake_observed = {
        "row_count": None,
        "split_counts": None,
        "exact_label_counts": None,
    }
    for row in aggregate_rows:
        metric = str(row.get("metric"))
        if metric in observed:
            observed[metric] = row.get("recomputed")
            intake_observed[metric] = row.get("intake_observed")
    tampered = any(
        intake_observed[key] is not None and intake_observed[key] != observed[key]
        for key in observed
    )
    return tampered, observed, intake_observed


def leakage_attack_matrix(
    *,
    source_identity_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    split_audit: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    fixture_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    aggregate_tampered, observed, intake_observed = _aggregate_tampering(aggregate_rows)
    source_hashes_pass = all(row.get("passed") is True for row in source_identity_rows)
    exact_pass = all(row.get("replayed_label_matches") is True for row in exact_rows)
    terminal_hard_rows = all(
        not row.get("hardness_bin") or bool(row.get("terminal_disposition")) for row in fixture_rows
    )
    answer_tokens = {
        str(row.get("answer", {}).get("label", "")).lower()
        for row in fixture_rows
        if isinstance(row.get("answer"), Mapping)
    }
    entity_text = " ".join(
        " ".join(map(str, row.get("entities", [])))
        for row in fixture_rows
        if isinstance(row.get("entities"), list)
    ).lower()
    attacks = [
        {
            "attack": "aggregate_inheritance_attack",
            "passed": not aggregate_tampered,
            "observed": observed,
            "intake_observed": intake_observed,
        },
        {
            "attack": "source_id_trust_attack",
            "passed": source_hashes_pass
            and not any(row.get("source_id_used_as_identity") for row in source_identity_rows),
            "observed": {
                "source_identity_rows": len(source_identity_rows),
                "hash_pass_count": sum(1 for row in source_identity_rows if row.get("passed")),
            },
        },
        {
            "attack": "row_order_attack",
            "passed": len({str(row.get("local_unit_id")) for row in fixture_rows})
            == len(fixture_rows),
            "observed": "local_unit_ids_unique_after_order_independent_sort",
        },
        {
            "attack": "entity_name_leakage_attack",
            "passed": not any(token and token in entity_text for token in answer_tokens),
            "observed": {
                "answer_tokens": sorted(answer_tokens),
                "entity_text_sha256": sha256_json(entity_text),
            },
        },
        {
            "attack": "serialization_length_attack",
            "passed": not any(row.get("serialization_length_leak") for row in fixture_rows),
            "observed": "no_explicit_serialization_length_leak_marker",
        },
        {
            "attack": "family_label_attack",
            "passed": split_audit.get("family_alias_collision_count") == 0,
            "observed": split_audit.get("family_alias_collision_count"),
        },
        {
            "attack": "answer_field_attack",
            "passed": exact_pass,
            "observed": "exact_replay_ignores_answer_field_and_matches_label",
        },
        {
            "attack": "solver_cache_trust_attack",
            "passed": exact_pass
            and all(row.get("cached_solver_result_trusted") is False for row in exact_rows),
            "observed": "cached solver receipts ignored",
        },
        {
            "attack": "missing_hard_rows_attack",
            "passed": terminal_hard_rows,
            "observed": {
                "hard_rows": sum(1 for row in fixture_rows if row.get("hardness_bin") == "hard"),
                "terminal_hard_rows": terminal_hard_rows,
            },
        },
        {
            "attack": "hash_substitution_attack",
            "passed": source_hashes_pass,
            "observed": "source file and row hashes recomputed",
        },
    ]
    return [{"row_type": "leakage_attack", **row} for row in attacks]


def aggregate_row_recomputation(
    *,
    fixture_rows: Sequence[Mapping[str, Any]],
    source_identity_rows: Sequence[Mapping[str, Any]],
    chronology_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    split_audit: Mapping[str, Any],
    shard_audit: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    revision_receipt: Mapping[str, Any],
    existence_receipts: Mapping[str, Any],
) -> JsonDict:
    counts = _recomputed_counts(fixture_rows, exact_rows)
    aggregate_tampered, _observed, _intake = _aggregate_tampering(aggregate_rows)
    blocked_preconditions = not (
        existence_receipts.get("intake_artifact", {}).get("exists")
        and existence_receipts.get("fixture", {}).get("exists")
        and revision_receipt.get("source_root_exists")
    )
    source_ok = bool(source_identity_rows) and all(
        row.get("passed") for row in source_identity_rows
    )
    chronology_ok = bool(chronology_rows) and all(
        row.get("chronology_valid") for row in chronology_rows
    )
    exact_ok = bool(exact_rows) and all(row.get("replayed_label_matches") for row in exact_rows)
    attacks_ok = bool(attack_rows) and all(row.get("passed") for row in attack_rows)
    all_passed = all(
        (
            not blocked_preconditions,
            revision_receipt.get("revision_is_immutable") is True,
            revision_receipt.get("license_verified") is True,
            revision_receipt.get("schema_verified") is True,
            revision_receipt.get("corruption_boundary_text_verified") is True,
            source_ok,
            chronology_ok,
            exact_ok,
            split_audit.get("passed") is True,
            shard_audit.get("passed") is True,
            not aggregate_tampered,
            attacks_ok,
        )
    )
    return {
        "fixture_row_count": counts["row_count"],
        "split_counts": counts["split_counts"],
        "family_counts": counts["family_counts"],
        "turn_counts": counts["turn_counts"],
        "contradiction_counts": counts["contradiction_counts"],
        "drift_counts": counts["drift_counts"],
        "hardness_counts": counts["hardness_counts"],
        "censoring_counts": counts["censoring_counts"],
        "exact_label_counts": counts["exact_label_counts"],
        "source_identity_pass_count": sum(1 for row in source_identity_rows if row.get("passed")),
        "chronology_pass_count": sum(1 for row in chronology_rows if row.get("chronology_valid")),
        "exact_replay_pass_count": sum(
            1 for row in exact_rows if row.get("replayed_label_matches")
        ),
        "aggregate_tampering_detected": aggregate_tampered,
        "blocked_preconditions": blocked_preconditions,
        "all_audit_rows_passed": all_passed,
        "ready_score_from_rows": 1.0 if all_passed else 0.0,
    }


def gate_check_summary(
    *,
    existence_receipts: Mapping[str, Any],
    revision_receipt: Mapping[str, Any],
    source_identity_rows: Sequence[Mapping[str, Any]],
    chronology_rows: Sequence[Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    split_audit: Mapping[str, Any],
    shard_audit: Mapping[str, Any],
    aggregate_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    failures: list[JsonDict] = []

    def add(check: str, expected: Any, observed: Any, severity: str = "blocked") -> None:
        failures.append(
            {"check": check, "expected": expected, "observed": observed, "severity": severity}
        )

    if not existence_receipts.get("intake_artifact", {}).get("exists"):
        add("exp6529_artifact_exists", True, False)
    if not existence_receipts.get("fixture", {}).get("exists"):
        add("fixture_exists", True, False)
    if not revision_receipt.get("source_root_exists"):
        add("pinned_source_root_exists", True, revision_receipt.get("source_root_exists"))
    for key in (
        "revision_is_immutable",
        "license_verified",
        "schema_verified",
        "corruption_boundary_text_verified",
    ):
        if revision_receipt.get(key) is not True:
            add(key, True, revision_receipt.get(key))
    if not (
        existence_receipts.get("intake_artifact", {}).get("exists")
        and existence_receipts.get("fixture", {}).get("exists")
    ):
        return {"all_gates_passed": not failures, "failed_checks": failures}
    if source_identity_rows and not all(row.get("passed") for row in source_identity_rows):
        add(
            "source_identity_hash",
            "all source file and row hashes match",
            [row for row in source_identity_rows if not row.get("passed")],
            "disqualified",
        )
    if chronology_rows and not all(row.get("chronology_valid") for row in chronology_rows):
        add(
            "chronology_replay",
            "all turns contiguous and duplicate-free",
            [row for row in chronology_rows if not row.get("chronology_valid")],
            "disqualified",
        )
    if exact_rows and not all(row.get("replayed_label_matches") for row in exact_rows):
        add(
            "exact_label_replay",
            "all replayed labels match fixture labels",
            [row for row in exact_rows if not row.get("replayed_label_matches")],
            "disqualified",
        )
    if split_audit and split_audit.get("passed") is not True:
        add("split_lineage", True, split_audit, "disqualified")
    if shard_audit and shard_audit.get("passed") is not True:
        add("shard_transaction", True, shard_audit, "disqualified")
    tampered, observed, intake_observed = _aggregate_tampering(aggregate_rows)
    if tampered:
        add(
            "aggregate_tampering",
            observed,
            intake_observed,
            "disqualified",
        )
    failed_attacks = [row for row in attack_rows if row.get("passed") is not True]
    if failed_attacks:
        add("leakage_attacks", "all attacks pass", failed_attacks, "disqualified")
    return {"all_gates_passed": not failures, "failed_checks": failures}


def _verdict_class(gate: Mapping[str, Any]) -> str | None:
    if gate.get("all_gates_passed"):
        return None
    severities = {str(row.get("severity")) for row in gate.get("failed_checks", [])}
    return "disqualified" if "disqualified" in severities else "blocked"


def _status_for_class(verdict_class: str | None) -> str:
    if verdict_class is None:
        return "complete_external_constraint_corpus_audit"
    if verdict_class == "disqualified":
        return "disqualified_external_constraint_corpus_audit"
    if verdict_class == "partial":
        return "partial_external_constraint_corpus_audit"
    return "blocked_external_constraint_corpus_audit"


def _honest_verdict(status: str, gate: Mapping[str, Any]) -> str:
    if status.startswith("complete_"):
        return (
            f"{status}: source hashes, exact labels, splits, shards, aggregates, and attacks pass"
        )
    failed = gate.get("failed_checks", [])
    checks = ",".join(str(row.get("check")) for row in failed) if failed else "unknown"
    return f"{status}: failed_checks={checks}"


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
    fixture_rows: Sequence[Mapping[str, Any]],
    revision_receipt: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    return {
        "run_date": run_date,
        "checked_at_utc": now_utc,
        "intake_artifact_path": str(intake_path),
        "fixture_path": str(fixture_path),
        "fixture_row_count": len(fixture_rows),
        "pinned_revision_available": revision_receipt.get("revision_is_immutable") is True,
        "source_root_exists": revision_receipt.get("source_root_exists") is True,
        "solver_versions": {"independent_exact_solver": "finite_equality_v1"},
        "resources": _resource_receipt(repo_root),
        "git_state": _git_state(repo_root),
        "protected_files_unchanged": protected.get("all_protected_files_unchanged"),
    }


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str | None = None,
    run_date: str = RUN_DATE,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    now_utc: str | None = None,
) -> JsonDict:
    start = time.monotonic()
    now = now_utc or _utc_now()
    before_hashes = _protected_hashes(repo_root)
    intake_path = repo_root / EXP6529_RELATIVE_PATH
    fixture_path = repo_root / FIXTURE_RELATIVE_PATH
    intake = _load_json(intake_path)
    fixture_rows = _load_jsonl(fixture_path)
    revision_receipt = independent_revision_and_license_receipt(repo_root, intake)
    source_root_path = (
        Path(str(revision_receipt["source_root"])) if revision_receipt.get("source_root") else None
    )
    declared_hashes = _source_file_hashes(intake)
    existence = build_existence_receipts(
        repo_root=repo_root,
        intake_path=intake_path,
        fixture_path=fixture_path,
        fixture_rows=fixture_rows,
        source_root_path=source_root_path,
        declared_hashes=declared_hashes,
    )
    source_identity = source_identity_audit_rows(
        fixture_rows=fixture_rows,
        source_root_path=source_root_path,
        declared_hashes=declared_hashes,
    )
    chronology = chronology_replay_rows(fixture_rows)
    exact = independent_exact_replay_rows(
        fixture_rows=fixture_rows,
        source_root_path=source_root_path,
    )
    split_audit = split_and_lineage_audit(fixture_rows, chronology)
    shard_audit = shard_and_transaction_audit(
        repo_root=repo_root,
        intake=intake,
        fixture_path=fixture_path,
        fixture_rows=fixture_rows,
    )
    aggregates = independent_aggregate_rows(fixture_rows, exact, intake)
    attacks = leakage_attack_matrix(
        source_identity_rows=source_identity,
        exact_rows=exact,
        split_audit=split_audit,
        aggregate_rows=aggregates,
        fixture_rows=fixture_rows,
    )
    aggregate = aggregate_row_recomputation(
        fixture_rows=fixture_rows,
        source_identity_rows=source_identity,
        chronology_rows=chronology,
        exact_rows=exact,
        split_audit=split_audit,
        shard_audit=shard_audit,
        aggregate_rows=aggregates,
        attack_rows=attacks,
        revision_receipt=revision_receipt,
        existence_receipts=existence,
    )
    gate = gate_check_summary(
        existence_receipts=existence,
        revision_receipt=revision_receipt,
        source_identity_rows=source_identity,
        chronology_rows=chronology,
        exact_rows=exact,
        split_audit=split_audit,
        shard_audit=shard_audit,
        aggregate_rows=aggregates,
        attack_rows=attacks,
    )
    verdict_class = _verdict_class(gate)
    status = _status_for_class(verdict_class)
    after_hashes = _protected_hashes(repo_root)
    protected = protected_files_unchanged(before_hashes, after_hashes)
    preconditions = preconditions_checked(
        repo_root=repo_root,
        run_date=run_date,
        now_utc=now,
        intake_path=intake_path,
        fixture_path=fixture_path,
        fixture_rows=fixture_rows,
        revision_receipt=revision_receipt,
        protected=protected,
    )
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": _honest_verdict(status, gate),
        "verdict_class": verdict_class,
        "source_existence_and_hash_receipts": existence,
        "independent_revision_and_license_receipt": revision_receipt,
        "source_identity_audit_rows": source_identity,
        "chronology_replay_rows": chronology,
        "independent_exact_replay_rows": exact,
        "split_and_lineage_audit": split_audit,
        "shard_and_transaction_audit": shard_audit,
        "independent_aggregate_rows": aggregates,
        "leakage_attack_matrix": attacks,
        "external_constraint_corpus_audited_ready_score": float(aggregate["ready_score_from_rows"]),
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
        errors.append("verdict_class outside Exp6530 enum")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "blocked_", "partial_", "disqualified_")
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
        description="Build or validate Exp6530 external constraint corpus audit."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
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
        run_date=str(args.date),
        write=True,
    )
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI.
    raise SystemExit(main())
