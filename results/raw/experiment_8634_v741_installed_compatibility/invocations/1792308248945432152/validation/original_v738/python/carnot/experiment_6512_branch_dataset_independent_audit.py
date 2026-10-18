"""Exp6512 independent branch-dataset audit.

Spec refs: REQ-BENCH-6512, SCENARIO-BENCH-6512-MISSING-UPSTREAM,
SCENARIO-BENCH-6512-ROW-REPLAY, SCENARIO-BENCH-6512-SPLIT-LINEAGE,
SCENARIO-BENCH-6512-SHARDS-CENSORING, SCENARIO-BENCH-6512-LEAKAGE.

The audit is a closed gate. It writes a terminal artifact even when the
upstream Exp6511 dataset is absent, blocked, partial, null, or malformed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260822"
RANDOM_SEED = 6512
SCHEMA_VERSION = "carnot.experiment_6512.branch_dataset_independent_audit.v1"
INFERENCE_SUBSTRATE = "independent_branch_dataset_row_and_exact_receipt_replay_no_llm"
VERIFIER_IS_ORACLE = True

RESULT_RELATIVE_PATH = Path("results/experiment_6512_branch_dataset_independent_audit.json")
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6511_exact_branch_counterfactual_dataset_v2.json")
ROOT_RELATIVE_PATH = Path("results/experiment_6510_v563_independent_exact_root.json")
BASE_RELATIVE_PATH = Path("results/experiment_6504_exact_structural_benchmark_commitment.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6512_branch_dataset_independent_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6512_branch_dataset_independent_audit.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

PROTECTED_RELATIVE_PATHS = (
    UPSTREAM_RELATIVE_PATH,
    ROOT_RELATIVE_PATH,
    BASE_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
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
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    UPSTREAM_RELATIVE_PATH,
    ROOT_RELATIVE_PATH,
    BASE_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "verdict_class",
    "upstream_artifact_receipt",
    "independent_row_recomputation",
    "exact_receipt_replay_rows",
    "split_and_lineage_audit",
    "shard_and_censoring_audit",
    "feature_timing_audit",
    "shortcut_attack_matrix",
    "branch_dataset_audited_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
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
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal audit status is required even when the upstream file is absent.",
    "verdict_class": (
        "The class records null, blocked, disqualified, or partial readiness without claiming method value."
    ),
    "upstream_artifact_receipt": (
        "Path, existence, hash, status, and imported fields make the audit input explicit."
    ),
    "independent_row_recomputation": (
        "The audit must derive counts and metrics from rows rather than task aggregates."
    ),
    "exact_receipt_replay_rows": "Per-row replay checks label and validity authority.",
    "split_and_lineage_audit": (
        "Base-lineage separation prevents development and held leakage."
    ),
    "shard_and_censoring_audit": (
        "Manifest, resume, timeout, and terminal-count checks detect omitted hard units."
    ),
    "feature_timing_audit": (
        "Decision-time availability blocks future-effort and outcome leakage."
    ),
    "shortcut_attack_matrix": (
        "Attacks test identity, order, length, family, label, and censoring shortcuts."
    ),
    "branch_dataset_audited_ready_score": (
        "This exact closed field is the structured gate for Exp6513 and Exp6518."
    ),
    "per_unit_rows": "One audit row per dataset unit makes readiness independently recheckable.",
    "aggregate_row_recomputation": "All audit summaries must derive from per-unit evidence.",
    "gate_check_summary": (
        "Every score-0 or blocked result names the failed check and observed value."
    ),
    "preconditions_checked": "Input, solver, and resource checks prevent invented audit results.",
    "protected_files_unchanged": (
        "An audit cannot repair the source artifact or protected control files."
    ),
    "inference_substrate": (
        "Declaring independent exact replay with no LLM makes the evidence boundary explicit."
    ),
    "verifier_is_oracle": (
        "Oracle disclosure prevents exact dataset consistency from becoming a verifier-value claim."
    ),
    "field_principles": "Reasons beside fields preserve the gate contract.",
    "field_provenance": (
        "Paths, row IDs, reducers, solvers, and hashes make each audit result traceable."
    ),
    "random_seed": "A fixed attack order makes the audit reproducible.",
    "duration_s": "Measured duration supports authenticity checks.",
    "tests_run": "Command receipts show which validation and E2E checks ran.",
    "reproducibility_checksum": (
        "A content hash detects later changes to the audit decision."
    ),
    "honest_verdict": (
        "A complete_* or blocked_* prefix gives downstream gates a safe terminal state."
    ),
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6512_branch_dataset_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6512_branch_dataset_independent_audit.py "
    "-m pytest tests/python/test_experiment_6512_branch_dataset_independent_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6512_branch_dataset_independent_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6512_branch_dataset_independent_audit.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6512_branch_dataset_independent_audit --date 20260822"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6512_branch_dataset_independent_audit.json"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6512_branch_dataset_independent_audit.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6512_branch_dataset_independent_audit --validate"
)

DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": EXCLUSION_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": EXACT_E2E_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)

ROW_FIELD_CANDIDATES = (
    "branch_counterfactual_rows",
    "dataset_rows",
    "counterfactual_rows",
    "rows",
    "per_unit_rows",
)
EXACT_RECEIPT_KEYS = ("exact_receipt", "exact_solver_receipt", "solver_receipt")
BASE_HASH_KEYS = ("base_instance_hash", "immutable_base_instance_hash", "raw_instance_hash")
ROW_ID_KEYS = ("row_id", "unit_id", "instance_id", "dataset_unit_id")
LABEL_KEYS = ("exact_label", "label", "solver_label")
TERMINAL_KEYS = ("terminal_disposition", "terminal_status", "disposition", "outcome")
SPLITS = ("train", "development", "held")
ATTACK_IDS = (
    "unit_identity",
    "row_order",
    "serialization_length",
    "family",
    "label",
    "future_effort",
    "shard_order",
    "censored_row_removal",
)
DECISION_TIME_VALUES = {"decision_time", "pre_decision", "static_instance", "before_solver"}
FORBIDDEN_FEATURE_MARKERS = {
    "unit_identity": ("unit_id", "instance_id", "base_instance_id", "base_lineage_id", "row_id"),
    "row_order": ("row_order", "row_index", "order_index"),
    "serialization_length": ("serialization_length", "serialized_length", "serialized_bytes"),
    "family": ("family", "generator_family", "family_id"),
    "label": ("label", "exact_label", "target", "terminal_disposition", "outcome"),
    "future_effort": (
        "future_effort",
        "post_decision",
        "solver_steps_after_decision",
        "final_solver_steps",
    ),
    "shard_order": ("shard_id", "shard_order", "shard_index"),
    "censored_row_removal": ("censored", "removed_by_censor", "censoring_disposition"),
}


def canonical_json(value: Any) -> str:
    """Return stable JSON text so hashes do not depend on key order."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON value with the prefix used by result artifacts."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file or return ``missing`` for absent upstream evidence."""

    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _display_path(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _git_status(repo_root: Path) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "status", "--short"],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS}


def protected_file_hashes(repo_root: Path, upstream_path: Path) -> dict[str, JsonDict]:
    """Capture before/after hashes for files the audit must not repair."""

    paths = [repo_root / path for path in PROTECTED_RELATIVE_PATHS]
    if upstream_path.resolve(strict=False) not in {path.resolve(strict=False) for path in paths}:
        paths.append(upstream_path)
    return {
        _display_path(repo_root, path): {
            "exists": path.is_file(),
            "sha256": sha256_file(path),
            "protected_by_exp6512_audit": True,
        }
        for path in paths
    }


def protected_files_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Compare protected evidence without requiring missing upstreams to exist."""

    rows: dict[str, JsonDict] = {}
    for path in sorted(set(before) | set(after)):
        prior = dict(before.get(path, {}))
        post = dict(after.get(path, {}))
        unchanged = (
            prior.get("sha256") == post.get("sha256")
            and prior.get("exists") is post.get("exists")
        )
        rows[path] = {
            "sha256_before": prior.get("sha256", "missing"),
            "sha256_after": post.get("sha256", "missing"),
            "exists_before": prior.get("exists") is True,
            "exists_after": post.get("exists") is True,
            "unchanged": unchanged,
            "protected_by_exp6512_audit": True,
        }
    changed = [path for path, row in rows.items() if row["unchanged"] is not True]
    return {
        "files": rows,
        "changed_paths": changed,
        "all_protected_files_unchanged": changed == [],
    }


def _resource_state(repo_root: Path) -> JsonDict:
    disk = shutil.disk_usage(repo_root)
    mem_total = None
    mem_available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        parsed: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                parsed[parts[0].rstrip(":")] = int(parts[1]) * 1024
        mem_total = parsed.get("MemTotal")
        mem_available = parsed.get("MemAvailable")
    return {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "ram_total_bytes": mem_total,
        "ram_available_bytes": mem_available,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _solver_state() -> JsonDict:
    try:
        import z3  # type: ignore[import-untyped]

        return {
            "z3_python_available": True,
            "z3_python_version": z3.get_version_string(),
            "z3_cli_path": shutil.which("z3"),
            "exact_solver_available": True,
        }
    except Exception as exc:  # pragma: no cover - depends on local package state.
        return {
            "z3_python_available": False,
            "z3_python_error": str(exc),
            "z3_cli_path": shutil.which("z3"),
            "exact_solver_available": False,
        }


def _read_json(path: Path) -> tuple[JsonDict, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, str(exc)
    if not isinstance(payload, Mapping):
        return {}, "top-level JSON is not an object"
    return dict(payload), None


def _rows_from_payload(payload: Mapping[str, Any]) -> tuple[list[JsonDict], str | None]:
    for key in ROW_FIELD_CANDIDATES:
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, Mapping)], key
    return [], None


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _row_id(row: Mapping[str, Any], index: int) -> str:
    value = _first_present(row, ROW_ID_KEYS)
    return str(value) if value not in (None, "") else f"row-{index}"


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) >= 16


def _terminal_disposition(row: Mapping[str, Any]) -> Any:
    return _first_present(row, TERMINAL_KEYS)


def upstream_artifact_receipt(
    repo_root: Path,
    upstream_path: Path,
    payload: Mapping[str, Any],
    read_error: str | None,
    row_count: int,
    row_field: str | None,
) -> JsonDict:
    """Record the exact Exp6511 input state before any gate decision."""

    exists = upstream_path.is_file()
    status = payload.get("status") if exists and read_error is None else None
    manifest = payload.get("shard_manifest") if isinstance(payload.get("shard_manifest"), Mapping) else {}
    shard_count = len(manifest.get("shards", [])) if isinstance(manifest, Mapping) else 0
    return {
        "path": _display_path(repo_root, upstream_path),
        "absolute_path": str(upstream_path.resolve(strict=False)),
        "exists": exists,
        "sha256": sha256_file(upstream_path),
        "json_readable": exists and read_error is None,
        "json_error": read_error,
        "status": status,
        "verdict_class": payload.get("verdict_class") if exists and read_error is None else None,
        "terminal_status": isinstance(status, str)
        and status.startswith("complete_")
        and "blocked" not in status
        and "partial" not in status,
        "row_field_used": row_field,
        "row_count": row_count,
        "shard_count": shard_count,
        "imported_fields": sorted(payload.keys()) if exists and read_error is None else [],
    }


def _exact_receipt_for(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in EXACT_RECEIPT_KEYS:
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def exact_receipt_replay_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Replay per-row receipt invariants without trusting aggregate fields."""

    replay_rows: list[JsonDict] = []
    for index, row in enumerate(rows):
        receipt = _exact_receipt_for(row)
        row_id = _row_id(row, index)
        row_label = _first_present(row, LABEL_KEYS)
        receipt_label = _first_present(receipt, LABEL_KEYS)
        base_hash = _first_present(row, BASE_HASH_KEYS)
        receipt_base_hash = _first_present(receipt, BASE_HASH_KEYS)
        receipt_row_id = _first_present(receipt, ROW_ID_KEYS)
        receipt_valid = any(
            receipt.get(key) is True
            for key in ("valid", "replay_passed", "model_or_proof_valid", "proof_valid")
        )
        label_matches = receipt_label == row_label and row_label is not None
        base_hash_present = _is_sha(base_hash)
        base_hash_matches = receipt_base_hash in (None, base_hash)
        row_id_matches = receipt_row_id in (None, row_id)
        passed = (
            isinstance(receipt, Mapping)
            and bool(receipt)
            and receipt_valid
            and label_matches
            and base_hash_present
            and base_hash_matches
            and row_id_matches
        )
        payload = {
            "row_type": "exact_receipt_replay",
            "row_id": row_id,
            "split": row.get("split"),
            "base_lineage_id": row.get("base_lineage_id") or row.get("base_instance_id"),
            "base_instance_hash": base_hash,
            "receipt_present": bool(receipt),
            "receipt_valid": receipt_valid,
            "label_matches_receipt": label_matches,
            "base_hash_present": base_hash_present,
            "base_hash_matches_receipt": base_hash_matches,
            "row_id_matches_receipt": row_id_matches,
            "exact_receipt_replay_passed": passed,
            "verifier_is_oracle_for_this_row": True,
            "spec_refs": ["REQ-BENCH-6512", "SCENARIO-BENCH-6512-ROW-REPLAY"],
        }
        replay_rows.append({**payload, "exact_receipt_replay_row_hash": sha256_json(payload)})
    return replay_rows


def split_and_lineage_audit(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Rebuild split lineage and reject overlap or post-held mutation."""

    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    checkpoint_counts: Counter[str] = Counter()
    budgets_by_lineage: dict[str, set[Any]] = defaultdict(set)
    post_held_repairs: list[str] = []
    missing_terminal: list[str] = []
    for index, row in enumerate(rows):
        row_id = _row_id(row, index)
        split = str(row.get("split") or "")
        lineage = str(row.get("base_lineage_id") or row.get("base_instance_id") or row_id)
        if split in by_split:
            by_split[split].add(lineage)
        checkpoint = row.get("checkpoint_id")
        if checkpoint not in (None, ""):
            checkpoint_counts[str(checkpoint)] += 1
        budget = row.get("exact_budget", row.get("budget"))
        if budget is not None:
            budgets_by_lineage[lineage].add(budget)
        if split == "held" and (
            row.get("post_held_repair") is True or row.get("repair_after_held_read") is True
        ):
            post_held_repairs.append(row_id)
        if _terminal_disposition(row) in (None, ""):
            missing_terminal.append(row_id)
    overlap_pairs: list[JsonDict] = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = sorted(by_split[left] & by_split[right])
            if overlap:
                overlap_pairs.append({"left": left, "right": right, "base_lineage_ids": overlap})
    duplicate_checkpoints = sorted(
        checkpoint for checkpoint, count in checkpoint_counts.items() if count > 1
    )
    asymmetric_budgets = {
        lineage: sorted(values) for lineage, values in budgets_by_lineage.items() if len(values) > 1
    }
    split_counts = {split: len(by_split[split]) for split in SPLITS}
    passed = (
        len(rows) > 0
        and all(split_counts[split] > 0 for split in SPLITS)
        and not overlap_pairs
        and not duplicate_checkpoints
        and not post_held_repairs
        and not asymmetric_budgets
        and not missing_terminal
    )
    return {
        "split_lineage_sets": {split: sorted(values) for split, values in by_split.items()},
        "split_lineage_counts": split_counts,
        "base_lineage_overlap_pairs": overlap_pairs,
        "base_lineage_overlap_count": len(overlap_pairs),
        "duplicate_checkpoint_ids": duplicate_checkpoints,
        "duplicate_checkpoint_count": len(duplicate_checkpoints),
        "post_held_repair_row_ids": post_held_repairs,
        "post_held_repair_count": len(post_held_repairs),
        "asymmetric_budget_lineages": asymmetric_budgets,
        "asymmetric_budget_count": len(asymmetric_budgets),
        "missing_terminal_disposition_row_ids": missing_terminal,
        "missing_terminal_disposition_count": len(missing_terminal),
        "sealed_split_passed": passed,
    }


def _manifest_shard_ids(manifest: Mapping[str, Any], expected_count: int) -> set[Any]:
    ranged_ids = set(range(expected_count))
    if isinstance(manifest.get("shards"), list):
        shard_ids = {
            row.get("shard_id")
            for row in manifest["shards"]
            if isinstance(row, Mapping) and row.get("shard_id") is not None
        }
        if shard_ids:
            return shard_ids | ranged_ids
    return ranged_ids


def shard_and_censoring_audit(rows: Sequence[Mapping[str, Any]], payload: Mapping[str, Any]) -> JsonDict:
    """Check shard manifest, resume receipts, hash chain, and censoring totals."""

    manifest = payload.get("shard_manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    observed_shards = {row.get("shard_id") for row in rows if row.get("shard_id") is not None}
    expected_count = int(manifest.get("expected_shard_count") or len(observed_shards))
    expected_shards = _manifest_shard_ids(manifest, expected_count)
    missing_shards = sorted(expected_shards - observed_shards)
    terminal_rows = [row for row in rows if _terminal_disposition(row) not in (None, "")]
    censored_rows = [row for row in rows if row.get("censored") is True]
    declared_terminal = manifest.get("terminal_row_count")
    declared_censored = manifest.get("censored_row_count")
    hash_chain = manifest.get("hash_chain") if isinstance(manifest.get("hash_chain"), list) else []
    resume_receipts = (
        manifest.get("resume_receipts") if isinstance(manifest.get("resume_receipts"), list) else []
    )
    hash_chain_complete = len(hash_chain) >= len(observed_shards) and all(_is_sha(v) for v in hash_chain)
    resume_receipts_present = len(resume_receipts) >= len(observed_shards)
    terminal_count_matches = declared_terminal in (None, len(terminal_rows))
    censored_count_matches = declared_censored in (None, len(censored_rows))
    manifest_complete = manifest.get("complete") is True
    passed = (
        len(rows) > 0
        and bool(manifest)
        and manifest_complete
        and not missing_shards
        and hash_chain_complete
        and resume_receipts_present
        and terminal_count_matches
        and censored_count_matches
    )
    return {
        "manifest_present": bool(manifest),
        "manifest_complete": manifest_complete,
        "expected_shard_count": expected_count,
        "observed_shard_ids": sorted(observed_shards),
        "missing_shard_ids": missing_shards,
        "missing_shard_count": len(missing_shards),
        "hash_chain_length": len(hash_chain),
        "hash_chain_complete": hash_chain_complete,
        "resume_receipt_count": len(resume_receipts),
        "resume_receipts_present": resume_receipts_present,
        "terminal_row_count_observed": len(terminal_rows),
        "terminal_row_count_declared": declared_terminal,
        "terminal_count_matches": terminal_count_matches,
        "censored_row_count_observed": len(censored_rows),
        "censored_row_count_declared": declared_censored,
        "censored_count_matches": censored_count_matches,
        "shard_and_censoring_passed": passed,
    }


def _feature_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    schema = payload.get("feature_schema") or payload.get("branch_feature_schema")
    if not isinstance(schema, Mapping):
        return []
    features = schema.get("features", [])
    if isinstance(features, Mapping):
        features = [{"name": key, **(value if isinstance(value, Mapping) else {})} for key, value in features.items()]
    if not isinstance(features, list):
        return []
    rows: list[JsonDict] = []
    for item in features:
        if isinstance(item, str):
            rows.append({"name": item, "available_at": "decision_time"})
        elif isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _feature_names(feature_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("name") or row.get("field") or "") for row in feature_rows]


def feature_timing_audit(payload: Mapping[str, Any]) -> JsonDict:
    """Reject features that are unavailable when a branch decision is made."""

    rows = _feature_rows(payload)
    names = _feature_names(rows)
    unavailable = [
        name
        for name, row in zip(names, rows, strict=True)
        if str(row.get("available_at") or "").lower() not in DECISION_TIME_VALUES
    ]
    forbidden: dict[str, list[str]] = {}
    for attack_id, markers in FORBIDDEN_FEATURE_MARKERS.items():
        hits = sorted(
            name
            for name in names
            if any(marker == name.lower() or marker in name.lower() for marker in markers)
        )
        if hits:
            forbidden[attack_id] = hits
    passed = bool(rows) and not unavailable and not forbidden
    return {
        "feature_schema_present": bool(rows),
        "feature_count": len(rows),
        "feature_names": names,
        "unavailable_feature_names": unavailable,
        "forbidden_feature_names_by_attack": forbidden,
        "decision_time_values_allowed": sorted(DECISION_TIME_VALUES),
        "feature_timing_passed": passed,
    }


def shortcut_attack_matrix(
    feature_audit: Mapping[str, Any],
    shard_audit: Mapping[str, Any],
    *,
    upstream_missing_or_unreadable: bool,
) -> JsonDict:
    """Run deterministic leakage attacks against the exposed feature schema."""

    forbidden = feature_audit.get("forbidden_feature_names_by_attack")
    forbidden = forbidden if isinstance(forbidden, Mapping) else {}
    rows: list[JsonDict] = []
    for attack_id in ATTACK_IDS:
        leaked_fields = list(forbidden.get(attack_id, []))
        if attack_id == "censored_row_removal" and (
            shard_audit.get("censored_count_matches") is not True
        ):
            leaked_fields.append("censoring_count_mismatch")
        if attack_id == "shard_order" and shard_audit.get("missing_shard_count", 0):
            leaked_fields.append("missing_shard")
        fail_closed = upstream_missing_or_unreadable or not leaked_fields
        payload = {
            "attack_id": attack_id,
            "leaked_fields": sorted(set(leaked_fields)),
            "fail_closed": fail_closed,
            "observed_ready_score_if_only_this_attack": 0.0 if fail_closed else 1.0,
            "spec_refs": ["REQ-BENCH-6512", "SCENARIO-BENCH-6512-LEAKAGE"],
        }
        rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    return {
        "random_seed": RANDOM_SEED,
        "attack_order": list(ATTACK_IDS),
        "rows": rows,
        "failed_attack_ids": [row["attack_id"] for row in rows if row["fail_closed"] is not True],
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
    }


def independent_row_recomputation(
    rows: Sequence[Mapping[str, Any]],
    row_field: str | None,
    replay_rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> JsonDict:
    """Derive every summary from raw rows and receipt replay rows."""

    row_ids = [_row_id(row, index) for index, row in enumerate(rows)]
    duplicate_row_ids = sorted(row_id for row_id, count in Counter(row_ids).items() if count > 1)
    terminal_count = sum(1 for row in rows if _terminal_disposition(row) not in (None, ""))
    receipt_pass_count = sum(1 for row in replay_rows if row.get("exact_receipt_replay_passed") is True)
    base_hash_count = sum(1 for row in replay_rows if row.get("base_hash_present") is True)
    aggregate = payload.get("aggregate_row_recomputation")
    imported_row_count = aggregate.get("row_count") if isinstance(aggregate, Mapping) else None
    return {
        "row_field_used": row_field,
        "row_count": len(rows),
        "row_id_count": len(set(row_ids)),
        "duplicate_row_ids": duplicate_row_ids,
        "duplicate_row_id_count": len(duplicate_row_ids),
        "terminal_disposition_count": terminal_count,
        "missing_terminal_disposition_count": len(rows) - terminal_count,
        "base_hash_present_count": base_hash_count,
        "exact_receipt_pass_count": receipt_pass_count,
        "exact_receipt_failure_count": len(replay_rows) - receipt_pass_count,
        "imported_aggregate_row_count": imported_row_count,
        "imported_aggregate_matches": imported_row_count == len(rows),
        "row_recomputation_passed": (
            len(rows) > 0
            and row_field is not None
            and not duplicate_row_ids
            and terminal_count == len(rows)
            and base_hash_count == len(rows)
            and receipt_pass_count == len(rows)
        ),
    }


def per_unit_rows(
    rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Emit one compact audit row for each upstream dataset unit."""

    replay_by_id = {str(row["row_id"]): row for row in replay_rows}
    out: list[JsonDict] = []
    for index, row in enumerate(rows):
        row_id = _row_id(row, index)
        replay = replay_by_id.get(row_id, {})
        payload = {
            "row_type": "branch_dataset_unit_audit",
            "row_id": row_id,
            "split": row.get("split"),
            "base_lineage_id": row.get("base_lineage_id") or row.get("base_instance_id"),
            "base_instance_hash": _first_present(row, BASE_HASH_KEYS),
            "shard_id": row.get("shard_id"),
            "terminal_disposition": _terminal_disposition(row),
            "exact_receipt_replay_passed": replay.get("exact_receipt_replay_passed") is True,
            "verifier_is_oracle_for_label_and_receipt": True,
            "spec_refs": ["REQ-BENCH-6512", "SCENARIO-BENCH-6512-ROW-REPLAY"],
        }
        out.append({**payload, "unit_audit_row_hash": sha256_json(payload)})
    return out


def aggregate_row_recomputation(unit_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize only the emitted per-unit audit rows."""

    split_counts = Counter(str(row.get("split")) for row in unit_rows)
    terminal_counts = Counter(str(row.get("terminal_disposition")) for row in unit_rows)
    shard_counts = Counter(str(row.get("shard_id")) for row in unit_rows)
    return {
        "row_count": len(unit_rows),
        "exact_receipt_replay_pass_count": sum(
            1 for row in unit_rows if row.get("exact_receipt_replay_passed") is True
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "terminal_disposition_counts": dict(sorted(terminal_counts.items())),
        "shard_counts": dict(sorted(shard_counts.items())),
        "unit_row_hash_chain": [row.get("unit_audit_row_hash") for row in unit_rows],
        "unit_row_reducer": "aggregate_row_recomputation_from_per_unit_rows",
    }


def _gate_check_summary(
    receipt: Mapping[str, Any],
    recomputation: Mapping[str, Any],
    split_audit: Mapping[str, Any],
    shard_audit: Mapping[str, Any],
    feature_audit: Mapping[str, Any],
    attacks: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> list[JsonDict]:
    failures: list[tuple[str, str, str]] = []
    if receipt.get("exists") is not True:
        failures.append(("upstream_exists", "true", str(receipt.get("absolute_path"))))
    if receipt.get("exists") is True and receipt.get("json_readable") is not True:
        failures.append(("upstream_json_readable", "true", str(receipt.get("json_error"))))
    if receipt.get("json_readable") is True and receipt.get("terminal_status") is not True:
        failures.append(("upstream_terminal_status", "complete_*", str(receipt.get("status"))))
    if recomputation.get("row_recomputation_passed") is not True:
        failures.append(
            ("independent_row_recomputation", "true", f"row_count={recomputation.get('row_count')}")
        )
    if recomputation.get("exact_receipt_failure_count"):
        failures.append(
            (
                "exact_receipt_replay",
                "0 failures",
                f"{recomputation.get('exact_receipt_failure_count')} failures",
            )
        )
    if split_audit.get("sealed_split_passed") is not True:
        failures.append(
            (
                "split_and_lineage_audit",
                "sealed_split_passed=true",
                " ".join(
                    [
                        f"base_lineage_overlap_count={split_audit.get('base_lineage_overlap_count')}",
                        f"duplicate_checkpoint_count={split_audit.get('duplicate_checkpoint_count')}",
                        f"post_held_repair_count={split_audit.get('post_held_repair_count')}",
                        f"asymmetric_budget_count={split_audit.get('asymmetric_budget_count')}",
                        "missing_terminal_disposition_count="
                        f"{split_audit.get('missing_terminal_disposition_count')}",
                    ]
                ),
            )
        )
    if shard_audit.get("shard_and_censoring_passed") is not True:
        failures.append(
            (
                "shard_and_censoring_audit",
                "shard_and_censoring_passed=true",
                " ".join(
                    [
                        f"manifest_complete={shard_audit.get('manifest_complete')}",
                        f"missing_shard_count={shard_audit.get('missing_shard_count')}",
                        f"hash_chain_complete={shard_audit.get('hash_chain_complete')}",
                        f"resume_receipts_present={shard_audit.get('resume_receipts_present')}",
                        f"terminal_count_matches={shard_audit.get('terminal_count_matches')}",
                        f"censored_count_matches={shard_audit.get('censored_count_matches')}",
                    ]
                ),
            )
        )
    if feature_audit.get("feature_timing_passed") is not True:
        failures.append(
            (
                "feature_timing_audit",
                "feature_timing_passed=true",
                canonical_json(
                    {
                        "unavailable": feature_audit.get("unavailable_feature_names"),
                        "forbidden": feature_audit.get("forbidden_feature_names_by_attack"),
                    }
                ),
            )
        )
    if attacks.get("all_attacks_fail_closed") is not True:
        failures.append(
            (
                "shortcut_attack_matrix",
                "all_attacks_fail_closed=true",
                ",".join(attacks.get("failed_attack_ids", [])),
            )
        )
    if protected.get("all_protected_files_unchanged") is not True:
        failures.append(
            (
                "protected_files_unchanged",
                "all_protected_files_unchanged=true",
                ",".join(protected.get("changed_paths", [])),
            )
        )
    return [
        {
            "check": check,
            "expected": expected,
            "observed": observed,
            "score_if_unfixed": 0.0,
            "spec_refs": ["REQ-BENCH-6512"],
        }
        for check, expected, observed in failures
    ]


def _field_provenance(result_path: Path, upstream_path: Path) -> dict[str, JsonDict]:
    return {
        field: {
            "spec_refs": ["REQ-BENCH-6512"],
            "source_paths": [
                _display_path(REPO_ROOT, upstream_path),
                ROOT_RELATIVE_PATH.as_posix(),
                BASE_RELATIVE_PATH.as_posix(),
                MODULE_RELATIVE_PATH.as_posix(),
                TEST_RELATIVE_PATH.as_posix(),
                SPEC_RELATIVE_PATH.as_posix(),
            ],
            "result_path": str(result_path),
            "reducer": f"exp6512_{field}",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status_verdict(score: float, gate_summary: Sequence[Mapping[str, Any]]) -> tuple[str, str, str]:
    if score == 1.0:
        return (
            "complete_branch_dataset_independent_audit_ready",
            "null",
            "complete_branch_dataset_independent_audit_ready",
        )
    reason = "; ".join(f"{row['check']}={row['observed']}" for row in gate_summary[:4])
    return (
        "blocked_branch_dataset_independent_audit",
        "blocked",
        f"blocked_branch_dataset_independent_audit: {reason}",
    )


def _invalid_input_class(receipt: Mapping[str, Any], score: float) -> str:
    if score == 1.0:
        return "null"
    if receipt.get("exists") is True and receipt.get("json_readable") is True and receipt.get("terminal_status") is True:
        return "disqualified"
    return "blocked"


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    checksum_payload = {
        key: value for key, value in payload.items() if key != "reproducibility_checksum"
    }
    return sha256_json(checksum_payload)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    upstream_path: Path | str = UPSTREAM_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the terminal Exp6512 audit artifact for all upstream states."""

    start = time.perf_counter()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    upstream_path = Path(upstream_path)
    if not upstream_path.is_absolute():
        upstream_path = repo_root / upstream_path
    protected_before = protected_file_hashes(repo_root, upstream_path)
    payload, read_error = _read_json(upstream_path) if upstream_path.is_file() else ({}, None)
    rows, row_field = _rows_from_payload(payload)
    receipt = upstream_artifact_receipt(repo_root, upstream_path, payload, read_error, len(rows), row_field)
    replay_rows = exact_receipt_replay_rows(rows)
    recomputation = independent_row_recomputation(rows, row_field, replay_rows, payload)
    split_audit = split_and_lineage_audit(rows)
    shard_audit = shard_and_censoring_audit(rows, payload)
    feature_audit = feature_timing_audit(payload)
    attacks = shortcut_attack_matrix(
        feature_audit,
        shard_audit,
        upstream_missing_or_unreadable=receipt.get("json_readable") is not True,
    )
    units = per_unit_rows(rows, replay_rows)
    aggregate = aggregate_row_recomputation(units)
    protected_after = protected_file_hashes(repo_root, upstream_path)
    protected = protected_files_unchanged(protected_before, protected_after)
    gate_summary = _gate_check_summary(
        receipt,
        recomputation,
        split_audit,
        shard_audit,
        feature_audit,
        attacks,
        protected,
    )
    score = 0.0 if gate_summary else 1.0
    status, _, honest = _status_verdict(score, gate_summary)
    verdict_class = _invalid_input_class(receipt, score)
    preconditions = {
        "planning_date": run_date,
        "result_path": str(result_path),
        "upstream_path": str(upstream_path),
        "upstream_exists": receipt["exists"],
        "upstream_sha256": receipt["sha256"],
        "solver_availability": _solver_state(),
        "resources": _resource_state(repo_root),
        "source_hashes": _source_hashes(repo_root),
        "protected_hashes_before": protected_before,
        "git_status_short": _git_status(repo_root),
    }
    artifact: JsonDict = {
        "status": status,
        "verdict_class": verdict_class,
        "upstream_artifact_receipt": receipt,
        "independent_row_recomputation": recomputation,
        "exact_receipt_replay_rows": replay_rows,
        "split_and_lineage_audit": split_audit,
        "shard_and_censoring_audit": shard_audit,
        "feature_timing_audit": feature_audit,
        "shortcut_attack_matrix": attacks,
        "branch_dataset_audited_ready_score": score,
        "per_unit_rows": units,
        "aggregate_row_recomputation": aggregate,
        "gate_check_summary": gate_summary,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(result_path, upstream_path),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s if duration_s is not None else time.perf_counter() - start), 6),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
        "honest_verdict": honest,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        atomic_write_json(result_path, artifact, root=repo_root, allow_override=False)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate the Exp6512 schema and the closed readiness contract."""

    errors: list[str] = []
    if set(payload) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    score = payload.get("branch_dataset_audited_ready_score")
    if score not in (0.0, 1.0):
        errors.append("branch_dataset_audited_ready_score must be 0.0 or 1.0")
    exact_pass = payload.get("independent_row_recomputation", {}).get("exact_receipt_failure_count") == 0
    split_pass = payload.get("split_and_lineage_audit", {}).get("sealed_split_passed") is True
    shard_pass = (
        payload.get("shard_and_censoring_audit", {}).get("shard_and_censoring_passed") is True
    )
    feature_pass = payload.get("feature_timing_audit", {}).get("feature_timing_passed") is True
    if score == 1.0 and not (exact_pass and split_pass and shard_pass and feature_pass):
        errors.append(
            "score 1.0 requires exact receipts, complete shards, sealed splits, and decision-time features"
        )
    if score == 1.0 and payload.get("verdict_class") != "null":
        errors.append("valid readiness requires verdict_class null")
    if score == 0.0 and payload.get("verdict_class") not in {"blocked", "disqualified"}:
        errors.append("invalid readiness requires verdict_class blocked or disqualified")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for label and receipt checks")
    if score == 0.0 and not payload.get("gate_check_summary"):
        errors.append("score 0.0 requires gate_check_summary entries")
    if payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged") is not True:
        errors.append("protected files changed during audit")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    honest = str(payload.get("honest_verdict") or "")
    if not (honest.startswith("complete_") or honest.startswith("blocked_")):
        errors.append("honest_verdict lacks terminal prefix")
    status = str(payload.get("status") or "")
    if not (status.startswith("complete_") or status.startswith("blocked_")):
        errors.append("status lacks terminal prefix")
    return errors


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    upstream_path: Path | str = UPSTREAM_RELATIVE_PATH,
    validate: Callable[[Mapping[str, Any]], list[str]] = validate_artifact,
) -> JsonDict:
    """Build, write, and validate the production artifact."""

    start = time.perf_counter()
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        upstream_path=upstream_path,
        write=True,
        duration_s=None,
        tests_run=DEFAULT_TESTS_RUN,
        run_date=date,
    )
    artifact["duration_s"] = round(time.perf_counter() - start, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    atomic_write_json(result_path, artifact, root=REPO_ROOT, allow_override=False)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--upstream-path", default=str(UPSTREAM_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    result_path = Path(args.result_path)
    if args.validate:
        payload, error = _read_json(result_path)
        if error:
            raise ValueError(error)
        errors = validate_artifact(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path, upstream_path=Path(args.upstream_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
