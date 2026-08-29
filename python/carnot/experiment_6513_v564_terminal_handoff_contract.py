"""Exp6513 V564 terminal handoff contract.

Spec refs: REQ-BENCH-6513, SCENARIO-BENCH-6513-DIRECT-IMMUTABLE,
SCENARIO-BENCH-6513-ROW-REPLAY, SCENARIO-BENCH-6513-TERMINAL-HISTORY,
SCENARIO-BENCH-6513-RETIRED-ISOLATION, SCENARIO-BENCH-6513-ATTACKS,
SCENARIO-BENCH-6513-ATOMIC-FINAL.

The handoff is a governance replay. It preserves V563 terminal history and
authorizes direct path-and-hash reads, but it does not make a new scientific
performance claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot import experiment_6504_exact_structural_benchmark_commitment as exp6504
from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260822"
RANDOM_SEED = 6513
SCHEMA_VERSION = "carnot.experiment_6513.v564_terminal_handoff_contract.v1"
INFERENCE_SUBSTRATE = "bounded_historical_artifact_and_conductor_replay_no_llm"
VERIFIER_IS_ORACLE = True

RESULT_RELATIVE_PATH = Path("results/experiment_6513_v564_terminal_handoff_contract.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6513_v564_terminal_handoff_contract.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6513_v564_terminal_handoff_contract.py")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")

EXP6504_RELATIVE_PATH = Path("results/experiment_6504_exact_structural_benchmark_commitment.json")
EXP6506_RELATIVE_PATH = Path(
    "results/experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.json"
)
EXP6510_RELATIVE_PATH = Path("results/experiment_6510_v563_independent_exact_root.json")
EXP6511_RELATIVE_PATH = Path("results/experiment_6511_exact_branch_counterfactual_dataset_v2.json")
EXP6512_RELATIVE_PATH = Path("results/experiment_6512_branch_dataset_independent_audit.json")

RETIRED_OR_INELIGIBLE_TASK_IDS = (
    "exp6506-v561-evidence-corrigendum-v562-lineage-lock",
    "exp6507-exact-branch-counterfactual-dataset",
    "exp6508-analytical-branch-refocus-ab",
    "exp6509-critical-variable-enumeration-ab",
    "exp6510-v563-independent-exact-root",
    "exp6511-exact-branch-counterfactual-dataset-v2",
)

DIRECT_INPUT_PATHS = (
    EXP6504_RELATIVE_PATH,
    EXP6506_RELATIVE_PATH,
    EXP6510_RELATIVE_PATH,
    EXP6511_RELATIVE_PATH,
    EXP6512_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
)

PROTECTED_RELATIVE_PATHS = (
    EXP6504_RELATIVE_PATH,
    EXP6506_RELATIVE_PATH,
    EXP6510_RELATIVE_PATH,
    EXP6512_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    Path("research-program.md"),
    Path("scripts/research_conductor.py"),
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    EXP6504_RELATIVE_PATH,
    EXP6506_RELATIVE_PATH,
    EXP6510_RELATIVE_PATH,
    EXP6512_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "prior_failure_receipts",
    "historical_task_rows",
    "immutable_input_receipts",
    "allowed_direct_input_rows",
    "forbidden_dependency_rows",
    "retired_dependency_attack_matrix",
    "v564_handoff_ready_score",
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
    "status": "A terminal state distinguishes the handoff from a bootstrap or partial write.",
    "honest_verdict": (
        "The verdict preserves terminal history without claiming scientific success."
    ),
    "verdict_class": (
        "A null or partial class prevents terminal governance from becoming a performance claim."
    ),
    "prior_failure_receipts": (
        "Receipts preserve Exp6506 and Exp6510 bootstrap failures, missing Exp6511, "
        "and the Exp6512 block without reactivating them."
    ),
    "historical_task_rows": (
        "One row per prior task separates task outcome from usable file content."
    ),
    "immutable_input_receipts": "Path and hash receipts make every direct read auditable.",
    "allowed_direct_input_rows": (
        "Direct input rows authorize content reads without structured dependencies."
    ),
    "forbidden_dependency_rows": (
        "Forbidden rows prove retired or missing IDs cannot gate V564 work."
    ),
    "retired_dependency_attack_matrix": (
        "Attacks catch aliases, indirect requires, stale paths, drift, positive framing, "
        "and terminal-success confusion."
    ),
    "v564_handoff_ready_score": (
        "The score opens only when all historical determinations are preserved and no "
        "retired structured dependency is planned."
    ),
    "gate_check_summary": "Each gate records expected and observed values for replay.",
    "per_unit_rows": (
        "Unit rows expose every task, file, event, input, dependency, and attack row."
    ),
    "aggregate_row_recomputation": (
        "The aggregate is recomputed from rows instead of imported totals."
    ),
    "preconditions_checked": (
        "Preconditions record git state, resources, solver, roadmap, exclusion, and required paths."
    ),
    "protected_files_unchanged": (
        "Historical files and conductor code must remain unchanged during the handoff."
    ),
    "inference_substrate": (
        "The declaration keeps the handoff in artifact and conductor replay with no LLM."
    ),
    "verifier_is_oracle": (
        "Oracle disclosure limits authority to row, hash, and terminal-state checks."
    ),
    "field_principles": "Reasons beside fields help future tasks preserve the contract.",
    "field_provenance": (
        "Provenance records path, hash, reducer, and constant sources for each field."
    ),
    "random_seed": "A fixed attack order makes replay deterministic.",
    "duration_s": "Measured wall time supports authenticity checks.",
    "tests_run": "Command receipts show which verification actually ran.",
    "reproducibility_checksum": "A content hash detects drift in inputs, rows, and decisions.",
}

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6513_v564_terminal_handoff_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6513_v564_terminal_handoff_contract.py "
    "-m pytest tests/python/test_experiment_6513_v564_terminal_handoff_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6513_v564_terminal_handoff_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6513_v564_terminal_handoff_contract.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6513_v564_terminal_handoff_contract --date 20260822"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6513_v564_terminal_handoff_contract.json"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6513_v564_terminal_handoff_contract.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6513_v564_terminal_handoff_contract --validate"
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

ATTACK_IDS = (
    "renamed_dependency",
    "indirect_requires_chain",
    "stale_path",
    "hash_drift",
    "positive_exact_oracle_framing",
    "terminal_means_scientific_success",
    "structured_dependency_on_direct_input",
)

ALLOWED_DIRECT_SOURCE_LABELS = {path.as_posix() for path in DIRECT_INPUT_PATHS}


def canonical_json(value: Any) -> str:
    """Return stable JSON text so repeated runs produce the same hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with the repository-visible prefix."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash immutable evidence files and return ``missing`` for absent paths."""

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
    return dict(payload) if isinstance(payload, Mapping) else {}


def _git_output(repo_root: Path, args: Sequence[str]) -> str:
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _source_hashes(repo_root: Path) -> dict[str, str]:
    return {path.as_posix(): sha256_file(repo_root / path) for path in SOURCE_RELATIVE_PATHS}


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
        "ram_total_bytes": meminfo.get("MemTotal"),
        "ram_available_bytes": meminfo.get("MemAvailable"),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def _solver_state() -> JsonDict:
    z3_cli = shutil.which("z3")
    return {
        "z3_python_available": True,
        "z3_python_version": exp6504.z3.get_version_string(),
        "z3_cli_path": z3_cli,
        "exact_solver_available": z3_cli is not None,
    }


def _exclusion_manifest_state(repo_root: Path) -> JsonDict:
    path = repo_root / EXCLUSION_MANIFEST_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    markers = [task_id for task_id in RETIRED_OR_INELIGIBLE_TASK_IDS if task_id in text]
    return {
        "path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        "present": path.is_file(),
        "sha256": sha256_file(path),
        "line_count": len(text.splitlines()),
        "retired_entry_marker_count": text.count("- id:") + text.count("- experiment_id:"),
        "retired_task_markers_present": markers,
    }


def protected_file_hashes(repo_root: Path) -> dict[str, JsonDict]:
    """Capture hashes for historical files that the handoff must not edit."""

    return {
        path.as_posix(): {
            "exists": (repo_root / path).is_file(),
            "sha256": sha256_file(repo_root / path),
            "protected_by_exp6513_handoff": True,
        }
        for path in PROTECTED_RELATIVE_PATHS
    }


def protected_files_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Compare pre/post hashes and fail closed on missing protected files."""

    files: dict[str, JsonDict] = {}
    for path in sorted(set(before) | set(after)):
        prior = dict(before.get(path, {}))
        post = dict(after.get(path, {}))
        unchanged = (
            prior.get("sha256") == post.get("sha256")
            and prior.get("sha256") not in {None, "missing"}
            and prior.get("exists") is True
            and post.get("exists") is True
        )
        files[path] = {
            "sha256_before": prior.get("sha256", "missing"),
            "sha256_after": post.get("sha256", "missing"),
            "exists_before": prior.get("exists") is True,
            "exists_after": post.get("exists") is True,
            "unchanged": unchanged,
            "protected_by_exp6513_handoff": True,
        }
    changed = [path for path, row in files.items() if row["unchanged"] is not True]
    return {
        "files": files,
        "changed_paths": changed,
        "all_protected_files_unchanged": changed == [],
    }


def _immutable_receipt(
    repo_root: Path,
    input_id: str,
    relative: Path,
    *,
    json_pointers: Sequence[str] = (),
    line_selectors: Sequence[str] = (),
) -> JsonDict:
    path = repo_root / relative
    payload = {
        "row_type": "immutable_file",
        "schema_version": SCHEMA_VERSION + ".immutable_input_receipt",
        "input_id": input_id,
        "path": relative.as_posix(),
        "absolute_path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "json_pointers": list(json_pointers),
        "line_selectors": list(line_selectors),
        "read_mode": "direct_path_and_hash",
        "counts_as_structured_dependency": False,
        "spec_refs": ["REQ-BENCH-6513", "SCENARIO-BENCH-6513-DIRECT-IMMUTABLE"],
    }
    return {**payload, "unit_row_hash": sha256_json(payload)}


def immutable_input_receipts(repo_root: Path) -> list[JsonDict]:
    """Record all file content and absence evidence used by the handoff."""

    return [
        _immutable_receipt(
            repo_root,
            "exp6504",
            EXP6504_RELATIVE_PATH,
            json_pointers=(
                "/raw_instance_rows",
                "/exact_label_rows",
                "/exact_replay_rows",
                "/split_commitment",
                "/reproducibility_checksum",
            ),
        ),
        _immutable_receipt(
            repo_root,
            "exp6506",
            EXP6506_RELATIVE_PATH,
            json_pointers=(
                "/exp6504_row_recomputation",
                "/exp6504_corrigendum",
                "/lineage_decision_rows",
                "/forbidden_dependency_attack_matrix",
            ),
        ),
        _immutable_receipt(
            repo_root,
            "exp6510",
            EXP6510_RELATIVE_PATH,
            json_pointers=(
                "/historical_input_receipts",
                "/independent_row_recomputation",
                "/lineage_decision_rows",
                "/retired_dependency_attack_matrix",
                "/per_unit_rows",
            ),
        ),
        _immutable_receipt(
            repo_root,
            "exp6511_missing",
            EXP6511_RELATIVE_PATH,
            json_pointers=("/branch_counterfactual_rows", "/per_unit_rows"),
        ),
        _immutable_receipt(
            repo_root,
            "exp6512",
            EXP6512_RELATIVE_PATH,
            json_pointers=(
                "/branch_dataset_audited_ready_score",
                "/gate_check_summary",
                "/per_unit_rows",
            ),
        ),
        _immutable_receipt(
            repo_root,
            "conductor_log",
            CONDUCTOR_LOG_RELATIVE_PATH,
            line_selectors=(
                "Exp6506 artifact_not_updated_past_bootstrap",
                "Exp6510 artifact_not_updated_past_bootstrap",
                "Exp6511 gate block",
            ),
        ),
        _immutable_receipt(repo_root, "exclusion_manifest", EXCLUSION_MANIFEST_RELATIVE_PATH),
        _immutable_receipt(repo_root, "research_roadmap", ROADMAP_RELATIVE_PATH),
        _immutable_receipt(repo_root, "v564_change_proposal", VNEXT_RELATIVE_PATH),
    ]


def recompute_exp6504_direct(repo_root: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Recompute Exp6504 from raw rows instead of trusting stored aggregates."""

    raw_rows = [dict(row) for row in payload.get("raw_instance_rows", [])]
    regenerated_raw_rows = exp6504.generate_instance_rows()
    recomputed_labels = [exp6504.label_instance(row) for row in raw_rows]
    label_by_id = {str(row["instance_id"]): row for row in recomputed_labels}
    recomputed_replays = [
        exp6504.replay_label(row, label_by_id[str(row["instance_id"])]) for row in raw_rows
    ]
    split = exp6504.split_commitment_rows(raw_rows)
    strata = exp6504.stratum_balance_rows(raw_rows, recomputed_labels)
    held_cells = exp6504.minimum_held_cell_size(recomputed_labels)
    leakage = exp6504.leakage_attack_matrix(raw_rows, recomputed_labels, split, held_cells)
    recomputed_rows = exp6504.per_unit_rows(
        raw_rows,
        recomputed_labels,
        recomputed_replays,
        split["rows"],
        strata,
        leakage["rows"],
        held_cells["planned_headline_cell_rows"],
    )
    aggregate = exp6504.recompute_aggregates_from_rows(recomputed_rows)
    regenerated_by_id = {str(row["instance_id"]): row for row in regenerated_raw_rows}
    stored_label_by_id = {
        str(row["instance_id"]): row for row in payload.get("exact_label_rows", [])
    }
    raw_hash_match_count = sum(
        1
        for row in raw_rows
        if regenerated_by_id.get(str(row["instance_id"]), {}).get("raw_instance_hash")
        == row.get("raw_instance_hash")
    )
    label_semantic_match_count = sum(
        1
        for row in recomputed_labels
        if stored_label_by_id.get(str(row["instance_id"]), {}).get("exact_label")
        == row.get("exact_label")
        and stored_label_by_id.get(str(row["instance_id"]), {}).get("accepted")
        == row.get("accepted")
        and stored_label_by_id.get(str(row["instance_id"]), {}).get("model_or_proof_valid")
        == row.get("model_or_proof_valid")
    )
    replay_failure_count = sum(
        1 for row in recomputed_replays if row.get("replay_passed") is not True
    )
    split_hash_matches = split.get("split_commitment_hash") == payload.get(
        "split_commitment", {}
    ).get("split_commitment_hash")
    historical_checksum_matches = payload.get("reproducibility_checksum") == (
        exp6504.reproducibility_checksum(payload)
    )
    row_replay_passed = (
        len(raw_rows) == exp6504.INSTANCE_COUNT
        and raw_hash_match_count == len(raw_rows)
        and label_semantic_match_count == len(recomputed_labels)
        and replay_failure_count == 0
        and split_hash_matches
        and historical_checksum_matches
        and aggregate == payload.get("aggregate_row_recomputation")
    )
    return {
        "schema_version": SCHEMA_VERSION + ".exp6504_row_recomputation",
        "source_artifact_path": EXP6504_RELATIVE_PATH.as_posix(),
        "source_artifact_sha256": sha256_file(repo_root / EXP6504_RELATIVE_PATH),
        "raw_row_count": len(raw_rows),
        "exact_label_row_count": len(recomputed_labels),
        "exact_replay_row_count": len(recomputed_replays),
        "raw_hash_match_count": raw_hash_match_count,
        "label_semantic_match_count": label_semantic_match_count,
        "replay_failure_count": replay_failure_count,
        "split_hash_matches": split_hash_matches,
        "stored_aggregate_matches_recomputed": aggregate
        == payload.get("aggregate_row_recomputation"),
        "historical_checksum_matches": historical_checksum_matches,
        "row_replay_passed": row_replay_passed,
        "verifier_is_oracle_for_exact_label_hash_and_row_checks": True,
    }


def _contains_retired_task_id(value: Any) -> bool:
    text = canonical_json(value).lower() if isinstance(value, (dict, list)) else str(value).lower()
    return any(
        task_id in text or task_id.split("-")[0] in text
        for task_id in RETIRED_OR_INELIGIBLE_TASK_IDS
    )


def recompute_exp6510_content(repo_root: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Recompute Exp6510 content status from rows and local checks."""

    per_unit_rows = [dict(row) for row in payload.get("per_unit_rows", [])]
    row_type_counts = Counter(str(row.get("row_type")) for row in per_unit_rows)
    direct_rows = [
        row for row in per_unit_rows if row.get("row_type") == "v563_exp6504_direct_replay"
    ]
    lineage_rows = [row for row in per_unit_rows if row.get("row_type") == "v563_lineage_decision"]
    attack_rows = [
        row for row in per_unit_rows if row.get("row_type") == "v563_retired_dependency_attack"
    ]
    direct_rows_pass = bool(direct_rows) and all(
        row.get("label_semantics_match") is True
        and row.get("replay_passed") is True
        and row.get("regenerated_raw_hash_matches") is True
        for row in direct_rows
    )
    attacks_fail_closed = bool(attack_rows) and all(
        row.get("fail_closed") is True for row in attack_rows
    )
    allowed_structured_violations = [
        row
        for row in lineage_rows
        if row.get("decision") == "allow"
        and row.get("dependency_kind") == "structured_dependency"
        and _contains_retired_task_id(
            {
                "source_label": row.get("source_label"),
                "field": row.get("field"),
                "downstream_task": row.get("downstream_task"),
            }
        )
    ]
    checksum_matches = payload.get("reproducibility_checksum") == _exp6510_checksum(payload)
    non_positive_class = payload.get("verdict_class") in {"partial", "null"}
    row_supported_ready = (
        len(direct_rows) == 480
        and len(lineage_rows) == 10
        and len(attack_rows) == 7
        and direct_rows_pass
        and attacks_fail_closed
        and allowed_structured_violations == []
        and checksum_matches
        and non_positive_class
        and payload.get("v563_independent_root_ready_score") == 1.0
    )
    return {
        "schema_version": SCHEMA_VERSION + ".exp6510_content_recomputation",
        "source_artifact_path": EXP6510_RELATIVE_PATH.as_posix(),
        "source_artifact_sha256": sha256_file(repo_root / EXP6510_RELATIVE_PATH),
        "status": payload.get("status"),
        "verdict_class": payload.get("verdict_class"),
        "total_per_unit_row_count": len(per_unit_rows),
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "direct_replay_row_count": len(direct_rows),
        "lineage_decision_row_count": len(lineage_rows),
        "retired_attack_row_count": len(attack_rows),
        "direct_replay_rows_pass": direct_rows_pass,
        "attack_rows_fail_closed": attacks_fail_closed,
        "allowed_structured_retired_violation_count": len(allowed_structured_violations),
        "historical_reproducibility_checksum_matches": checksum_matches,
        "non_positive_verdict_class": non_positive_class,
        "ready_score_from_rows": 1.0 if row_supported_ready else 0.0,
        "usable_content": row_supported_ready,
        "eligible_task_dependency": False,
        "source_task_is_retired": True,
        "verifier_is_oracle_for_exact_row_hash_and_terminal_checks": True,
    }


def _exp6510_checksum(payload: Mapping[str, Any]) -> str:
    checksum_payload = {
        "status": payload.get("status"),
        "verdict_class": payload.get("verdict_class"),
        "prior_failure_receipt": payload.get("prior_failure_receipt"),
        "historical_input_receipts": payload.get("historical_input_receipts"),
        "independent_row_recomputation": payload.get("independent_row_recomputation"),
        "lineage_decision_rows": payload.get("lineage_decision_rows"),
        "retired_dependency_attack_matrix": payload.get("retired_dependency_attack_matrix"),
        "v563_independent_root_ready_score": payload.get("v563_independent_root_ready_score"),
        "per_unit_rows": payload.get("per_unit_rows"),
        "gate_check_summary": payload.get("gate_check_summary"),
        "protected_files_unchanged": payload.get("protected_files_unchanged"),
        "inference_substrate": payload.get("inference_substrate"),
        "verifier_is_oracle": payload.get("verifier_is_oracle"),
        "random_seed": payload.get("random_seed"),
        "honest_verdict": payload.get("honest_verdict"),
    }
    return sha256_json(checksum_payload)


def _conductor_event_row(line_no: int, line: str, event_id: str, task_id: str) -> JsonDict:
    payload = {
        "row_type": "conductor_terminal_event",
        "schema_version": SCHEMA_VERSION + ".conductor_terminal_event",
        "event_id": event_id,
        "task_id": task_id,
        "source_path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        "line_number": line_no,
        "line_text": line,
        "terminal_result": "artifact_not_updated_past_bootstrap"
        if "artifact_not_updated_past_bootstrap" in line
        else "gate_block",
        "spec_refs": ["REQ-BENCH-6513", "SCENARIO-BENCH-6513-TERMINAL-HISTORY"],
    }
    return {**payload, "conductor_row_hash": sha256_json(payload)}


def conductor_terminal_event_rows(repo_root: Path) -> list[JsonDict]:
    """Extract historical terminal rows from the conductor log."""

    path = repo_root / CONDUCTOR_LOG_RELATIVE_PATH
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    rows: list[JsonDict] = []
    counters: Counter[str] = Counter()
    patterns = (
        (
            "V561 evidence corrigendum and V562 exact-branch",
            "artifact_not_updated_past_bootstrap",
            "exp6506-v561-evidence-corrigendum-v562-lineage-lock",
        ),
        (
            "Fresh independent V563 exact-evidence root",
            "artifact_not_updated_past_bootstrap",
            "exp6510-v563-independent-exact-root",
        ),
        (
            "Sealed exact branch-counterfactual dataset v2",
            "GATE_BLOCK",
            "exp6511-exact-branch-counterfactual-dataset-v2",
        ),
    )
    for line_no, line in enumerate(lines, start=1):
        for title, result, task_id in patterns:
            if title in line and result in line:
                counters[task_id] += 1
                rows.append(
                    _conductor_event_row(
                        line_no,
                        line,
                        f"{task_id}:{counters[task_id]}",
                        task_id,
                    )
                )
    return rows


def prior_failure_receipts(
    repo_root: Path,
    events: Sequence[Mapping[str, Any]],
    exp6512_payload: Mapping[str, Any],
) -> list[JsonDict]:
    """Preserve old terminal outcomes without making them dependencies."""

    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in events:
        by_task.setdefault(str(row.get("task_id")), []).append(row)
    exp6511_path = repo_root / EXP6511_RELATIVE_PATH
    rows = [
        {
            "receipt_id": "exp6506_bootstrap_failures",
            "task_id": "exp6506-v561-evidence-corrigendum-v562-lineage-lock",
            "source_path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            "source_sha256": sha256_file(repo_root / CONDUCTOR_LOG_RELATIVE_PATH),
            "terminal_result": "artifact_not_updated_past_bootstrap",
            "event_count": len(
                by_task.get("exp6506-v561-evidence-corrigendum-v562-lineage-lock", [])
            ),
            "event_row_ids": [
                row.get("event_id")
                for row in by_task.get("exp6506-v561-evidence-corrigendum-v562-lineage-lock", [])
            ],
            "source_task_reactivated": False,
            "eligible_task_dependency": False,
        },
        {
            "receipt_id": "exp6510_bootstrap_failures",
            "task_id": "exp6510-v563-independent-exact-root",
            "source_path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            "source_sha256": sha256_file(repo_root / CONDUCTOR_LOG_RELATIVE_PATH),
            "terminal_result": "artifact_not_updated_past_bootstrap",
            "event_count": len(by_task.get("exp6510-v563-independent-exact-root", [])),
            "event_row_ids": [
                row.get("event_id")
                for row in by_task.get("exp6510-v563-independent-exact-root", [])
            ],
            "source_task_reactivated": False,
            "eligible_task_dependency": False,
        },
        {
            "receipt_id": "exp6511_missing_dataset",
            "task_id": "exp6511-exact-branch-counterfactual-dataset-v2",
            "source_path": EXP6511_RELATIVE_PATH.as_posix(),
            "absolute_path": str(exp6511_path),
            "exists": exp6511_path.is_file(),
            "sha256": sha256_file(exp6511_path),
            "conductor_gate_block_count": len(
                by_task.get("exp6511-exact-branch-counterfactual-dataset-v2", [])
            ),
            "terminal_result": "missing_deliverable_after_gate_blocks",
            "eligible_task_dependency": False,
        },
        {
            "receipt_id": "exp6512_score_zero_block",
            "task_id": "exp6512-branch-dataset-independent-audit",
            "source_path": EXP6512_RELATIVE_PATH.as_posix(),
            "source_sha256": sha256_file(repo_root / EXP6512_RELATIVE_PATH),
            "observed_status": exp6512_payload.get("status"),
            "observed_verdict_class": exp6512_payload.get("verdict_class"),
            "observed_score": exp6512_payload.get("branch_dataset_audited_ready_score"),
            "observed_per_unit_row_count": len(exp6512_payload.get("per_unit_rows", [])),
            "gate_check_count": len(exp6512_payload.get("gate_check_summary", [])),
            "terminal_result": "blocked_score_zero",
            "eligible_task_dependency": False,
        },
    ]
    out: list[JsonDict] = []
    for row in rows:
        payload = {
            "row_type": "prior_failure_receipt",
            "schema_version": SCHEMA_VERSION + ".prior_failure_receipt",
            "spec_refs": ["REQ-BENCH-6513", "SCENARIO-BENCH-6513-TERMINAL-HISTORY"],
            **row,
        }
        out.append({**payload, "receipt_hash": sha256_json(payload)})
    return out


def historical_task_rows(
    repo_root: Path,
    exp6504_payload: Mapping[str, Any],
    exp6506_payload: Mapping[str, Any],
    exp6510_payload: Mapping[str, Any],
    exp6511_missing: Mapping[str, Any],
    exp6512_payload: Mapping[str, Any],
    exp6504_replay: Mapping[str, Any],
    exp6510_replay: Mapping[str, Any],
) -> list[JsonDict]:
    """Emit one row per historical task named by the V564 handoff."""

    base_rows = [
        {
            "task_id": "exp6504-exact-structural-benchmark-commitment",
            "source_path": EXP6504_RELATIVE_PATH.as_posix(),
            "exists": (repo_root / EXP6504_RELATIVE_PATH).is_file(),
            "status": exp6504_payload.get("status"),
            "verdict_class": exp6504_payload.get("verdict_class"),
            "terminal_score": exp6504_payload.get("base_structural_benchmark_ready_score"),
            "usable_content": exp6504_replay.get("row_replay_passed") is True,
            "source_task_is_retired": False,
            "eligible_task_dependency": False,
            "determination": "usable_direct_rows_only",
        },
        {
            "task_id": "exp6506-v561-evidence-corrigendum-v562-lineage-lock",
            "source_path": EXP6506_RELATIVE_PATH.as_posix(),
            "exists": (repo_root / EXP6506_RELATIVE_PATH).is_file(),
            "status": exp6506_payload.get("status"),
            "verdict_class": exp6506_payload.get("verdict_class"),
            "terminal_score": exp6506_payload.get("v562_exact_branch_ready_score"),
            "usable_content": True,
            "source_task_is_retired": True,
            "eligible_task_dependency": False,
            "determination": "retired_task_with_usable_immutable_receipts",
        },
        {
            "task_id": "exp6507-exact-branch-counterfactual-dataset",
            "source_path": "results/experiment_6507_exact_branch_counterfactual_dataset.json",
            "exists": False,
            "status": "blocked_preemptive_skip_upstream_retired",
            "verdict_class": "blocked",
            "terminal_score": 0.0,
            "usable_content": False,
            "source_task_is_retired": True,
            "eligible_task_dependency": False,
            "determination": "retired_gate_block",
        },
        {
            "task_id": "exp6508-analytical-branch-refocus-ab",
            "source_path": "results/experiment_6508_analytical_branch_refocus_ab.json",
            "exists": (
                repo_root / "results/experiment_6508_analytical_branch_refocus_ab.json"
            ).is_file(),
            "status": "blocked_gate_check_failed",
            "verdict_class": "blocked",
            "terminal_score": 0.0,
            "usable_content": False,
            "source_task_is_retired": True,
            "eligible_task_dependency": False,
            "determination": "retired_gate_block",
        },
        {
            "task_id": "exp6509-critical-variable-enumeration-ab",
            "source_path": "results/experiment_6509_critical_variable_enumeration_ab.json",
            "exists": False,
            "status": "blocked_preemptive_skip_upstream_retired",
            "verdict_class": "blocked",
            "terminal_score": 0.0,
            "usable_content": False,
            "source_task_is_retired": True,
            "eligible_task_dependency": False,
            "determination": "retired_gate_block",
        },
        {
            "task_id": "exp6510-v563-independent-exact-root",
            "source_path": EXP6510_RELATIVE_PATH.as_posix(),
            "exists": (repo_root / EXP6510_RELATIVE_PATH).is_file(),
            "status": exp6510_payload.get("status"),
            "verdict_class": exp6510_payload.get("verdict_class"),
            "terminal_score": exp6510_replay.get("ready_score_from_rows"),
            "usable_content": exp6510_replay.get("usable_content") is True,
            "source_task_is_retired": True,
            "eligible_task_dependency": False,
            "determination": "retired_task_with_usable_immutable_content",
        },
        {
            "task_id": "exp6511-exact-branch-counterfactual-dataset-v2",
            "source_path": EXP6511_RELATIVE_PATH.as_posix(),
            "exists": exp6511_missing.get("exists") is True,
            "status": "missing_deliverable",
            "verdict_class": "blocked",
            "terminal_score": 0.0,
            "usable_content": False,
            "source_task_is_retired": True,
            "eligible_task_dependency": False,
            "determination": "missing_after_retired_upstream_gate_blocks",
        },
        {
            "task_id": "exp6512-branch-dataset-independent-audit",
            "source_path": EXP6512_RELATIVE_PATH.as_posix(),
            "exists": (repo_root / EXP6512_RELATIVE_PATH).is_file(),
            "status": exp6512_payload.get("status"),
            "verdict_class": exp6512_payload.get("verdict_class"),
            "terminal_score": exp6512_payload.get("branch_dataset_audited_ready_score"),
            "usable_content": exp6512_payload.get("branch_dataset_audited_ready_score") == 0.0,
            "source_task_is_retired": False,
            "eligible_task_dependency": False,
            "determination": "closed_blocked_audit_with_zero_rows",
        },
    ]
    rows: list[JsonDict] = []
    for row in base_rows:
        payload = {
            "row_type": "historical_task",
            "schema_version": SCHEMA_VERSION + ".historical_task",
            "source_sha256": sha256_file(repo_root / str(row["source_path"])),
            "spec_refs": ["REQ-BENCH-6513", "SCENARIO-BENCH-6513-TERMINAL-HISTORY"],
            **row,
        }
        rows.append({**payload, "historical_task_row_hash": sha256_json(payload)})
    return rows


def classify_dependency(row: Mapping[str, Any]) -> JsonDict:
    """Classify one direct input or structured dependency with closed defaults."""

    candidate = dict(row)
    text = " ".join(str(value) for value in candidate.values()).lower()
    hash_present = candidate.get("required_hash_present") is True
    source_label = str(candidate.get("source_label", ""))
    kind = str(candidate.get("dependency_kind", ""))
    if not hash_present:
        decision = "block"
        reason = "missing_or_stale_hash"
    elif (
        kind == "direct_file_input"
        and source_label in ALLOWED_DIRECT_SOURCE_LABELS
        and candidate.get("read_mode", "direct_path_and_hash") == "direct_path_and_hash"
    ):
        decision = "allow"
        reason = "direct_path_and_hash_read"
    elif kind in {"requires", "gated_on"} and _contains_retired_task_id(candidate):
        decision = "forbid"
        reason = "indirect_retired_requires_chain"
    elif kind == "structured_dependency" and _contains_retired_task_id(candidate):
        decision = "forbid"
        reason = "retired_task_dependency"
    elif "terminal" in text and "scientific success" in text:
        decision = "forbid"
        reason = "terminal_success_is_not_scientific_success"
    elif "positive" in text or "oracle" in text:
        decision = "forbid"
        reason = "positive_exact_oracle_framing_forbidden"
    elif "retired" in text or "alias" in text or "v562" in text or "v563" in text:
        decision = "forbid"
        reason = "renamed_retired_dependency"
    else:
        decision = "forbid"
        reason = "unknown_dependency_fail_closed"
    return {
        **candidate,
        "decision": decision,
        "reason": reason,
        "fail_closed": decision != "allow",
        "classifier": "classify_dependency",
    }


def allowed_direct_input_rows(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Authorize direct file reads while keeping task dependencies closed."""

    receipt_by_id = {str(row.get("input_id")): row for row in receipts}
    specs = (
        ("exp6504_rows", "exp6504", "raw_instance_rows,exact_label_rows"),
        ("exp6506_receipts", "exp6506", "corrigendum_receipts"),
        ("exp6510_content", "exp6510", "usable_immutable_content"),
        ("exp6512_block", "exp6512", "blocked_score_zero_receipt"),
        ("conductor_rows", "conductor_log", "terminal_event_rows"),
        ("roadmap_v564", "research_roadmap", "planned_structured_dependencies"),
        ("exclusion_state", "exclusion_manifest", "retired_scope_state"),
    )
    rows: list[JsonDict] = []
    for input_id, receipt_id, imported_fields in specs:
        receipt = receipt_by_id[receipt_id]
        classified = classify_dependency(
            {
                "row_type": "allowed_direct_input",
                "schema_version": SCHEMA_VERSION + ".allowed_direct_input",
                "input_id": input_id,
                "dependency_kind": "direct_file_input",
                "source_label": receipt["path"],
                "source_sha256": receipt["sha256"],
                "required_hash_present": str(receipt["sha256"]).startswith("sha256:"),
                "read_mode": "direct_path_and_hash",
                "imported_fields": imported_fields,
                "source_task_is_retired": receipt_id in {"exp6506", "exp6510"},
                "eligible_task_dependency": False,
                "counts_as_structured_dependency": False,
                "spec_refs": ["REQ-BENCH-6513", "SCENARIO-BENCH-6513-DIRECT-IMMUTABLE"],
            }
        )
        rows.append({**classified, "allowed_direct_input_row_hash": sha256_json(classified)})
    return rows


def forbidden_dependency_rows() -> list[JsonDict]:
    """Emit one forbidden row for each retired or missing historical task ID."""

    rows: list[JsonDict] = []
    for task_id in RETIRED_OR_INELIGIBLE_TASK_IDS:
        classified = classify_dependency(
            {
                "row_type": "forbidden_dependency",
                "schema_version": SCHEMA_VERSION + ".forbidden_dependency",
                "dependency_id": task_id,
                "dependency_kind": "structured_dependency",
                "source_label": task_id,
                "required_hash_present": True,
                "eligible_task_dependency": False,
                "allowed_direct_file_read_only": task_id
                in {
                    "exp6506-v561-evidence-corrigendum-v562-lineage-lock",
                    "exp6510-v563-independent-exact-root",
                },
                "spec_refs": ["REQ-BENCH-6513", "SCENARIO-BENCH-6513-RETIRED-ISOLATION"],
            }
        )
        rows.append({**classified, "forbidden_dependency_row_hash": sha256_json(classified)})
    return rows


def retired_dependency_attack_matrix() -> JsonDict:
    """Probe shortcuts that must not open the V564 handoff score."""

    attacks = [
        (
            "renamed_dependency",
            {
                "dependency_id": "v563_root_alias",
                "dependency_kind": "structured_dependency",
                "source_label": "renamed V563 independent exact root alias exp6510",
                "required_hash_present": True,
            },
        ),
        (
            "indirect_requires_chain",
            {
                "dependency_id": "requires_exp6510",
                "dependency_kind": "requires",
                "source_label": "exp6516 requires exp6510-v563-independent-exact-root",
                "required_hash_present": True,
            },
        ),
        (
            "stale_path",
            {
                "dependency_id": "stale_exp6510_path",
                "dependency_kind": "direct_file_input",
                "source_label": EXP6510_RELATIVE_PATH.as_posix(),
                "required_hash_present": False,
            },
        ),
        (
            "hash_drift",
            {
                "dependency_id": "exp6504_hash_drift",
                "dependency_kind": "direct_file_input",
                "source_label": EXP6504_RELATIVE_PATH.as_posix(),
                "required_hash_present": False,
            },
        ),
        (
            "positive_exact_oracle_framing",
            {
                "dependency_id": "positive_oracle_claim",
                "dependency_kind": "interpretation",
                "source_label": "positive exact oracle verdict_class",
                "required_hash_present": True,
            },
        ),
        (
            "terminal_means_scientific_success",
            {
                "dependency_id": "terminal_success_claim",
                "dependency_kind": "interpretation",
                "source_label": "terminal complete means scientific success",
                "required_hash_present": True,
            },
        ),
        (
            "structured_dependency_on_direct_input",
            {
                "dependency_id": "structured_exp6510_file",
                "dependency_kind": "structured_dependency",
                "source_label": EXP6510_RELATIVE_PATH.as_posix(),
                "required_hash_present": True,
            },
        ),
    ]
    rows: list[JsonDict] = []
    for attack_id, payload in attacks:
        classified = classify_dependency(
            {
                "row_type": "retired_dependency_attack",
                "schema_version": SCHEMA_VERSION + ".retired_dependency_attack",
                "attack_id": attack_id,
                "expected_decision": "block_or_forbid",
                "spec_refs": ["REQ-BENCH-6513", "SCENARIO-BENCH-6513-ATTACKS"],
                **payload,
            }
        )
        fail_closed = classified["decision"] in {"block", "forbid"}
        row = {
            **classified,
            "fail_closed": fail_closed,
            "observed_ready_score_if_only_this_attack": 0.0 if fail_closed else 1.0,
        }
        rows.append({**row, "attack_row_hash": sha256_json(row)})
    return {
        "schema_version": SCHEMA_VERSION + ".retired_dependency_attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_fail_closed": all(row["fail_closed"] is True for row in rows),
        "false_accept_count": sum(1 for row in rows if row["fail_closed"] is not True),
    }


def _planned_structured_dependencies(repo_root: Path) -> list[JsonDict]:
    roadmap_path = repo_root / ROADMAP_RELATIVE_PATH
    if not roadmap_path.is_file():
        return []
    data = yaml.safe_load(roadmap_path.read_text(encoding="utf-8")) or {}
    tasks = data.get("tasks", []) if isinstance(data, Mapping) else []
    rows: list[JsonDict] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id", ""))
        for key in ("gated_on", "requires"):
            values = task.get(key, [])
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for index, value in enumerate(values):
                upstream = value.get("upstream") if isinstance(value, Mapping) else value
                field = value.get("artifact_field") if isinstance(value, Mapping) else None
                payload = {
                    "task_id": task_id,
                    "dependency_kind": key,
                    "upstream": str(upstream),
                    "artifact_field": field,
                    "index": index,
                    "source_path": ROADMAP_RELATIVE_PATH.as_posix(),
                    "retired_id_present": _contains_retired_task_id(upstream),
                }
                rows.append({**payload, "planned_dependency_row_hash": sha256_json(payload)})
    return rows


def planned_structured_dependency_retired_id_violations(
    artifact_or_preconditions: Mapping[str, Any],
) -> list[JsonDict]:
    """Return planned gate or requires rows that name retired IDs."""

    preconditions = artifact_or_preconditions.get(
        "preconditions_checked", artifact_or_preconditions
    )
    explicit = preconditions.get("planned_structured_dependency_retired_id_violations", [])
    if explicit:
        return [dict(row) for row in explicit if isinstance(row, Mapping)]
    rows = preconditions.get("planned_structured_dependencies", [])
    return [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("retired_id_present") is True
    ]


def aggregate_row_recomputation(
    rows: Sequence[Mapping[str, Any]],
    *,
    exp6504_replay: Mapping[str, Any],
    exp6510_replay: Mapping[str, Any],
    prior_receipts: Sequence[Mapping[str, Any]],
    protected: Mapping[str, Any],
    planned_violations: Sequence[Mapping[str, Any]],
    attacks: Mapping[str, Any],
    forbidden: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce row evidence into the handoff readiness inputs."""

    row_types = Counter(str(row.get("row_type")) for row in rows)
    receipts = {str(row.get("receipt_id")): row for row in prior_receipts}
    forbidden_false_accepts = [row for row in forbidden if row.get("decision") == "allow"]
    return {
        "schema_version": SCHEMA_VERSION + ".aggregate_row_recomputation",
        "row_count": len(rows),
        "row_type_counts": dict(sorted(row_types.items())),
        "exp6504_raw_row_count": exp6504_replay.get("raw_row_count"),
        "exp6504_exact_label_row_count": exp6504_replay.get("exact_label_row_count"),
        "exp6504_exact_replay_row_count": exp6504_replay.get("exact_replay_row_count"),
        "exp6504_row_replay_passed": exp6504_replay.get("row_replay_passed") is True,
        "exp6510_total_per_unit_row_count": exp6510_replay.get("total_per_unit_row_count"),
        "exp6510_row_type_counts": exp6510_replay.get("row_type_counts"),
        "exp6510_ready_score_from_rows": exp6510_replay.get("ready_score_from_rows"),
        "exp6510_usable_content": exp6510_replay.get("usable_content") is True,
        "exp6510_eligible_task_dependency": exp6510_replay.get("eligible_task_dependency") is True,
        "exp6506_bootstrap_failure_count": receipts.get("exp6506_bootstrap_failures", {}).get(
            "event_count"
        ),
        "exp6510_bootstrap_failure_count": receipts.get("exp6510_bootstrap_failures", {}).get(
            "event_count"
        ),
        "exp6511_missing_confirmed": receipts.get("exp6511_missing_dataset", {}).get("exists")
        is False,
        "exp6512_blocked_score_zero_confirmed": (
            receipts.get("exp6512_score_zero_block", {}).get("observed_score") == 0.0
            and str(
                receipts.get("exp6512_score_zero_block", {}).get("observed_status", "")
            ).startswith("blocked_")
        ),
        "protected_files_unchanged": protected.get("all_protected_files_unchanged") is True,
        "planned_structured_dependency_retired_violation_count": len(planned_violations),
        "forbidden_dependency_false_accept_count": len(forbidden_false_accepts),
        "retired_attack_false_accept_count": attacks.get("false_accept_count"),
        "all_historical_determinations_preserved": (
            receipts.get("exp6506_bootstrap_failures", {}).get("event_count") == 3
            and receipts.get("exp6510_bootstrap_failures", {}).get("event_count") == 3
            and receipts.get("exp6511_missing_dataset", {}).get("exists") is False
            and receipts.get("exp6512_score_zero_block", {}).get("observed_score") == 0.0
            and str(
                receipts.get("exp6512_score_zero_block", {}).get("observed_status", "")
            ).startswith("blocked_")
        ),
    }


def gate_check_summary(
    *,
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    attacks: Mapping[str, Any],
    verdict_class: str,
    planned_violations: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Build replayable gate rows with expected and observed values."""

    checks = [
        (
            "exp6504_row_replay",
            True,
            aggregate.get("exp6504_row_replay_passed"),
            EXP6504_RELATIVE_PATH,
        ),
        (
            "exp6510_usable_content",
            True,
            aggregate.get("exp6510_usable_content"),
            EXP6510_RELATIVE_PATH,
        ),
        (
            "exp6510_eligible_task_dependency",
            False,
            aggregate.get("exp6510_eligible_task_dependency"),
            EXP6510_RELATIVE_PATH,
        ),
        (
            "exp6506_bootstrap_failures",
            3,
            aggregate.get("exp6506_bootstrap_failure_count"),
            CONDUCTOR_LOG_RELATIVE_PATH,
        ),
        (
            "exp6510_bootstrap_failures",
            3,
            aggregate.get("exp6510_bootstrap_failure_count"),
            CONDUCTOR_LOG_RELATIVE_PATH,
        ),
        (
            "exp6511_missing",
            True,
            aggregate.get("exp6511_missing_confirmed"),
            EXP6511_RELATIVE_PATH,
        ),
        (
            "exp6512_blocked_score_zero",
            True,
            aggregate.get("exp6512_blocked_score_zero_confirmed"),
            EXP6512_RELATIVE_PATH,
        ),
        (
            "historical_determinations_preserved",
            True,
            aggregate.get("all_historical_determinations_preserved"),
            CONDUCTOR_LOG_RELATIVE_PATH,
        ),
        (
            "protected_files_unchanged",
            True,
            protected.get("all_protected_files_unchanged"),
            ROADMAP_RELATIVE_PATH,
        ),
        (
            "retired_dependency_attacks_fail_closed",
            True,
            attacks.get("all_attacks_fail_closed"),
            ROADMAP_RELATIVE_PATH,
        ),
        (
            "no_planned_retired_structured_dependency",
            [],
            list(planned_violations),
            ROADMAP_RELATIVE_PATH,
        ),
        (
            "verdict_class_non_positive",
            True,
            verdict_class in {"partial", "null"},
            RESULT_RELATIVE_PATH,
        ),
    ]
    rows: list[JsonDict] = []
    for check, expected, observed, source_path in checks:
        passed = observed == expected
        payload = {
            "row_type": "gate_check",
            "schema_version": SCHEMA_VERSION + ".gate_check",
            "check": check,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "source_path": source_path.as_posix(),
            "score_if_unfixed": 0.0,
            "spec_refs": ["REQ-BENCH-6513"],
        }
        rows.append({**payload, "gate_check_row_hash": sha256_json(payload)})
    return rows


def _v564_score(gate_rows: Sequence[Mapping[str, Any]]) -> float:
    return 1.0 if all(row.get("passed") is True for row in gate_rows) else 0.0


def status_and_verdict(score: float, gate_rows: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if score == 1.0:
        return (
            "complete_v564_terminal_handoff_contract_ready",
            (
                "complete_v564_terminal_handoff_contract: V563 terminal failures, "
                "usable immutable content, missing Exp6511 data, and the Exp6512 "
                "score-0 block are preserved without any retired structured dependency"
            ),
        )
    failed = [row for row in gate_rows if row.get("passed") is not True]
    reason = "; ".join(f"{row.get('check')}={row.get('observed')}" for row in failed[:3])
    if not reason and gate_rows:
        reason = f"{gate_rows[0].get('check')}={gate_rows[0].get('observed')}"
    return (
        "blocked_v564_terminal_handoff_contract",
        f"blocked_v564_terminal_handoff_contract: {reason}",
    )


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    return [dict(row) for row in (tests_run or DEFAULT_TESTS_RUN)]


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = _source_hashes(repo_root)
    reducers = {
        "status": "status_and_verdict",
        "honest_verdict": "status_and_verdict",
        "verdict_class": "build_artifact",
        "prior_failure_receipts": "prior_failure_receipts",
        "historical_task_rows": "historical_task_rows",
        "immutable_input_receipts": "immutable_input_receipts",
        "allowed_direct_input_rows": "allowed_direct_input_rows",
        "forbidden_dependency_rows": "forbidden_dependency_rows",
        "retired_dependency_attack_matrix": "retired_dependency_attack_matrix",
        "v564_handoff_ready_score": "_v564_score",
        "gate_check_summary": "gate_check_summary",
        "per_unit_rows": "build_artifact",
        "aggregate_row_recomputation": "aggregate_row_recomputation",
        "preconditions_checked": "preconditions_checked",
        "protected_files_unchanged": "protected_files_unchanged",
        "inference_substrate": "constant",
        "verifier_is_oracle": "constant",
        "field_principles": "constant",
        "field_provenance": "_field_provenance",
        "random_seed": "constant",
        "duration_s": "run/build_artifact",
        "tests_run": "_tests_run_receipts",
        "reproducibility_checksum": "reproducibility_checksum",
    }
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "spec_refs": ["REQ-BENCH-6513"],
            "source_hashes": source_hashes,
            "source_paths": [path.as_posix() for path in SOURCE_RELATIVE_PATHS],
            "json_pointers": [f"/{field}"],
            "local_reducer": reducers[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    run_date: str,
    protected_before: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    planned_dependencies: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Record environment and evidence state before the final write."""

    receipt_hashes = {str(row.get("input_id")): row.get("sha256") for row in receipts}
    return {
        "schema_version": SCHEMA_VERSION + ".preconditions",
        "planning_date": run_date,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "git_head": _git_output(repo_root, ("rev-parse", "HEAD")),
        "git_status_short": _git_output(repo_root, ("status", "--short")),
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "solver_availability": _solver_state(),
        "resources": _resource_state(repo_root),
        "input_path_hashes": receipt_hashes,
        "conductor_row_identifiers": [
            {
                "event_id": row.get("event_id"),
                "task_id": row.get("task_id"),
                "line_number": row.get("line_number"),
                "conductor_row_hash": row.get("conductor_row_hash"),
            }
            for row in event_rows
        ],
        "exclusion_manifest_state": _exclusion_manifest_state(repo_root),
        "planned_structured_dependencies": list(planned_dependencies),
        "planned_structured_dependency_retired_id_violations": [
            dict(row) for row in planned_dependencies if row.get("retired_id_present") is True
        ],
        "protected_hashes_before_replay": protected_before,
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    checksum_payload = {
        key: value for key, value in payload.items() if key != "reproducibility_checksum"
    }
    return sha256_json(checksum_payload)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build the terminal Exp6513 handoff artifact."""

    start = time.perf_counter()
    repo_root = Path(repo_root)
    result_path = Path(result_path)
    protected_before = protected_file_hashes(repo_root)
    exp6504_payload = _read_json(repo_root / EXP6504_RELATIVE_PATH)
    exp6506_payload = _read_json(repo_root / EXP6506_RELATIVE_PATH)
    exp6510_payload = _read_json(repo_root / EXP6510_RELATIVE_PATH)
    exp6512_payload = _read_json(repo_root / EXP6512_RELATIVE_PATH)
    receipts = immutable_input_receipts(repo_root)
    event_rows = conductor_terminal_event_rows(repo_root)
    exp6504_replay = recompute_exp6504_direct(repo_root, exp6504_payload)
    exp6510_replay = recompute_exp6510_content(repo_root, exp6510_payload)
    prior_receipts = prior_failure_receipts(repo_root, event_rows, exp6512_payload)
    receipt_by_id = {str(row.get("input_id")): row for row in receipts}
    tasks = historical_task_rows(
        repo_root,
        exp6504_payload,
        exp6506_payload,
        exp6510_payload,
        receipt_by_id["exp6511_missing"],
        exp6512_payload,
        exp6504_replay,
        exp6510_replay,
    )
    allowed = allowed_direct_input_rows(receipts)
    forbidden = forbidden_dependency_rows()
    attacks = retired_dependency_attack_matrix()
    planned_dependencies = _planned_structured_dependencies(repo_root)
    protected_after = protected_file_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    per_unit_rows = [
        *tasks,
        *receipts,
        *event_rows,
        *allowed,
        *forbidden,
        *attacks["rows"],
    ]
    planned_violations = [
        dict(row) for row in planned_dependencies if row.get("retired_id_present") is True
    ]
    aggregate = aggregate_row_recomputation(
        per_unit_rows,
        exp6504_replay=exp6504_replay,
        exp6510_replay=exp6510_replay,
        prior_receipts=prior_receipts,
        protected=protected,
        planned_violations=planned_violations,
        attacks=attacks,
        forbidden=forbidden,
    )
    # The handoff replay attempts every unit and claims no positive result, so
    # a clean run is null. Partial would mark a finished run as retryable
    # (REQ-CONDUCTOR-VERDICT-3).
    verdict_class = "null"
    gates = gate_check_summary(
        aggregate=aggregate,
        protected=protected,
        attacks=attacks,
        verdict_class=verdict_class,
        planned_violations=planned_violations,
    )
    score = _v564_score(gates)
    status, honest = status_and_verdict(score, gates)
    preconditions = preconditions_checked(
        repo_root=repo_root,
        result_path=result_path,
        run_date=run_date,
        protected_before=protected_before,
        receipts=receipts,
        event_rows=event_rows,
        planned_dependencies=planned_dependencies,
    )
    artifact: JsonDict = {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class if score == 1.0 else "blocked",
        "prior_failure_receipts": prior_receipts,
        "historical_task_rows": tasks,
        "immutable_input_receipts": receipts,
        "allowed_direct_input_rows": allowed,
        "forbidden_dependency_rows": forbidden,
        "retired_dependency_attack_matrix": attacks,
        "v564_handoff_ready_score": score,
        "gate_check_summary": gates,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": preconditions,
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start), 6
        ),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        atomic_write_json(result_path, artifact, root=repo_root, allow_override=False)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Validate schema and fail closed on handoff drift."""

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
    if payload.get("v564_handoff_ready_score") not in (0.0, 1.0):
        errors.append("v564_handoff_ready_score must be 0.0 or 1.0")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true")
    prior = {str(row.get("receipt_id")): row for row in payload.get("prior_failure_receipts", [])}
    determinations_preserved = (
        prior.get("exp6506_bootstrap_failures", {}).get("event_count") == 3
        and prior.get("exp6510_bootstrap_failures", {}).get("event_count") == 3
        and prior.get("exp6511_missing_dataset", {}).get("exists") is False
        and prior.get("exp6512_score_zero_block", {}).get("observed_score") == 0.0
    )
    if not determinations_preserved:
        errors.append("historical determination not preserved")
    if (
        payload.get("retired_dependency_attack_matrix", {}).get("all_attacks_fail_closed")
        is not True
    ):
        errors.append("retired dependency attack false accepts")
    if planned_structured_dependency_retired_id_violations(payload):
        errors.append("planned structured dependency retired id violation")
    if (
        payload.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    gates = payload.get("gate_check_summary", [])
    all_gates_pass = bool(gates) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in gates
    )
    if payload.get("v564_handoff_ready_score") == 1.0 and not all_gates_pass:
        errors.append("v564_handoff_ready_score mismatch")
    if payload.get("v564_handoff_ready_score") == 0.0 and all_gates_pass:
        errors.append("v564_handoff_ready_score mismatch")
    # A ready handoff finished its run; its class is null, never partial
    # (REQ-CONDUCTOR-VERDICT-3, SCENARIO-CONDUCTOR-VERDICT-5).
    if payload.get("v564_handoff_ready_score") == 1.0 and (payload.get("verdict_class") != "null"):
        errors.append("ready handoff requires null verdict_class")
    if payload.get("v564_handoff_ready_score") == 0.0 and payload.get("verdict_class") != "blocked":
        errors.append("blocked handoff requires blocked verdict_class")
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
) -> JsonDict:
    """Build, write, and re-validate the production handoff."""

    start = time.perf_counter()
    artifact = build_artifact(
        repo_root=REPO_ROOT,
        result_path=result_path,
        write=True,
        duration_s=None,
        tests_run=DEFAULT_TESTS_RUN,
        run_date=date,
    )
    artifact["duration_s"] = round(time.perf_counter() - start, 6)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    atomic_write_json(result_path, artifact, root=REPO_ROOT, allow_override=False)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        payload = _read_json(result_path if result_path.is_absolute() else REPO_ROOT / result_path)
        errors = validate_artifact(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
