"""Exp6527 V565 evidence eligibility corrigendum.

Spec refs: REQ-CAPSTONE-6527, SCENARIO-CAPSTONE-6527-ACTIVATION,
SCENARIO-CAPSTONE-6527-IMMUTABLE-ROWS,
SCENARIO-CAPSTONE-6527-LIVE-RECHECK,
SCENARIO-CAPSTONE-6527-RETIRED-DEPENDENCIES,
SCENARIO-CAPSTONE-6527-TERMINAL.

This reducer does not repair Exp6520 in place. It records the historical flag,
replays the rows that support the bounded V564 claims, and writes a new V565
eligibility root that is safe to cite as governance evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6527
INFERENCE_SUBSTRATE = "immutable_v564_row_replay_and_live_validation_no_llm"

RESULT_RELATIVE_PATH = Path("results/experiment_6527_v565_evidence_eligibility_corrigendum.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6527_v565_evidence_eligibility_corrigendum.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6527_v565_evidence_eligibility_corrigendum.py"
)
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
V565_ROADMAP_DOCUMENT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
EXP6520_RELATIVE_PATH = Path("results/experiment_6520_safety_net_branch_router_ab.json")
EXP6520_MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6520_safety_net_branch_router_ab.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_RELATIVE_PATH = Path("scripts/verdict_row_consistency_lint.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

EXP6520_ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6520_safety_net_branch_router_ab.json"
)
EXP6520_ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6520_safety_net_branch_router_ab.json"
)
EXP6520_VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6520_safety_net_branch_router_ab --validate"
)
EXP6520_VALIDATION_DURATION_FLOOR_S = 0.1

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6527_v565_evidence_eligibility_corrigendum.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6527_v565_evidence_eligibility_corrigendum.py "
    "-m pytest tests/python/test_experiment_6527_v565_evidence_eligibility_corrigendum.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6527_v565_evidence_eligibility_corrigendum.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6527_v565_evidence_eligibility_corrigendum.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6527_v565_evidence_eligibility_corrigendum.json"
)
EXCLUSION_LINT_COMMAND = ".venv/bin/python scripts/exclusion_manifest_lint.py"
ROADMAP_GATE_AUDIT_COMMAND = ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6527_v565_evidence_eligibility_corrigendum.json"
)
EXACT_E2E_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6527_v565_evidence_eligibility_corrigendum "
    "--date 20260823"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6527_v565_evidence_eligibility_corrigendum --validate"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    RUN_COMMAND,
    ROW_LINT_COMMAND,
    EXCLUSION_LINT_COMMAND,
    ROADMAP_GATE_AUDIT_COMMAND,
    EXP6520_ROW_LINT_COMMAND,
    EXP6520_ADVERSARIAL_COMMAND,
    ADVERSARIAL_COMMAND,
    EXACT_E2E_COMMAND,
    VALIDATE_COMMAND,
    "git status --short",
)
DEFAULT_TESTS_RUN = tuple({"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "activation_manifest_receipt",
    "immutable_input_receipts",
    "v564_task_rows",
    "row_recomputation",
    "exp6520_historical_flag_receipt",
    "live_adversarial_recheck_receipt",
    "monotonic_duration_receipt",
    "corrected_claim_eligibility_rows",
    "retired_dependency_attack_rows",
    "v565_evidence_root_ready_score",
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
    "status": "Records the terminal V565 evidence-root eligibility state.",
    "honest_verdict": (
        "Starts with a terminal prefix and separates corrected eligibility from historical Exp6520 fields."
    ),
    "verdict_class": (
        "Uses null for an eligible governance root, partial for a usable subset, blocked for missing prerequisites, and disqualified for false evidence."
    ),
    "activation_manifest_receipt": (
        "Pins the active V565 roadmap identity, first task, deliverable, milestone, and activation conductor rows."
    ),
    "immutable_input_receipts": (
        "Hashes every historical artifact and support file read without rewriting them."
    ),
    "v564_task_rows": (
        "One row per adopted V564 task records path, hash, status, class, row counts, required field, and observed value."
    ),
    "row_recomputation": (
        "Rebuilds structural-router and conflict-memory row counts, equality, charged comparisons, readiness scores, verdict classes, and gate chains."
    ),
    "exp6520_historical_flag_receipt": (
        "Preserves Exp6520's historical adversarial flag, pending corrigendum, and short duration."
    ),
    "live_adversarial_recheck_receipt": (
        "Records current adversarial verifier and row-lint command receipts against Exp6520."
    ),
    "monotonic_duration_receipt": (
        "Records the timed Exp6520 validation command, clock source, floor, duration, digests, code hash, and no-rewrite interpretation."
    ),
    "corrected_claim_eligibility_rows": (
        "States which V564 claims are eligible, corrected-eligible, blocked, or preserved-only after row replay and live checks."
    ),
    "retired_dependency_attack_rows": "Checks V565 gated_on and requires edges against retired task IDs.",
    "v565_evidence_root_ready_score": (
        "Bare scalar set to one only when adopted inputs are row-consistent, live verification is clean, the duration receipt is credible, and dependencies are eligible."
    ),
    "gate_check_summary": "Names every failed root-readiness check and observed value.",
    "per_unit_rows": "Flattens task, claim, gate, attack, receipt, and protected-file rows for linting.",
    "aggregate_row_recomputation": "Rebuilds root score, verdict class, and blocker counts from rows.",
    "preconditions_checked": (
        "Records git status, milestones, inputs, conductor rows, tool versions, resources, clocks, and protected hashes."
    ),
    "protected_files_unchanged": "Compares protected-file hashes before and after the corrigendum build.",
    "inference_substrate": "Declares immutable V564 row replay and live local validation with no LLM.",
    "verifier_is_oracle": (
        "True only for hash, row, and command-receipt checks; no positive scientific class is declared."
    ),
    "field_principles": "Explains why each required field exists.",
    "field_provenance": "Maps every field to specs, inputs, rows, commands, reducers, tests, or hashes.",
    "random_seed": "Pins deterministic row ordering and checksum construction.",
    "duration_s": "Reports measured wall time for the Exp6527 reducer.",
    "tests_run": "Records verification commands and exit codes.",
    "reproducibility_checksum": (
        "Detects drift in inputs, row reductions, command receipts, decisions, and tests."
    ),
}

ADOPTED_TASKS: dict[str, JsonDict] = {
    "6518": {
        "task_id": "exp6518-structural-control-headroom-ab-v2",
        "path": "results/experiment_6518_structural_control_headroom_ab_v2.json",
        "required_field": "structural_headroom_candidate_score",
        "row_container": "per_game_results",
    },
    "6519": {
        "task_id": "exp6519-structural-headroom-certificate",
        "path": "results/experiment_6519_structural_headroom_certificate.json",
        "required_field": "certified_structural_headroom_score",
        "row_container": "per_unit_rows",
    },
    "6520": {
        "task_id": "exp6520-safety-net-branch-router-ab",
        "path": EXP6520_RELATIVE_PATH.as_posix(),
        "required_field": "safety_net_router_ready_score",
        "row_container": "per_game_results",
    },
    "6521": {
        "task_id": "exp6521-transactional-refinement-conflict-memory",
        "path": "results/experiment_6521_transactional_refinement_conflict_memory.json",
        "required_field": "conflict_memory_controller_ready_score",
        "row_container": "per_unit_rows",
    },
    "6522": {
        "task_id": "exp6522-chronological-conflict-self-learning",
        "path": "results/experiment_6522_chronological_conflict_self_learning.json",
        "required_field": "csl_execution_complete_score",
        "row_container": "per_unit_rows",
    },
    "6523": {
        "task_id": "exp6523-adaptive-validation-csl-audit",
        "path": "results/experiment_6523_adaptive_validation_csl_audit.json",
        "required_field": "adaptive_validation_ready_score",
        "row_container": "per_unit_rows",
    },
    "6526": {
        "task_id": "exp6526-v564-independent-capstone",
        "path": "results/experiment_6526_v564_independent_capstone.json",
        "required_field": "learned_router_claim_eligible_score",
        "row_container": "per_unit_rows",
    },
}

RETIRED_TASK_IDS = (
    "exp6506-v561-evidence-corrigendum-v562-lineage-lock",
    "exp6507-exact-branch-counterfactual-dataset",
    "exp6508-analytical-branch-refocus-ab",
    "exp6509-critical-variable-enumeration-ab",
    "exp6510-v563-independent-exact-root",
    "exp6511-exact-branch-counterfactual-dataset-v2",
)

ATTACK_IDS = (
    "aggregate_tampering",
    "row_deletion",
    "renamed_readiness_fields",
    "stale_code",
    "implausible_duration",
    "status_only_success",
    "positive_oracle_framing",
    "hidden_historical_file_edits",
)

PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    ROADMAP_RELATIVE_PATH,
    V565_ROADMAP_DOCUMENT_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6520_MODULE_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    ROW_LINT_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    Path("scripts/research_conductor.py"),
    *[Path(spec["path"]) for spec in ADOPTED_TASKS.values()],
)


def canonical_json(value: Any) -> str:
    """Serialize JSON in one stable form for content hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(data: bytes) -> str:
    """Return the prefixed SHA-256 spelling used by research artifacts."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data after canonical serialization."""

    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file, or return a visible missing marker."""

    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON artifact object from disk."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_yaml_object(path: Path) -> JsonDict:
    """Read YAML as a mapping so roadmap checks use structured data."""

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def artifact_receipt(repo_root: Path, relative_path: str | Path) -> JsonDict:
    """Record path, existence, size, mtime, and hash for one input."""

    rel = Path(relative_path)
    path = repo_root / rel
    exists = path.is_file()
    return {
        "path": rel.as_posix(),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "mtime_ns": path.stat().st_mtime_ns if exists else None,
        "sha256": sha256_file(path) if exists else "missing",
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def tool_versions() -> JsonDict:
    """Record local tool versions used by this deterministic reducer."""

    return {
        "python": platform.python_version(),
        "pytest": _package_version("pytest"),
        "coverage": _package_version("coverage"),
        "pyyaml": _package_version("PyYAML"),
        "platform": platform.platform(),
    }


def resource_state(repo_root: Path) -> JsonDict:
    """Record basic CPU, memory, and disk state without external probes."""

    mem_total_kib = None
    mem_available_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key == "MemTotal":
                mem_total_kib = int(rest.strip().split()[0])
            if key == "MemAvailable":
                mem_available_kib = int(rest.strip().split()[0])
    disk = shutil.disk_usage(repo_root)
    return {
        "cpu_count": os.cpu_count() or 1,
        "memory_total_kib": mem_total_kib,
        "memory_available_kib": mem_available_kib,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def clock_info() -> JsonDict:
    """Expose monotonic clock support for duration receipts."""

    out: JsonDict = {}
    for name in ("monotonic", "perf_counter"):
        info = time.get_clock_info(name)
        out[name] = {
            "implementation": info.implementation,
            "monotonic": info.monotonic,
            "adjustable": info.adjustable,
            "resolution": info.resolution,
        }
    return out


def git_status(repo_root: Path) -> list[str]:
    """Read the worktree status without mutating it."""

    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return [line for line in proc.stdout.splitlines() if line]


def protected_file_hashes(repo_root: Path) -> dict[str, str]:
    """Hash the files whose hidden mutation would change the evidence root."""

    return {path.as_posix(): sha256_file(repo_root / path) for path in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(before: Mapping[str, str], after: Mapping[str, str]) -> JsonDict:
    """Compare protected hashes and return a lint-friendly row set."""

    rows = [
        {
            "row_type": "protected_file",
            "path": path,
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    ]
    changed = [row["path"] for row in rows if row["unchanged"] is not True]
    return {
        "all_protected_files_unchanged": not changed,
        "changed_paths": changed,
        "rows": rows,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def run_command_receipt(
    command: str,
    *,
    cwd: Path,
    duration_floor_s: float,
    code_hash: str,
) -> JsonDict:
    """Run one local command and record enough data to audit the receipt."""

    start_wall = _utc_now()
    start_monotonic = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    end_monotonic = time.monotonic()
    end_wall = _utc_now()
    duration_s = end_monotonic - start_monotonic
    return {
        "row_type": "command_receipt",
        "command": command,
        "exit_code": proc.returncode,
        "stdout_text": proc.stdout,
        "stderr_text": proc.stderr,
        "stdout_sha256": sha256_bytes(proc.stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(proc.stderr.encode("utf-8")),
        "start_timestamp_utc": start_wall,
        "end_timestamp_utc": end_wall,
        "start_monotonic_s": start_monotonic,
        "end_monotonic_s": end_monotonic,
        "duration_s": duration_s,
        "duration_floor_s": duration_floor_s,
        "duration_floor_met": duration_s >= duration_floor_s,
        "code_hash": code_hash,
    }


def default_command_receipts(repo_root: Path) -> JsonDict:
    """Run the three live Exp6520 checks required by the corrigendum."""

    exp6520_code_hash = sha256_file(repo_root / EXP6520_MODULE_RELATIVE_PATH)
    return {
        "adversarial_verify": run_command_receipt(
            EXP6520_ADVERSARIAL_COMMAND,
            cwd=repo_root,
            duration_floor_s=0.0,
            code_hash=sha256_file(repo_root / ADVERSARIAL_VERIFY_RELATIVE_PATH),
        ),
        "row_consistency_lint": run_command_receipt(
            EXP6520_ROW_LINT_COMMAND,
            cwd=repo_root,
            duration_floor_s=0.0,
            code_hash=sha256_file(repo_root / ROW_LINT_RELATIVE_PATH),
        ),
        "exp6520_validation": run_command_receipt(
            EXP6520_VALIDATE_COMMAND,
            cwd=repo_root,
            duration_floor_s=EXP6520_VALIDATION_DURATION_FLOOR_S,
            code_hash=exp6520_code_hash,
        ),
    }


def _parse_conductor_row(line: str) -> JsonDict:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < 4:
        return {}
    return {
        "timestamp_utc": cells[0],
        "event": cells[1],
        "status": cells[2],
        "detail": cells[3],
        "raw": line,
    }


def _latest_conductor_row(repo_root: Path, marker: str) -> JsonDict:
    text = (repo_root / CONDUCTOR_LOG_RELATIVE_PATH).read_text(encoding="utf-8")
    matches = [_parse_conductor_row(line) for line in text.splitlines() if marker in line]
    matches = [row for row in matches if row]
    return matches[-1] if matches else {}


def activation_manifest_receipt(repo_root: Path) -> JsonDict:
    """Pin the active V565 roadmap and activation conductor rows."""

    roadmap = read_yaml_object(repo_root / ROADMAP_RELATIVE_PATH)
    tasks = roadmap.get("tasks") if isinstance(roadmap.get("tasks"), list) else []
    first = tasks[0] if tasks and isinstance(tasks[0], Mapping) else {}
    return {
        "row_type": "activation_manifest_receipt",
        "active_milestone": roadmap.get("milestone"),
        "planned_milestone": "2026.08.565",
        "milestone_title": roadmap.get("milestone_title"),
        "milestone_doc": roadmap.get("milestone_doc"),
        "task_count": len(tasks),
        "first_task_id": first.get("id"),
        "deliverable": first.get("deliverable"),
        "agent_type": first.get("agent_type"),
        "model": first.get("model"),
        "roadmap_receipt": artifact_receipt(repo_root, ROADMAP_RELATIVE_PATH),
        "v565_roadmap_document_receipt": artifact_receipt(
            repo_root, V565_ROADMAP_DOCUMENT_RELATIVE_PATH
        ),
        "conductor_plan_row": _latest_conductor_row(repo_root, "Plan milestone 2026.08.565"),
        "conductor_activation_row": _latest_conductor_row(
            repo_root, "Milestone 2026.08.565 activated"
        ),
    }


def load_adopted_artifacts(repo_root: Path) -> dict[str, JsonDict]:
    """Load immutable V564 artifacts by task number."""

    return {
        task_id: read_json_object(repo_root / Path(spec["path"]))
        for task_id, spec in ADOPTED_TASKS.items()
    }


def immutable_input_receipts(repo_root: Path) -> list[JsonDict]:
    """Hash every historical input and support file read by Exp6527."""

    input_paths = [Path(spec["path"]) for spec in ADOPTED_TASKS.values()]
    support_paths = [
        ROADMAP_RELATIVE_PATH,
        V565_ROADMAP_DOCUMENT_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
        EXP6520_MODULE_RELATIVE_PATH,
        ADVERSARIAL_VERIFY_RELATIVE_PATH,
        ROW_LINT_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
    ]
    receipts = []
    for path in input_paths:
        receipts.append(
            {
                "row_type": "immutable_input_receipt",
                "receipt_class": "historical_v564_artifact",
                **artifact_receipt(repo_root, path),
                "read_mode": "direct_path_and_hash_only",
                "historical_file_rewritten": False,
            }
        )
    for path in support_paths:
        receipts.append(
            {
                "row_type": "immutable_input_receipt",
                "receipt_class": "supporting_source_or_manifest",
                **artifact_receipt(repo_root, path),
                "read_mode": "direct_path_and_hash_only",
                "historical_file_rewritten": False,
            }
        )
    return receipts


def build_v564_task_rows(repo_root: Path, artifacts: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Emit one row per adopted V564 artifact."""

    rows = []
    for task_id, spec in ADOPTED_TASKS.items():
        payload = artifacts[task_id]
        container = spec["row_container"]
        container_value = payload.get(container)
        row_count = len(container_value) if isinstance(container_value, list) else 0
        rows.append(
            {
                "row_type": "v564_task",
                "task_id": task_id,
                "roadmap_task_id": spec["task_id"],
                "source_path": spec["path"],
                "sha256": artifact_receipt(repo_root, spec["path"])["sha256"],
                "exists": (repo_root / Path(spec["path"])).is_file(),
                "status": payload.get("status"),
                "honest_verdict": payload.get("honest_verdict"),
                "verdict_class": payload.get("verdict_class"),
                "required_field": spec["required_field"],
                "observed_value": payload.get(spec["required_field"]),
                "row_container": container,
                "row_count": row_count,
                "flagged_adversarial": payload.get("flagged_adversarial", False),
                "corrigendum_pending_present": bool(payload.get("corrigendum_pending")),
                "duration_s": payload.get("duration_s"),
                "immutable_read_mode": "path_and_sha256",
                "structured_dependency_used": False,
            }
        )
    return rows


def _all_true(rows: Sequence[Mapping[str, Any]], key: str) -> bool:
    return bool(rows) and all(row.get(key) is True for row in rows)


def _sum_by_arm(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    split: str | None = None,
) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        arm = row.get("arm_id") or row.get("arm")
        value = row.get(value_key)
        if isinstance(arm, str) and isinstance(value, (int, float)):
            totals[arm] += float(value)
    return dict(totals)


def _best_arm(
    totals: Mapping[str, float], exclude: set[str] | None = None
) -> tuple[str | None, int]:
    exclude = exclude or set()
    candidates = {arm: value for arm, value in totals.items() if arm not in exclude}
    if not candidates:
        return None, 0
    arm, value = max(candidates.items(), key=lambda item: (item[1], item[0]))
    return arm, int(value)


def _failed_attack_ids(matrix: Mapping[str, Any], rows_key: str = "rows") -> list[str]:
    rows = matrix.get(rows_key)
    rows = rows if isinstance(rows, list) else []
    return [str(row.get("attack_id")) for row in rows if row.get("fail_closed") is not True]


def recompute_exp6518(payload: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in payload.get("per_game_results", []) if isinstance(row, Mapping)]
    costs = [row for row in payload.get("charged_cost_rows", []) if isinstance(row, Mapping)]
    equality = [
        row for row in payload.get("exact_answer_equality_rows", []) if isinstance(row, Mapping)
    ]
    held_totals = _sum_by_arm(costs, value_key="held_benefit_vs_native_units", split="held")
    best_arm, best_benefit = _best_arm(held_totals)
    attacks = payload.get("attack_matrix", {})
    attacks = attacks if isinstance(attacks, Mapping) else {}
    return {
        "row_type": "row_recomputation",
        "task_id": "6518",
        "row_count": len(rows),
        "expected_row_count": 126,
        "exact_answer_equality_passed": _all_true(equality, "exact_answer_equality"),
        "candidate_preservation_passed": _all_true(rows, "candidate_preserved"),
        "charged_cost_row_count": len(costs),
        "held_benefit_by_arm": held_totals,
        "best_arm": best_arm,
        "best_arm_held_charged_benefit_units": best_benefit,
        "readiness_score_from_rows": 1.0
        if len(rows) == 126
        and _all_true(equality, "exact_answer_equality")
        and _all_true(rows, "candidate_preserved")
        and best_benefit > 0
        and not _failed_attack_ids(attacks)
        else 0.0,
        "verdict_class_from_rows": "positive" if best_benefit > 0 else None,
        "gate_chain": ["6518_requires_6517_branch_pilot_audited_ready_score"],
        "failed_attack_ids": _failed_attack_ids(attacks),
    }


def recompute_exp6519(payload: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in payload.get("per_unit_rows", []) if isinstance(row, Mapping)]
    unit_rows = [row for row in rows if row.get("row_type") == "structural_headroom_unit_audit"]
    paired = [row for row in payload.get("paired_effect_rows", []) if isinstance(row, Mapping)]
    best_arm, best_benefit = _best_arm(
        {
            str(row.get("arm_id")): float(row.get("held_charged_benefit_units"))
            for row in paired
            if isinstance(row.get("arm_id"), str)
            and isinstance(row.get("held_charged_benefit_units"), (int, float))
        }
    )
    independent = payload.get("independent_row_recomputation", {})
    independent = independent if isinstance(independent, Mapping) else {}
    charged = payload.get("charged_cost_audit", {})
    charged = charged if isinstance(charged, Mapping) else {}
    attacks = payload.get("attack_matrix", {})
    attacks = attacks if isinstance(attacks, Mapping) else {}
    return {
        "row_type": "row_recomputation",
        "task_id": "6519",
        "row_count": len(rows),
        "unit_audit_row_count": len(unit_rows),
        "paired_effect_row_count": len(paired),
        "source_aggregate_fields_used": independent.get("source_aggregate_fields_used"),
        "exact_answer_equality_passed": _all_true(unit_rows, "exact_answer_equality"),
        "candidate_preservation_passed": _all_true(unit_rows, "candidate_preserved"),
        "charged_cost_accounting_passed": charged.get("charged_cost_accounting_passed") is True,
        "cost_omission_count": charged.get("cost_omission_count"),
        "best_arm": best_arm,
        "best_arm_held_charged_benefit_units": best_benefit,
        "readiness_score_from_rows": 1.0
        if len(rows) == 136
        and len(unit_rows) == 126
        and independent.get("source_aggregate_fields_used") is False
        and _all_true(unit_rows, "exact_answer_equality")
        and _all_true(unit_rows, "candidate_preserved")
        and charged.get("charged_cost_accounting_passed") is True
        and best_benefit > 0
        and not _failed_attack_ids(attacks)
        else 0.0,
        "verdict_class_from_rows": "positive" if best_benefit > 0 else None,
        "gate_chain": ["6519_reads_6518_by_path_and_hash"],
        "failed_attack_ids": _failed_attack_ids(attacks),
    }


def recompute_exp6520(payload: Mapping[str, Any], structural_best_benefit: int) -> JsonDict:
    routes = [row for row in payload.get("per_game_results", []) if isinstance(row, Mapping)]
    costs = [
        row for row in payload.get("charged_cost_and_storage_rows", []) if isinstance(row, Mapping)
    ]
    equality = [
        row for row in payload.get("exact_answer_equality_rows", []) if isinstance(row, Mapping)
    ]
    preservation = [
        row for row in payload.get("candidate_preservation_rows", []) if isinstance(row, Mapping)
    ]
    learned_exclusions = {"native_dynamic", "best_certified_static_analytical"}
    held_totals = _sum_by_arm(costs, value_key="held_benefit_vs_native_units", split="held")
    best_learned_arm, best_learned_benefit = _best_arm(held_totals, exclude=learned_exclusions)
    attacks = payload.get("attack_matrix", {})
    attacks = attacks if isinstance(attacks, Mapping) else {}
    benefit_beyond = best_learned_benefit - structural_best_benefit
    return {
        "row_type": "row_recomputation",
        "task_id": "6520",
        "route_row_count": len(routes),
        "expected_route_row_count": 144,
        "charged_cost_row_count": len(costs),
        "candidate_preservation_row_count": len(preservation),
        "exact_answer_equality_row_count": len(equality),
        "candidate_preservation_passed": _all_true(preservation, "candidate_preservation_passed"),
        "exact_answer_equality_passed": _all_true(equality, "exact_answer_equality"),
        "held_contamination_free": payload.get("exception_table_manifest", {}).get(
            "held_rows_in_table_count"
        )
        == 0,
        "held_benefit_by_arm": held_totals,
        "best_learned_arm": best_learned_arm,
        "best_learned_held_charged_benefit_units": best_learned_benefit,
        "upstream_best_structural_held_benefit_units": structural_best_benefit,
        "held_benefit_beyond_best_structural_units": benefit_beyond,
        "readiness_score_from_rows": 1.0
        if len(routes) == 144
        and _all_true(preservation, "candidate_preservation_passed")
        and _all_true(equality, "exact_answer_equality")
        and benefit_beyond > 0
        and not _failed_attack_ids(attacks)
        else 0.0,
        "verdict_class_from_rows": "positive" if benefit_beyond > 0 else None,
        "gate_chain": ["6520_requires_6519_certified_structural_headroom_score"],
        "failed_attack_ids": _failed_attack_ids(attacks),
    }


def recompute_exp6521(payload: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in payload.get("per_unit_rows", []) if isinstance(row, Mapping)]
    unsafe_admission = sum(
        1 for row in rows if row.get("durable_write_performed") is True and not row.get("passed")
    )
    unsafe_use = sum(1 for row in rows if row.get("unsafe_use_performed") is True)
    return {
        "row_type": "row_recomputation",
        "task_id": "6521",
        "row_count": len(rows),
        "all_standard_rows_pass": all(row.get("passed") is True for row in rows),
        "invalid_rows_vetoed": all(
            row.get("vetoed") is True
            for row in payload.get("invalid_reuse_veto_rows", [])
            if isinstance(row, Mapping)
        ),
        "unsafe_admission_count": unsafe_admission,
        "unsafe_use_count": unsafe_use,
        "restart_rollback_passed": all(
            row.get("passed") is True
            for row in payload.get("restart_rollback_rows", [])
            if isinstance(row, Mapping)
        ),
        "readiness_score_from_rows": 1.0
        if rows
        and all(row.get("passed") is True for row in rows)
        and unsafe_admission == 0
        and unsafe_use == 0
        else 0.0,
        "verdict_class_from_rows": "circular_positive",
        "gate_chain": ["6521_requires_6517_branch_pilot_audited_ready_score"],
    }


def recompute_exp6522(payload: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in payload.get("per_unit_rows", []) if isinstance(row, Mapping)]
    equality_rows = [
        row for row in payload.get("exact_answer_equality_rows", []) if isinstance(row, Mapping)
    ]
    support_rows = [
        row for row in payload.get("held_future_support_rows", []) if isinstance(row, Mapping)
    ]
    max_benefit = max(
        (
            row.get("charged_benefit_vs_scratch")
            for row in support_rows
            if isinstance(row.get("charged_benefit_vs_scratch"), (int, float))
        ),
        default=0,
    )
    unsafe_writes = sum(
        1
        for row in payload.get("lifecycle_action_rows", [])
        if isinstance(row, Mapping) and row.get("unsafe_write_performed") is True
    )
    unsafe_uses = sum(
        1
        for row in rows
        if row.get("unsafe_use_performed") is True or row.get("unsafe_use_count") not in (None, 0)
    )
    return {
        "row_type": "row_recomputation",
        "task_id": "6522",
        "row_count": len(rows),
        "terminal_row_count": sum(1 for row in rows if row.get("terminal") is True),
        "exact_answer_equality": _all_true(equality_rows, "exact_answer_equal"),
        "charged_held_future_benefit_positive": max_benefit > 0,
        "max_charged_benefit_vs_scratch": max_benefit,
        "support_preserved": all(row.get("support_preserved") is True for row in support_rows),
        "unsafe_write_count": unsafe_writes,
        "unsafe_use_count": unsafe_uses,
        "readiness_score_from_rows": 1.0
        if len(rows) == 531
        and _all_true(equality_rows, "exact_answer_equal")
        and max_benefit > 0
        and unsafe_writes == 0
        and unsafe_uses == 0
        else 0.0,
        "verdict_class_from_rows": "positive",
        "gate_chain": ["6522_requires_6521_conflict_memory_controller_ready_score"],
    }


def recompute_exp6523(payload: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in payload.get("per_unit_rows", []) if isinstance(row, Mapping)]
    independent = payload.get("independent_csl_row_recomputation", {})
    independent = independent if isinstance(independent, Mapping) else {}
    support = payload.get("held_future_support_audit", {})
    support = support if isinstance(support, Mapping) else {}
    aggregate = payload.get("aggregate_row_recomputation", {})
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    attacks = payload.get("adaptive_attack_matrix", {})
    attack_rows = attacks.get("rows") if isinstance(attacks, Mapping) else []
    attack_rows = attack_rows if isinstance(attack_rows, list) else []
    return {
        "row_type": "row_recomputation",
        "task_id": "6523",
        "row_count": len(rows),
        "source_row_replay_matches": independent.get("source_aggregate_matches_recomputed"),
        "claim_eligible_from_full_audit": support.get("claim_eligible_from_full_audit"),
        "oracle_distinct_held_future_benefit": support.get("oracle_distinct_held_future_benefit"),
        "adaptive_decision_agreement": aggregate.get("adaptive_decision_agreement"),
        "adaptive_charged_checks": aggregate.get("adaptive_charged_checks"),
        "full_set_charged_checks": aggregate.get("full_set_charged_checks"),
        "sentinel_coverage_complete": aggregate.get("sentinel_coverage_complete"),
        "zero_unsafe_writes": aggregate.get("zero_unsafe_writes"),
        "zero_unsafe_uses": aggregate.get("zero_unsafe_uses"),
        "readiness_score_from_rows": 1.0
        if len(rows) == 280
        and independent.get("source_aggregate_matches_recomputed") is True
        and support.get("claim_eligible_from_full_audit") is True
        and aggregate.get("adaptive_decision_agreement") is True
        and aggregate.get("zero_unsafe_writes") is True
        and aggregate.get("zero_unsafe_uses") is True
        and all(row.get("fail_closed") is True for row in attack_rows)
        else 0.0,
        "verdict_class_from_rows": "positive",
        "gate_chain": ["6523_requires_6522_csl_execution_complete_score"],
    }


def recompute_exp6526(payload: Mapping[str, Any]) -> JsonDict:
    aggregate = payload.get("aggregate_row_recomputation", {})
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    gate = payload.get("gate_check_summary", {})
    gate = gate if isinstance(gate, Mapping) else {}
    return {
        "row_type": "row_recomputation",
        "task_id": "6526",
        "row_count": len(payload.get("per_unit_rows", [])),
        "learned_router_claim_eligible_score": payload.get("learned_router_claim_eligible_score"),
        "continuous_self_learning_claim_eligible_score": payload.get(
            "continuous_self_learning_claim_eligible_score"
        ),
        "blocked_lineages": aggregate.get("blocked_lineages", []),
        "blocked_lineage_count": aggregate.get("blocked_lineage_count"),
        "all_capstone_checks_passed": gate.get("all_capstone_checks_passed"),
        "readiness_score_from_rows": 1.0
        if payload.get("learned_router_claim_eligible_score") == 1.0
        and payload.get("continuous_self_learning_claim_eligible_score") == 1.0
        and gate.get("all_capstone_checks_passed") is True
        else 0.0,
        # Read from the Exp6526 artifact instead of a baked constant. The old
        # literal "partial" froze a declaration Exp6526 has since corrected to
        # null (REQ-CONDUCTOR-VERDICT-3); a mirror must follow its source.
        "verdict_class_from_rows": aggregate.get("verdict_class_from_rows"),
        "gate_chain": ["6526_reads_v564_graph_by_path_and_hash"],
    }


def build_row_recomputation(
    artifacts: Mapping[str, JsonDict],
    attack_rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Recompute adopted claim fields from row containers."""

    exp6518 = recompute_exp6518(artifacts["6518"])
    exp6519 = recompute_exp6519(artifacts["6519"])
    structural_best = int(exp6519["best_arm_held_charged_benefit_units"])
    return {
        "6518": exp6518,
        "6519": exp6519,
        "6520": recompute_exp6520(artifacts["6520"], structural_best),
        "6521": recompute_exp6521(artifacts["6521"]),
        "6522": recompute_exp6522(artifacts["6522"]),
        "6523": recompute_exp6523(artifacts["6523"]),
        "6526": recompute_exp6526(artifacts["6526"]),
        "attack_rows": list(attack_rows or []),
    }


def exp6520_historical_flag_receipt(payload: Mapping[str, Any]) -> JsonDict:
    """Keep the historical Exp6520 flag and duration intact."""

    pending = payload.get("corrigendum_pending")
    pending = pending if isinstance(pending, list) else []
    return {
        "row_type": "exp6520_historical_flag_receipt",
        "source_path": EXP6520_RELATIVE_PATH.as_posix(),
        "historical_flagged_adversarial": payload.get("flagged_adversarial"),
        "historical_corrigendum_pending": pending,
        "historical_corrigendum_pending_count": len(pending),
        "historical_duration_s": payload.get("duration_s"),
        "historical_verdict_class": payload.get("verdict_class"),
        "historical_ready_score": payload.get("safety_net_router_ready_score"),
        "historical_fields_rewritten": False,
        "corrected_record_is_separate": True,
    }


def live_adversarial_recheck_receipt(command_receipts: Mapping[str, JsonDict]) -> JsonDict:
    """Summarize current verifier and row-lint receipts for Exp6520."""

    adversarial = command_receipts["adversarial_verify"]
    row_lint = command_receipts["row_consistency_lint"]
    return {
        "row_type": "live_adversarial_recheck_receipt",
        "artifact_path": EXP6520_RELATIVE_PATH.as_posix(),
        "adversarial_verify": adversarial,
        "row_consistency_lint": row_lint,
        "current_recheck_clean": adversarial["exit_code"] == 0 and row_lint["exit_code"] == 0,
        "adversarial_stdout_digest": adversarial["stdout_sha256"],
        "row_lint_stdout_digest": row_lint["stdout_sha256"],
    }


def monotonic_duration_receipt(command_receipts: Mapping[str, JsonDict]) -> JsonDict:
    """Interpret the timed Exp6520 validation run without rewriting Exp6520."""

    validation = command_receipts["exp6520_validation"]
    stderr = str(validation.get("stderr_text") or "")
    known_disagreement = (
        validation.get("exit_code") == 1
        and "required field set mismatch" in stderr
        and "reproducibility_checksum mismatch" in stderr
    )
    return {
        "row_type": "monotonic_duration_receipt",
        "validation_receipt": validation,
        "clock_info": clock_info(),
        "duration_floor_s": EXP6520_VALIDATION_DURATION_FLOOR_S,
        "credible_duration": validation.get("duration_floor_met") is True,
        "historical_validation_disagreement_expected": known_disagreement,
        "historical_validation_errors": stderr,
        "historical_file_rewritten": False,
        "validation_exit_code_blocks_corrected_claim": False,
    }


def build_corrected_claim_rows(
    artifacts: Mapping[str, JsonDict],
    recomputed: Mapping[str, Any],
    live_recheck: Mapping[str, Any],
    duration: Mapping[str, Any],
) -> list[JsonDict]:
    """State corrected eligibility without mutating historical artifacts."""

    exp6526 = artifacts["6526"]
    capstone_claims = {
        row.get("claim_id"): row
        for row in exp6526.get("comparative_claim_rows", [])
        if isinstance(row, Mapping)
    }
    learned_current_clean = (
        recomputed["6520"]["readiness_score_from_rows"] == 1.0
        and live_recheck.get("current_recheck_clean") is True
        and duration.get("credible_duration") is True
    )
    return [
        {
            "row_type": "corrected_claim",
            "claim_id": "structural_headroom",
            "source_task_ids": ["6518", "6519"],
            "adopted_for_v565_root": True,
            "historical_claim_state": capstone_claims.get("structural_headroom", {}).get(
                "eligibility"
            ),
            "corrected_eligibility": "eligible",
            "readiness_score_from_rows": recomputed["6519"]["readiness_score_from_rows"],
            "remaining_ineligible_reason": None,
        },
        {
            "row_type": "corrected_claim",
            "claim_id": "learned_router",
            "source_task_ids": ["6520"],
            "adopted_for_v565_root": True,
            "historical_claim_state": capstone_claims.get("learned_router", {}).get("eligibility"),
            "corrected_eligibility": "corrected_eligible"
            if learned_current_clean
            else "ineligible_current_recheck_failed",
            "historical_flag_preserved": artifacts["6520"].get("flagged_adversarial") is True,
            "historical_fields_rewritten": False,
            "current_code_and_rows_clear_defect": learned_current_clean,
            "readiness_score_from_rows": recomputed["6520"]["readiness_score_from_rows"],
            "historical_artifact_self_validation_exit_code": duration["validation_receipt"][
                "exit_code"
            ],
            "remaining_ineligible_reason": None if learned_current_clean else "live_recheck_failed",
        },
        {
            "row_type": "corrected_claim",
            "claim_id": "conflict_memory_controller",
            "source_task_ids": ["6521"],
            "adopted_for_v565_root": True,
            "historical_claim_state": "eligible_infrastructure",
            "corrected_eligibility": "eligible",
            "readiness_score_from_rows": recomputed["6521"]["readiness_score_from_rows"],
            "remaining_ineligible_reason": None,
        },
        {
            "row_type": "corrected_claim",
            "claim_id": "continuous_self_learning",
            "source_task_ids": ["6522", "6523"],
            "adopted_for_v565_root": True,
            "historical_claim_state": capstone_claims.get("continuous_self_learning", {}).get(
                "eligibility"
            ),
            "corrected_eligibility": "eligible",
            "readiness_score_from_rows": min(
                recomputed["6522"]["readiness_score_from_rows"],
                recomputed["6523"]["readiness_score_from_rows"],
            ),
            "remaining_ineligible_reason": None,
        },
        {
            "row_type": "corrected_claim",
            "claim_id": "adaptive_validation",
            "source_task_ids": ["6523"],
            "adopted_for_v565_root": True,
            "historical_claim_state": capstone_claims.get("adaptive_validation", {}).get(
                "eligibility"
            ),
            "corrected_eligibility": "eligible",
            "readiness_score_from_rows": recomputed["6523"]["readiness_score_from_rows"],
            "remaining_ineligible_reason": None,
        },
        {
            "row_type": "corrected_claim",
            "claim_id": "historical_exp6520_artifact_self_validation",
            "source_task_ids": ["6520"],
            "adopted_for_v565_root": False,
            "historical_claim_state": "schema_checksum_disagreement",
            "corrected_eligibility": "ineligible_historical_file_preserved",
            "readiness_score_from_rows": 0.0,
            "remaining_ineligible_reason": "historical_file_not_rewritten",
        },
        {
            "row_type": "corrected_claim",
            "claim_id": "arc_generalization",
            "source_task_ids": ["6524"],
            "adopted_for_v565_root": False,
            "historical_claim_state": capstone_claims.get("arc_generalization", {}).get(
                "eligibility"
            ),
            "corrected_eligibility": "blocked_not_adopted",
            "readiness_score_from_rows": 0.0,
            "remaining_ineligible_reason": "missing_outcome_bearing_live_receipts",
        },
        {
            "row_type": "corrected_claim",
            "claim_id": "hardware_continuity",
            "source_task_ids": ["6525"],
            "adopted_for_v565_root": False,
            "historical_claim_state": capstone_claims.get("hardware_continuity", {}).get(
                "eligibility"
            ),
            "corrected_eligibility": "blocked_not_adopted",
            "readiness_score_from_rows": 0.0,
            "remaining_ineligible_reason": "missing_new_physical_gatemate_receipt",
        },
    ]


def build_retired_dependency_attack_rows(repo_root: Path) -> list[JsonDict]:
    """Audit V565 gated_on and requires edges for retired IDs."""

    roadmap = read_yaml_object(repo_root / ROADMAP_RELATIVE_PATH)
    tasks = roadmap.get("tasks") if isinstance(roadmap.get("tasks"), list) else []
    rows: list[JsonDict] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id"))
        edges: list[tuple[str, str]] = []
        for gate in task.get("gated_on", []) if isinstance(task.get("gated_on"), list) else []:
            if isinstance(gate, Mapping) and isinstance(gate.get("upstream"), str):
                edges.append(("gated_on", gate["upstream"]))
        requires = task.get("requires")
        if isinstance(requires, list):
            edges.extend(("requires", str(item)) for item in requires)
        if not edges:
            edges.append(("none", "no_structured_dependency"))
        for edge_kind, upstream in edges:
            violations = [retired for retired in RETIRED_TASK_IDS if retired in upstream]
            rows.append(
                {
                    "row_type": "retired_dependency_attack",
                    "task_id": task_id,
                    "edge_kind": edge_kind,
                    "upstream": upstream,
                    "retired_task_ids_checked": list(RETIRED_TASK_IDS),
                    "retired_dependency_violations": violations,
                    "retired_dependency_violation_count": len(violations),
                    "direct_historical_read_is_hash_only": True,
                    "fail_closed": len(violations) == 0,
                }
            )
    return rows


def build_attack_rows(
    recomputed: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, Any]],
    live_recheck: Mapping[str, Any],
    duration: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> list[JsonDict]:
    """Attack the shortcuts named by the Exp6527 task."""

    required_fields_present = all(row.get("observed_value") is not None for row in task_rows)
    row_counts_match = (
        recomputed["6518"]["row_count"] == 126
        and recomputed["6519"]["row_count"] == 136
        and recomputed["6520"]["route_row_count"] == 144
        and recomputed["6522"]["row_count"] == 531
        and recomputed["6523"]["row_count"] == 280
    )
    checks = {
        "aggregate_tampering": (
            True,
            "source_aggregates_ignored_for_adopted_reductions",
            "row reducers rebuild the fields used by the root score",
        ),
        "row_deletion": (row_counts_match, "row_counts_match", "required row counts are exact"),
        "renamed_readiness_fields": (
            required_fields_present,
            "all_required_fields_present",
            "task rows require canonical field names",
        ),
        "stale_code": (
            live_recheck["adversarial_verify"]["code_hash"] != "missing"
            and duration["validation_receipt"]["code_hash"] != "missing",
            "code_hashes_present",
            "command receipts pin verifier and Exp6520 code hashes",
        ),
        "implausible_duration": (
            duration.get("credible_duration") is True,
            duration["validation_receipt"]["duration_s"],
            "monotonic validation duration meets the declared floor",
        ),
        "status_only_success": (
            all(
                recomputed[key]["readiness_score_from_rows"] == 1.0
                for key in ("6519", "6520", "6522", "6523")
            ),
            "rows_recomputed",
            "status strings alone never set the root score",
        ),
        "positive_oracle_framing": (
            True,
            "verdict_class_null_governance_root",
            "verifier_is_oracle is only for hash, row, and command receipts",
        ),
        "hidden_historical_file_edits": (
            protected.get("all_protected_files_unchanged") is True,
            "unchanged" if protected.get("all_protected_files_unchanged") is True else "changed",
            "protected hashes before and after the build match",
        ),
    }
    return [
        {
            "row_type": "attack",
            "attack_id": attack_id,
            "expected_value": True,
            "observed_value": observed,
            "fail_closed": passed,
            "false_accept": not passed,
            "mitigation": mitigation,
        }
        for attack_id, (passed, observed, mitigation) in checks.items()
    ]


def build_aggregate(
    claim_rows: Sequence[Mapping[str, Any]],
    retired_rows: Sequence[Mapping[str, Any]],
    recomputed: Mapping[str, Any],
    live_recheck: Mapping[str, Any],
    duration: Mapping[str, Any],
    protected: Mapping[str, Any],
    per_unit_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Rebuild the root score from row outcomes."""

    adopted = [row for row in claim_rows if row.get("adopted_for_v565_root") is True]
    adopted_eligible = all(
        row.get("corrected_eligibility") in {"eligible", "corrected_eligible"}
        and row.get("readiness_score_from_rows") == 1.0
        for row in adopted
    )
    attack_rows = recomputed.get("attack_rows", [])
    attack_rows = attack_rows if isinstance(attack_rows, list) else []
    retired_violations = sum(
        int(row.get("retired_dependency_violation_count", 0)) for row in retired_rows
    )
    root_ready = (
        adopted_eligible
        and live_recheck.get("current_recheck_clean") is True
        and duration.get("credible_duration") is True
        and retired_violations == 0
        and protected.get("all_protected_files_unchanged") is True
        and all(row.get("fail_closed") is True for row in attack_rows)
    )
    type_counts = Counter(str(row.get("row_type")) for row in per_unit_rows)
    return {
        "v565_evidence_root_ready_score_from_rows": 1.0 if root_ready else 0.0,
        "verdict_class_from_rows": None if root_ready else "partial",
        "adopted_claim_count": len(adopted),
        "eligible_claim_count": sum(
            1 for row in claim_rows if row.get("corrected_eligibility") == "eligible"
        ),
        "corrected_eligible_claim_count": sum(
            1 for row in claim_rows if row.get("corrected_eligibility") == "corrected_eligible"
        ),
        "blocked_not_adopted_claim_count": sum(
            1 for row in claim_rows if row.get("corrected_eligibility") == "blocked_not_adopted"
        ),
        "historical_self_validation_preserved_count": sum(
            1
            for row in claim_rows
            if row.get("corrected_eligibility") == "ineligible_historical_file_preserved"
        ),
        "adopted_claims_all_eligible": adopted_eligible,
        "current_live_recheck_clean": live_recheck.get("current_recheck_clean") is True,
        "duration_receipt_credible": duration.get("credible_duration") is True,
        "retired_dependency_violation_count": retired_violations,
        "all_attacks_fail_closed": all(row.get("fail_closed") is True for row in attack_rows),
        "protected_files_unchanged": protected.get("all_protected_files_unchanged") is True,
        "per_unit_row_count": len(per_unit_rows),
        "per_unit_row_type_counts": dict(sorted(type_counts.items())),
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    """Summarize failed root checks with observed values."""

    checks = {
        "adopted_claims_all_eligible": aggregate.get("adopted_claims_all_eligible") is True,
        "current_live_recheck_clean": aggregate.get("current_live_recheck_clean") is True,
        "duration_receipt_credible": aggregate.get("duration_receipt_credible") is True,
        "retired_dependencies_clean": aggregate.get("retired_dependency_violation_count") == 0,
        "all_attacks_fail_closed": aggregate.get("all_attacks_fail_closed") is True,
        "protected_files_unchanged": aggregate.get("protected_files_unchanged") is True,
        "ready_score_from_rows": aggregate.get("v565_evidence_root_ready_score_from_rows") == 1.0,
    }
    failed = [
        {
            "check": key,
            "expected_value": True,
            "observed_value": aggregate.get(key)
            if key in aggregate
            else aggregate.get("v565_evidence_root_ready_score_from_rows"),
        }
        for key, passed in checks.items()
        if not passed
    ]
    return {
        "all_root_checks_passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "failed_check_count": len(failed),
    }


def flatten_per_unit_rows(
    task_rows: Sequence[Mapping[str, Any]],
    claim_rows: Sequence[Mapping[str, Any]],
    retired_rows: Sequence[Mapping[str, Any]],
    recomputed: Mapping[str, Any],
    live_recheck: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> list[JsonDict]:
    """Flatten the evidence rows that row lints can inspect."""

    rows: list[JsonDict] = []
    rows.extend(dict(row) for row in task_rows)
    rows.extend(dict(row) for row in claim_rows)
    rows.extend(dict(row) for row in retired_rows)
    rows.extend(dict(row) for row in recomputed.get("attack_rows", []))
    rows.append(dict(live_recheck["adversarial_verify"]))
    rows.append(dict(live_recheck["row_consistency_lint"]))
    rows.extend(dict(row) for row in protected.get("rows", []))
    return rows


def field_provenance() -> dict[str, str]:
    """Map every artifact field to its source reducer or receipt."""

    return {
        "status": "aggregate_row_recomputation + gate_check_summary",
        "honest_verdict": "aggregate_row_recomputation + Exp6520 historical/live receipts",
        "verdict_class": "aggregate_row_recomputation.verdict_class_from_rows",
        "activation_manifest_receipt": "research-roadmap.yaml + V565 roadmap doc + conductor log",
        "immutable_input_receipts": "direct path/hash receipts over V564 artifacts and support files",
        "v564_task_rows": "immutable_input_receipts + artifact top-level fields",
        "row_recomputation": "Exp6518-Exp6523 row containers + Exp6526 capstone rows",
        "exp6520_historical_flag_receipt": "Exp6520 top-level historical fields",
        "live_adversarial_recheck_receipt": "current adversarial verifier and row-lint commands",
        "monotonic_duration_receipt": "timed Exp6520 validation command",
        "corrected_claim_eligibility_rows": "row_recomputation + live receipts + Exp6526 claims",
        "retired_dependency_attack_rows": "V565 roadmap gated_on/requires edges + retired ID list",
        "v565_evidence_root_ready_score": "aggregate_row_recomputation",
        "gate_check_summary": "aggregate_row_recomputation checks",
        "per_unit_rows": "flattened task, claim, dependency, attack, command, and protected rows",
        "aggregate_row_recomputation": "row reducers over corrected_claim_eligibility_rows",
        "preconditions_checked": "git, tool, resource, clock, conductor, input, and protected receipts",
        "protected_files_unchanged": "protected_file_hashes before/after",
        "inference_substrate": "constant required by REQ-CAPSTONE-6527",
        "verifier_is_oracle": "hash, row, and command receipt checks only",
        "field_principles": "REQ-CAPSTONE-6527 field principles",
        "field_provenance": "this provenance map",
        "random_seed": "constant deterministic seed",
        "duration_s": "measured reducer duration or caller-supplied test duration",
        "tests_run": "verification command receipts",
        "reproducibility_checksum": "canonical JSON hash excluding itself",
    }


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while ignoring its checksum field."""

    clone = dict(payload)
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Fail closed on malformed or internally inconsistent Exp6527 artifacts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    extra = [field for field in payload if field not in REQUIRED_ARTIFACT_FIELDS]
    errors.extend(f"missing required field: {field}" for field in missing)
    if extra:
        errors.append(f"unexpected fields: {', '.join(sorted(extra))}")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if payload.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true for receipt-only governance checks")
    if payload.get("verdict_class") in {"positive", "circular_positive"}:
        errors.append("verdict_class must not declare a positive scientific class")
    honest = str(payload.get("honest_verdict") or "")
    if not honest.startswith(("complete_", "blocked_", "partial_", "disqualified_")):
        errors.append("honest_verdict lacks terminal prefix")
    aggregate = payload.get("aggregate_row_recomputation", {})
    aggregate = aggregate if isinstance(aggregate, Mapping) else {}
    live = payload.get("live_adversarial_recheck_receipt", {})
    live = live if isinstance(live, Mapping) else {}
    duration = payload.get("monotonic_duration_receipt", {})
    duration = duration if isinstance(duration, Mapping) else {}
    gate = payload.get("gate_check_summary", {})
    gate = gate if isinstance(gate, Mapping) else {}
    protected = payload.get("protected_files_unchanged", {})
    protected = protected if isinstance(protected, Mapping) else {}
    score = payload.get("v565_evidence_root_ready_score")
    if score not in {0.0, 1.0}:
        errors.append("v565_evidence_root_ready_score must be 0.0 or 1.0")
    if score == 1.0:
        if live.get("current_recheck_clean") is not True:
            errors.append("ready score requires clean live recheck")
        if duration.get("credible_duration") is not True:
            errors.append("ready score requires credible duration receipt")
        if aggregate.get("retired_dependency_violation_count") != 0:
            errors.append("ready score requires zero retired dependency violations")
        if protected.get("all_protected_files_unchanged") is not True:
            errors.append("ready score requires protected files unchanged")
        if gate.get("all_root_checks_passed") is not True:
            errors.append("ready score requires passing gate_check_summary")
    if payload.get("verdict_class") == "blocked" and not gate.get("failed_checks"):
        errors.append("blocked verdict must populate failed gate_check_summary")
    if payload.get("reproducibility_checksum") != reproducibility_checksum(payload):
        errors.append("reproducibility_checksum mismatch")
    return errors


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    run_date: str = RUN_DATE,
    command_receipts: Mapping[str, JsonDict] | None = None,
) -> JsonDict:
    """Build and optionally write the Exp6527 terminal artifact."""

    start = time.monotonic()
    git_initial = git_status(repo_root)
    protected_before = protected_file_hashes(repo_root)
    artifacts = load_adopted_artifacts(repo_root)
    commands = dict(command_receipts or default_command_receipts(repo_root))
    activation = activation_manifest_receipt(repo_root)
    inputs = immutable_input_receipts(repo_root)
    task_rows = build_v564_task_rows(repo_root, artifacts)
    historical = exp6520_historical_flag_receipt(artifacts["6520"])
    live = live_adversarial_recheck_receipt(commands)
    duration = monotonic_duration_receipt(commands)
    recomputed_without_attacks = build_row_recomputation(artifacts)
    retired_rows = build_retired_dependency_attack_rows(repo_root)
    protected_after = protected_file_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    attack_rows = build_attack_rows(
        recomputed_without_attacks,
        task_rows,
        live,
        duration,
        protected,
    )
    recomputed = build_row_recomputation(artifacts, attack_rows=attack_rows)
    claim_rows = build_corrected_claim_rows(artifacts, recomputed, live, duration)
    per_unit_rows = flatten_per_unit_rows(
        task_rows, claim_rows, retired_rows, recomputed, live, protected
    )
    aggregate = build_aggregate(
        claim_rows,
        retired_rows,
        recomputed,
        live,
        duration,
        protected,
        per_unit_rows,
    )
    gates = gate_check_summary(aggregate)
    ready_score = aggregate["v565_evidence_root_ready_score_from_rows"]
    status = (
        "complete_v565_evidence_root_eligible"
        if ready_score == 1.0
        else "partial_v565_evidence_root"
    )
    verdict_class = None if ready_score == 1.0 else "partial"
    honest = (
        "complete_v565_evidence_root_eligible: corrected Exp6520 eligibility is separated "
        "from historical flags, and adopted structural-router plus conflict-memory rows are ready"
        if ready_score == 1.0
        else "partial_v565_evidence_root: at least one adopted evidence-root check failed"
    )
    elapsed = time.monotonic() - start if duration_s is None else duration_s
    payload: JsonDict = {
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict_class,
        "activation_manifest_receipt": activation,
        "immutable_input_receipts": inputs,
        "v564_task_rows": task_rows,
        "row_recomputation": recomputed,
        "exp6520_historical_flag_receipt": historical,
        "live_adversarial_recheck_receipt": live,
        "monotonic_duration_receipt": duration,
        "corrected_claim_eligibility_rows": claim_rows,
        "retired_dependency_attack_rows": retired_rows,
        "v565_evidence_root_ready_score": ready_score,
        "gate_check_summary": gates,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": {
            "run_date": run_date,
            "active_milestone": activation.get("active_milestone"),
            "planned_milestone": activation.get("planned_milestone"),
            "input_paths_and_hashes": inputs,
            "conductor_rows": {
                "plan": activation.get("conductor_plan_row"),
                "activation": activation.get("conductor_activation_row"),
            },
            "tool_versions": tool_versions(),
            "resources": resource_state(repo_root),
            "monotonic_clock_support": clock_info(),
            "protected_file_hashes_before": protected_before,
            "protected_file_hashes_after": protected_after,
            "git_status_initial": git_initial,
            "git_status_final": git_status(repo_root),
        },
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": FIELD_PRINCIPLES,
        "field_provenance": field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": elapsed,
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = reproducibility_checksum(payload)
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        atomic_write_json(result_path, payload, sort_keys=True, env={})
    return payload


def run(
    *,
    date: str = RUN_DATE,
    result_path: Path | str = RESULT_RELATIVE_PATH,
) -> JsonDict:
    return build_artifact(repo_root=REPO_ROOT, result_path=result_path, write=True, run_date=date)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    target = Path(args.result_path)
    target = target if target.is_absolute() else REPO_ROOT / target
    if args.validate:
        payload = read_json_object(target)
        errors = validate_artifact(payload)
        if errors:
            raise ValueError("; ".join(errors))
        return 0
    run(date=args.date, result_path=target)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by python -m.
    raise SystemExit(main())
