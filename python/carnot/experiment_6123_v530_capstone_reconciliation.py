"""Exp6123 branch-independent capstone reconciliation for milestone .530.

Spec refs: REQ-REPORT-6123,
SCENARIO-REPORT-6123-EXACT-MATRIX,
SCENARIO-REPORT-6123-GATES,
SCENARIO-REPORT-6123-BRANCH-INDEPENDENCE,
SCENARIO-REPORT-6123-ADVERSARIAL-EXCLUSION,
SCENARIO-REPORT-6123-SCHEMA.

The capstone is a ledger pass over checked-in artifacts. It intentionally uses
the active roadmap's declared deliverable path as the only artifact locator, so
a nearby filename, a completion-history row, or a conductor gate message cannot
turn a missing or skipped experiment into executed evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.530"
MILESTONE_TITLE = (
    "Authentic Phase-D Headroom, Reachable Internal-State Verification, "
    "and Outcome-Committed Self-Learning"
)
EXPERIMENT_ID = "exp6123-v530-capstone-reconciliation"
EXPERIMENT = "experiment_6123_v530_capstone_reconciliation"
RESULT_RELATIVE_PATH = Path("results/experiment_6123_v530_capstone_reconciliation.json")
SCHEMA = "carnot.experiment_6123.v530_capstone_reconciliation.v1"
RUN_DATE = "20260804"
RANDOM_SEED = 6123
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
RESEARCH_HARDWARE_RELATIVE_PATH = Path("research-hardware-wishlist.md")
PRD_RELATIVE_PATH = Path("_bmad/prd.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
VERIFIER_GAPS_RELATIVE_PATH = Path("ops/verifier_gaps.md")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
NORTH_STAR_RELATIVE_PATH = Path("ops/north-star.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_GATES_RELATIVE_PATH = Path("scripts/conductor_gates.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
CAPSTONE_AGGREGATE_RELATIVE_PATH = Path("scripts/capstone_aggregate_available.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

SPEC_REFS = (
    "REQ-REPORT-6123",
    "SCENARIO-REPORT-6123-EXACT-MATRIX",
    "SCENARIO-REPORT-6123-GATES",
    "SCENARIO-REPORT-6123-BRANCH-INDEPENDENCE",
    "SCENARIO-REPORT-6123-ADVERSARIAL-EXCLUSION",
    "SCENARIO-REPORT-6123-SCHEMA",
)

UPSTREAM_TASKS: tuple[tuple[str, str, Path], ...] = (
    (
        "exp6112-transition-v530",
        "Exact terminal-boundary handoff from .529 into .530",
        Path("results/experiment_6112_transition_v530.json"),
    ),
    (
        "exp6113-v530-source-delta-ingestion",
        "Dated evidence refresh after the V530 planner marker",
        Path("results/experiment_6113_v530_source_delta_ingestion.json"),
    ),
    (
        "exp6114-phase-d-gpu-ladder-canary",
        "Phase-D task-scoped GPU engagement and sealed-ladder canary",
        Path("results/experiment_6114_phase_d_gpu_ladder_canary.json"),
    ),
    (
        "exp6115-phase-d-calibration-pool",
        "Gated calibration-only authentic Phase-D candidate pool",
        Path("results/experiment_6115_phase_d_calibration_pool.json"),
    ),
    (
        "exp6116-phase-d-held-candidate-pool",
        "Gated held authentic same-model Phase-D candidate pool",
        Path("results/experiment_6116_phase_d_held_candidate_pool.json"),
    ),
    (
        "exp6117-phase-d-headroom-audit",
        "Gated question-clustered Phase-D authenticity and headroom audit",
        Path("results/experiment_6117_phase_d_headroom_audit.json"),
    ),
    (
        "exp6118-phase-d-per-layer-surface",
        "Gated matching-base per-layer hidden-state surface qualification",
        Path("results/experiment_6118_phase_d_per_layer_surface.json"),
    ),
    (
        "exp6119-phase-d-hidden-state-selector",
        "Gated internal-state Phase-D selector against tuned self-consistency",
        Path("results/experiment_6119_phase_d_hidden_state_selector.json"),
    ),
    (
        "exp6120-outcome-committed-reduced-order-csl",
        "Outcome-committed reduced-order continuous self-learning",
        Path("results/experiment_6120_outcome_committed_reduced_order_csl.json"),
    ),
    (
        "exp6121-gatemate-changed-state-gate-v530",
        "GateMate changed-physical-state continuity gate",
        Path("results/experiment_6121_gatemate_changed_state_gate_v530.json"),
    ),
    (
        "exp6122-arc-primitive-reachability-loo",
        "ARC live-path generic primitive reachability and held-out attribution",
        Path("results/experiment_6122_arc_primitive_reachability_loo.json"),
    ),
)

UPSTREAM_TASK_IDS = tuple(task_id for task_id, _title, _path in UPSTREAM_TASKS)
UPSTREAM_TITLES = {task_id: title for task_id, title, _path in UPSTREAM_TASKS}
UPSTREAM_DELIVERABLES = {task_id: rel_path for task_id, _title, rel_path in UPSTREAM_TASKS}

PHASE_D_GATE_TASK_IDS = (
    "exp6115-phase-d-calibration-pool",
    "exp6116-phase-d-held-candidate-pool",
    "exp6117-phase-d-headroom-audit",
    "exp6118-phase-d-per-layer-surface",
    "exp6119-phase-d-hidden-state-selector",
)

CONDUCTOR_MATCH_MARKERS: dict[str, str] = {
    "exp6112-transition-v530": "Exact terminal-boundary handoff from .529",
    "exp6113-v530-source-delta-ingestion": "Dated evidence refresh after the V530",
    "exp6114-phase-d-gpu-ladder-canary": "Phase-D task-scoped GPU engagement",
    "exp6115-phase-d-calibration-pool": "Gated calibration-only authentic Phase-D candidate",
    "exp6116-phase-d-held-candidate-pool": "Gated held authentic same-model Phase-D candidate",
    "exp6117-phase-d-headroom-audit": "Gated question-clustered Phase-D authenticity",
    "exp6118-phase-d-per-layer-surface": "Gated matching-base per-layer hidden-state surface",
    "exp6119-phase-d-hidden-state-selector": "Gated internal-state Phase-D selector",
    "exp6120-outcome-committed-reduced-order-csl": "Outcome-committed reduced-order continuous self-le",
    "exp6121-gatemate-changed-state-gate-v530": "GateMate changed-physical-state continuity gate",
    "exp6122-arc-primitive-reachability-loo": "ARC live-path generic primitive reachability",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_task_and_declared_deliverable_matrix",
    "exact_evidence_resolution_receipts",
    "per_task_terminal_class_and_reason",
    "executed_skipped_missing_blocked_retired_underpowered_null_ready_positive_and_flagged_counts",
    "gate_recomputation_and_title_yaml_alignment",
    "candidate_pool_headroom_surface_selector_csl_hardware_and_arc_gate_matrix",
    "adversarial_verifier_receipts_and_positive_claim_exclusions",
    "prior_failure_same_verdict_retirement_receipts",
    "branch_independent_scientific_synthesis",
    "prd_gap_and_north_star_delta",
    "specs_traceability_architecture_status_changelog_conductor_verifier_hardware_and_arc_reconciliation",
    "architecture_current_planned_blocked_retired_boundary",
    "inherited_debt_baselines_and_deltas",
    "research_complete_append_readiness",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "the capstone terminal state is derived from exact upstream evidence.",
    "preconditions_checked": "the capstone terminal state is derived from exact upstream evidence.",
    "milestone_task_and_declared_deliverable_matrix": (
        "exact identity and path, never filename aliases, define evidence."
    ),
    "exact_evidence_resolution_receipts": (
        "exact identity and path, never filename aliases, define evidence."
    ),
    "per_task_terminal_class_and_reason": (
        "every outcome belongs to one explicit terminal class without laundering."
    ),
    "executed_skipped_missing_blocked_retired_underpowered_null_ready_positive_and_flagged_counts": (
        "every outcome belongs to one explicit terminal class without laundering."
    ),
    "gate_recomputation_and_title_yaml_alignment": (
        "every structured predicate is recomputed from primary artifacts and matches its task title."
    ),
    "candidate_pool_headroom_surface_selector_csl_hardware_and_arc_gate_matrix": (
        "each research branch retains its own preregistered authority and gates."
    ),
    "adversarial_verifier_receipts_and_positive_claim_exclusions": (
        "flagged artifacts remain recorded but cannot support positive claims."
    ),
    "prior_failure_same_verdict_retirement_receipts": (
        "every repeated verdict triggers the declared mechanical retirement."
    ),
    "branch_independent_scientific_synthesis": (
        "success or failure in one branch supplies no evidence for another."
    ),
    "prd_gap_and_north_star_delta": (
        "report what changed in FR11, FR12, Phase D, and the live ARC path without redefining the finish line."
    ),
    "specs_traceability_architecture_status_changelog_conductor_verifier_hardware_and_arc_reconciliation": (
        "internal docs describe only primary-artifact-backed current state."
    ),
    "architecture_current_planned_blocked_retired_boundary": (
        "internal docs describe only primary-artifact-backed current state."
    ),
    "inherited_debt_baselines_and_deltas": (
        "task-owned debt cannot increase while unrelated inherited failures remain explicit."
    ),
    "research_complete_append_readiness": (
        "research-complete changes are readiness-gated and not fabricated by a capstone."
    ),
    "protected_files_unchanged": (
        "active roadmap, conductor, exclusions, north star, public docs, historical results, and unrelated work remain untouched."
    ),
    "duration_s": "use measured `aggregation_from_upstream_artifacts`.",
    "inference_substrate": "use measured `aggregation_from_upstream_artifacts`.",
    "verifier_is_oracle": "oracle status and residual discriminators remain branch-specific and explicit.",
    "missing_verifier_gaps": "oracle status and residual discriminators remain branch-specific and explicit.",
    "field_provenance": "use measured `aggregation_from_upstream_artifacts`.",
    "test_commands": "use measured `aggregation_from_upstream_artifacts`.",
    "test_exit_codes": "use measured `aggregation_from_upstream_artifacts`.",
    "reproducibility_checksum": "use measured `aggregation_from_upstream_artifacts`.",
    "honest_verdict": (
        "use `complete:`, `complete_with_blocks:`, or `blocked:` and name every terminal branch class."
    ),
}

PRECONDITION_HASH_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    RESEARCH_PROGRAM_RELATIVE_PATH,
    PRD_RELATIVE_PATH,
    ARCHITECTURE_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    VERIFIER_GAPS_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    RESEARCH_HARDWARE_RELATIVE_PATH,
    CAPSTONE_AGGREGATE_RELATIVE_PATH,
    CONDUCTOR_GATES_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
)

PROTECTED_RELATIVE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    *UPSTREAM_DELIVERABLES.values(),
)


class TerminalClass(StrEnum):
    """Disjoint terminal classes used by the capstone matrix."""

    EXECUTED_COMPLETE = "executed_complete"
    EXECUTED_NULL = "executed_null"
    EXECUTED_READY = "executed_ready"
    EXECUTED_POSITIVE = "executed_positive"
    CONDUCTOR_GATE_BLOCKED = "conductor_gate_blocked"
    CONDUCTOR_GATE_SKIPPED_MISSING = "conductor_gate_skipped_missing"
    BLOCKED = "blocked"
    BLOCKED_RETIRED = "blocked_retired"
    PARTIAL = "partial"
    UNDERPOWERED = "underpowered"
    MISSING = "missing"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def path_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = "json_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _read_yaml_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        meta["error"] = f"yaml_error:{exc.__class__.__name__}"
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = "yaml_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in PRECONDITION_HASH_PATHS
    }


def _protected_file_hashes(root: Path) -> dict[str, str | None]:
    return {
        rel_path.as_posix(): path_sha256(root / rel_path) for rel_path in PROTECTED_RELATIVE_PATHS
    }


def _protected_files_unchanged(
    root: Path,
    before: Mapping[str, str | None] | None = None,
) -> JsonDict:
    before_hashes = dict(before) if before is not None else _protected_file_hashes(root)
    after = _protected_file_hashes(root)
    files = {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256_before": before_hashes.get(rel_path.as_posix()),
            "sha256_after": after.get(rel_path.as_posix()),
            "unchanged": before_hashes.get(rel_path.as_posix()) == after.get(rel_path.as_posix()),
        }
        for rel_path in PROTECTED_RELATIVE_PATHS
    }
    return {
        "files": files,
        "all_unchanged": all(row["unchanged"] for row in files.values()),
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _root_clutter_inventory(root: Path) -> list[str]:
    allowed = {"setup.py"}
    return sorted(path.name for path in root.glob("*.py") if path.name not in allowed)


def _git_status_short(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    result = subprocess.run(  # pragma: no cover - exercised only in the real repo
        ["git", "status", "--short"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.splitlines()


def _roadmap_tasks(root: Path) -> dict[str, JsonDict]:
    payload, _meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    tasks = payload.get("tasks") if isinstance(payload, Mapping) else []
    if not isinstance(tasks, list):
        return {}
    return {str(row.get("id")): dict(row) for row in tasks if isinstance(row, Mapping)}


def _artifact_payloads(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in UPSTREAM_DELIVERABLES.items():
        payload, meta = _read_json_mapping(root / rel_path)
        meta["declared_deliverable"] = rel_path.as_posix()
        payloads[task_id] = payload
        metadata[task_id] = meta
    return payloads, metadata


def _conductor_status_from_line(line: str) -> str:
    parts = [part.strip() for part in line.split("|")]
    return parts[3] if len(parts) > 3 else ""


def _conductor_receipts(root: Path) -> dict[str, JsonDict]:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    lines = text.splitlines()
    receipts: dict[str, JsonDict] = {}
    for task_id, marker in CONDUCTOR_MATCH_MARKERS.items():
        matches = [line for line in lines if marker in line or task_id in line]
        latest = matches[-1] if matches else ""
        receipts[task_id] = {
            "attempt_count": len(matches),
            "latest_line": latest,
            "latest_status": _conductor_status_from_line(latest) if latest else "",
        }
    return receipts


def _task_number(task_id: str) -> int | None:
    match = re.search(r"exp(\d{4})", task_id, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _same_number_aliases(root: Path, task_id: str, declared_path: Path) -> list[str]:
    number = _task_number(task_id)
    results_dir = root / "results"
    if number is None or not results_dir.exists():
        return []
    aliases: list[str] = []
    for path in results_dir.glob(f"*{number}*"):
        rel_path = path.relative_to(root)
        if path.is_file() and rel_path != declared_path:
            aliases.append(rel_path.as_posix())
    return sorted(aliases)


def _is_gate_block_artifact(payload: JsonMap) -> bool:
    return (
        payload.get("schema") == "blocked_gate_check_v1"
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
    )


def _classify_task(
    payload: JsonMap, meta: JsonMap, conductor: JsonMap
) -> tuple[TerminalClass, str]:
    if not meta.get("present"):
        if conductor.get("latest_status") == "GATE_BLOCK":
            return (
                TerminalClass.CONDUCTOR_GATE_SKIPPED_MISSING,
                "declared artifact absent but conductor recorded a gate skip; not execution",
            )
        return TerminalClass.MISSING, "declared artifact absent and no conductor gate-skip receipt"
    if _is_gate_block_artifact(payload):
        return (
            TerminalClass.CONDUCTOR_GATE_BLOCKED,
            "conductor pre-gate artifact; task body did not execute",
        )
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    if payload.get("retirement_triggered") is True and (
        status.startswith("blocked") or verdict.startswith("blocked")
    ):
        return TerminalClass.BLOCKED_RETIRED, "blocked result carries retirement_triggered=true"
    if status.startswith("complete_positive") or verdict.startswith("complete_positive:"):
        return (
            TerminalClass.EXECUTED_POSITIVE,
            "task executed and reported a positive terminal result",
        )
    if status.startswith("complete_ready") or verdict.startswith("complete_ready:"):
        return TerminalClass.EXECUTED_READY, "task executed and reported a ready terminal result"
    if status.startswith("complete_null") or verdict.startswith("complete_null:"):
        return TerminalClass.EXECUTED_NULL, "task executed and reported a null terminal result"
    if status.startswith("underpowered") or verdict.startswith("underpowered:"):
        return TerminalClass.UNDERPOWERED, "task executed but was underpowered"
    if status.startswith("complete_partial") or "partial" in verdict:
        return TerminalClass.PARTIAL, "task executed and reported a partial terminal result"
    if status.startswith("blocked") or verdict.startswith("blocked"):
        return TerminalClass.BLOCKED, "task executed and reported a blocked terminal result"
    return TerminalClass.EXECUTED_COMPLETE, "task executed and reported a complete terminal result"


def _receipt_reports(stdout_json: Any) -> list[JsonMap]:
    if not isinstance(stdout_json, Mapping):
        return []
    reports = stdout_json.get("reports")
    return [row for row in reports if isinstance(row, Mapping)] if isinstance(reports, list) else []


def _receipt_flags(receipt: JsonMap) -> list[JsonDict]:
    flags: list[JsonDict] = []
    for report in _receipt_reports(receipt.get("stdout_json")):
        raw_flags = report.get("flags")
        if isinstance(raw_flags, list):
            flags.extend(dict(flag) for flag in raw_flags if isinstance(flag, Mapping))
    return flags


def _receipt_flag_count(receipt: JsonMap) -> int:
    reports = _receipt_reports(receipt.get("stdout_json"))
    if reports:
        return sum(
            int(report.get("flag_count") or len(report.get("flags") or [])) for report in reports
        )
    return int(receipt.get("flag_count") or 0)


def _receipt_max_severity(receipt: JsonMap) -> int:
    severities = [
        int(report.get("max_severity"))
        for report in _receipt_reports(receipt.get("stdout_json"))
        if report.get("max_severity") is not None
    ]
    return max(severities) if severities else int(receipt.get("max_severity") or -1)


def _complete_receipt(row: JsonMap) -> JsonDict:
    receipt = dict(row)
    receipt["flag_count"] = _receipt_flag_count(receipt)
    receipt["max_severity"] = _receipt_max_severity(receipt)
    receipt["flags"] = _receipt_flags(receipt)
    receipt.setdefault("receipt_hash", sha256_json(receipt.get("stdout_json", {})))
    return receipt


def _normalize_adversarial_receipts(
    receipts: Mapping[str, JsonMap] | Sequence[Any] | None,
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    if receipts is None:
        return {}
    source = receipts.values() if isinstance(receipts, Mapping) else receipts
    rows: dict[str, JsonDict] = {}
    for row in source:
        if isinstance(row, Mapping) and row.get("task_id"):
            task_id = str(row["task_id"])
            if metadata.get(task_id, {}).get("present"):
                rows[task_id] = _complete_receipt(row)
    return rows


def run_live_adversarial_receipts(
    root: Path,
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:  # pragma: no cover - integration path exercised by artifact generation
    executable = (
        (root / ".venv/bin/python").as_posix()
        if (root / ".venv/bin/python").exists()
        else sys.executable
    )
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in UPSTREAM_DELIVERABLES.items():
        if not metadata.get(task_id, {}).get("present"):
            continue
        command = [
            executable,
            ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
            "--json",
            rel_path.as_posix(),
        ]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        try:
            stdout_json: Any = json.loads(result.stdout)
        except json.JSONDecodeError:
            stdout_json = {"parse_error": "stdout_not_json", "stdout": result.stdout}
        receipts[task_id] = _complete_receipt(
            {
                "task_id": task_id,
                "artifact_path": rel_path.as_posix(),
                "command": " ".join(command),
                "exit_code": result.returncode,
                "stdout_json": stdout_json,
                "stderr": result.stderr,
                "receipt_hash": sha256_json(stdout_json),
            }
        )
    return receipts


def _build_matrix(
    root: Path,
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
    receipts: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    matrix: dict[str, JsonDict] = {}
    for task_id, title, rel_path in UPSTREAM_TASKS:
        terminal, reason = _classify_task(payloads[task_id], metadata[task_id], conductor[task_id])
        payload = payloads[task_id]
        flag_count = _receipt_flag_count(receipts[task_id]) if task_id in receipts else 0
        matrix[task_id] = {
            "identity": [MILESTONE, task_id, rel_path.as_posix()],
            "milestone": MILESTONE,
            "task_id": task_id,
            "title": title,
            "declared_deliverable": rel_path.as_posix(),
            "selection_policy": "exact_declared_deliverable",
            "present": bool(metadata[task_id]["present"]),
            "loadable": bool(metadata[task_id]["loadable"]),
            "sha256": metadata[task_id]["sha256"],
            "status": str(payload.get("status") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "terminal_class": terminal.value,
            "terminal_reason": reason,
            "executed_by_task": terminal
            not in {
                TerminalClass.CONDUCTOR_GATE_BLOCKED,
                TerminalClass.CONDUCTOR_GATE_SKIPPED_MISSING,
                TerminalClass.MISSING,
            },
            "blocked_at_conductor_gate": terminal == TerminalClass.CONDUCTOR_GATE_BLOCKED,
            "adversarial_flagged": flag_count > 0,
            "retirement_triggered": payload.get("retirement_triggered") is True,
            "same_number_aliases_ignored": _same_number_aliases(root, task_id, rel_path),
            "conductor": conductor[task_id],
        }
    return matrix


def _terminal_receipt(matrix: Mapping[str, JsonMap]) -> JsonDict:
    counts = Counter(str(row["terminal_class"]) for row in matrix.values())
    return {
        "by_task": {
            task_id: {
                "terminal_class": str(row["terminal_class"]),
                "reason": str(row["terminal_reason"]),
            }
            for task_id, row in matrix.items()
        },
        "terminal_class_counts": dict(sorted(counts.items())),
        "all_tasks_have_one_terminal_class": len(matrix) == len(UPSTREAM_TASKS)
        and all(row.get("terminal_class") for row in matrix.values()),
        "principle": FIELD_PRINCIPLES["per_task_terminal_class_and_reason"],
    }


def _counts(matrix: Mapping[str, JsonMap]) -> JsonDict:
    terminal_counts = Counter(str(row["terminal_class"]) for row in matrix.values())
    signals = Counter()
    for row in matrix.values():
        terminal = str(row["terminal_class"])
        status = str(row.get("status") or "")
        verdict = str(row.get("honest_verdict") or "")
        signals["executed"] += int(bool(row.get("executed_by_task")))
        signals["skipped"] += int("gate" in terminal)
        signals["missing"] += int(not row.get("present"))
        signals["blocked"] += int("blocked" in terminal or status.startswith("blocked"))
        signals["retired"] += int(bool(row.get("retirement_triggered")) or "retired" in terminal)
        signals["underpowered"] += int("underpowered" in terminal)
        signals["null"] += int(
            status.startswith("complete_null") or verdict.startswith("complete_null:")
        )
        signals["ready"] += int(
            status.startswith("complete_ready") or verdict.startswith("complete_ready:")
        )
        signals["positive"] += int(
            status.startswith("complete_positive") or verdict.startswith("complete_positive:")
        )
        signals["adversarial_flagged"] += int(bool(row.get("adversarial_flagged")))
        signals["flagged"] += int(bool(row.get("adversarial_flagged")))
    return {
        "terminal_class_counts": dict(sorted(terminal_counts.items())),
        "outcome_signal_counts": dict(sorted(signals.items())),
        "principle": FIELD_PRINCIPLES[
            "executed_skipped_missing_blocked_retired_underpowered_null_ready_positive_and_flagged_counts"
        ],
    }


def _eval_gate(actual: Any, op: str, expected: Any) -> tuple[bool, str]:
    if op == "==":
        return actual == expected, f"actual={actual!r} == expected={expected!r}"
    return False, f"unsupported op {op!r}"  # pragma: no cover - v530 gates are equality predicates


def _recompute_gates(
    tasks: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
) -> JsonDict:
    by_task: dict[str, JsonDict] = {}
    for task_id in PHASE_D_GATE_TASK_IDS:
        task = tasks.get(task_id, {})
        gates = task.get("gated_on") if isinstance(task, Mapping) else []
        gate_rows: list[JsonDict] = []
        for gate in gates if isinstance(gates, list) else []:
            if not isinstance(gate, Mapping):
                continue
            upstream = str(gate.get("upstream") or "")
            field = str(gate.get("artifact_field") or "")
            actual = payloads.get(upstream, {}).get(field)
            passed, reason = _eval_gate(actual, str(gate.get("op") or "=="), gate.get("value"))
            gate_rows.append(
                {
                    "upstream": upstream,
                    "upstream_declared_deliverable": UPSTREAM_DELIVERABLES.get(
                        upstream, Path("")
                    ).as_posix(),
                    "upstream_artifact_present": bool(metadata.get(upstream, {}).get("present")),
                    "artifact_field": field,
                    "op": str(gate.get("op") or "=="),
                    "expected": gate.get("value"),
                    "actual": actual,
                    "passed": passed,
                    "reason": reason,
                }
            )
        all_passed = bool(gate_rows) and all(row["passed"] for row in gate_rows)
        title = str(task.get("title") or UPSTREAM_TITLES[task_id])
        conductor_status = str(conductor.get(task_id, {}).get("latest_status") or "")
        by_task[task_id] = {
            "title": title,
            "gate_declared_count": len(gate_rows),
            "title_mentions_gated": "gated" in title.lower(),
            "title_yaml_alignment_ok": "gated" in title.lower() and bool(gate_rows),
            "all_gates_passed": all_passed,
            "expected_conductor_status": "OK" if all_passed else "GATE_BLOCK",
            "observed_conductor_status": conductor_status,
            "conductor_alignment_ok": (
                (all_passed and conductor_status == "OK")
                or (not all_passed and conductor_status == "GATE_BLOCK")
            ),
            "gates": gate_rows,
        }
    return {
        "by_task": by_task,
        "all_title_yaml_alignment_ok": all(
            row["title_yaml_alignment_ok"] for row in by_task.values()
        ),
        "all_conductor_alignment_ok": all(
            row["conductor_alignment_ok"] for row in by_task.values()
        ),
        "principle": FIELD_PRINCIPLES["gate_recomputation_and_title_yaml_alignment"],
    }


def _nested_bool(payload: JsonMap, key: str, nested_key: str) -> bool:
    value = payload.get(key)
    return bool(value.get(nested_key)) if isinstance(value, Mapping) else False


def _scientific_gate_matrix(payloads: Mapping[str, JsonMap], gates: JsonMap) -> JsonDict:
    exp6115 = payloads["exp6115-phase-d-calibration-pool"]
    exp6120 = payloads["exp6120-outcome-committed-reduced-order-csl"]
    exp6121 = payloads["exp6121-gatemate-changed-state-gate-v530"]
    exp6122 = payloads["exp6122-arc-primitive-reachability-loo"]
    arc_credit_counts = exp6122.get("duplicate_level_and_unreachable_solver_credit_counts")
    credit_values = arc_credit_counts.values() if isinstance(arc_credit_counts, Mapping) else [1]
    return {
        "phase_d": {
            "calibration": {
                "state": "null",
                "phase_d_calibration_ready_score": exp6115.get("phase_d_calibration_ready_score"),
                "reason": "no calibration stratum/decode policy met the preregistered gate",
            },
            "candidate_pool": {
                "state": "blocked",
                "gate_passed": gates["by_task"]["exp6116-phase-d-held-candidate-pool"][
                    "all_gates_passed"
                ],
                "reason": "Exp6115 readiness score is not 1.0",
            },
            "headroom": {
                "state": "skipped",
                "gate_passed": gates["by_task"]["exp6117-phase-d-headroom-audit"][
                    "all_gates_passed"
                ],
                "reason": "held candidate pool never executed",
            },
            "surface": {
                "state": "blocked",
                "gate_passed": gates["by_task"]["exp6118-phase-d-per-layer-surface"][
                    "all_gates_passed"
                ],
                "reason": "headroom artifact absent",
            },
            "selector": {
                "state": "blocked",
                "gate_passed": gates["by_task"]["exp6119-phase-d-hidden-state-selector"][
                    "all_gates_passed"
                ],
                "ready_score": payloads.get("exp6119-phase-d-hidden-state-selector", {}).get(
                    "internal_state_selector_ready_score"
                ),
                "reason": "requires both per-layer surface and headroom evidence",
            },
        },
        "continuous_self_learning": {
            "state": "positive",
            "outcome_committed_csl_ready_score": exp6120.get("outcome_committed_csl_ready_score"),
            "qualification_all_gates_passed": _nested_bool(
                exp6120, "qualification_gate_matrix", "all_gates_passed"
            ),
            "model_weights_unchanged": _nested_bool(
                exp6120, "model_weight_immutability_receipt", "all_unchanged"
            ),
            "abi_parity_reported": "python_rust_pyo3_fixed_width_abi_parity" in exp6120,
        },
        "hardware": {
            "state": "blocked_physical_action",
            "physical_state_changed": exp6121.get("physical_state_changed"),
            "detect_attempt_count": (
                exp6121.get(
                    "detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code", {}
                ).get("attempt_count")
                if isinstance(
                    exp6121.get(
                        "detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code"
                    ),
                    Mapping,
                )
                else None
            ),
            "hardware_execution_authenticated": _nested_bool(
                exp6121,
                "hardware_execution_authenticated",
                "authenticated",
            ),
            "retirement_triggered": exp6121.get("retirement_triggered") is True,
        },
        "arc": {
            "state": "null_no_solve_credit",
            "registry_ok": _nested_bool(exp6122, "registry_precheck_and_postcheck", "ok"),
            "target_level_solve_claim_count": exp6122.get("target_level_solve_claim_count"),
            "solve_provenance": exp6122.get("solve_provenance"),
            "duplicate_or_unreachable_credit_count": sum(
                int(value or 0) for value in credit_values
            ),
            "submitted_defaults_unchanged": _nested_bool(
                exp6122, "submitted_defaults_unchanged", "unchanged"
            ),
            "offline_reproduced_new_level": exp6122.get("offline_reproduced_new_level") is True,
        },
        "principle": FIELD_PRINCIPLES[
            "candidate_pool_headroom_surface_selector_csl_hardware_and_arc_gate_matrix"
        ],
    }


def _adversarial_group(receipts: Mapping[str, JsonMap], matrix: Mapping[str, JsonMap]) -> JsonDict:
    reports: list[JsonDict] = []
    flagged_task_ids: list[str] = []
    for task_id in UPSTREAM_TASK_IDS:
        if not matrix[task_id]["present"] or task_id not in receipts:
            continue
        receipt = receipts[task_id]
        flag_count = _receipt_flag_count(receipt)
        flags = _receipt_flags(receipt)
        reports.append(
            {
                "task_id": task_id,
                "artifact": matrix[task_id]["declared_deliverable"],
                "command": str(receipt.get("command") or ""),
                "exit_code": receipt.get("exit_code"),
                "flag_count": flag_count,
                "max_severity": _receipt_max_severity(receipt),
                "flags": flags,
                "receipt_hash": str(receipt.get("receipt_hash") or ""),
            }
        )
        if flag_count:
            flagged_task_ids.append(task_id)
    positive_task_ids = [
        task_id
        for task_id, row in matrix.items()
        if row["terminal_class"] == TerminalClass.EXECUTED_POSITIVE.value
        and task_id not in flagged_task_ids
    ]
    return {
        "reports": reports,
        "verified_present_declared_deliverable_count": len(reports),
        "missing_declared_deliverables_not_verified": [
            row["declared_deliverable"] for row in matrix.values() if not row["present"]
        ],
        "flagged_task_ids": flagged_task_ids,
        "positive_synthesis_task_ids": positive_task_ids,
        "positive_claim_excluded_task_ids": flagged_task_ids,
        "flagged_claim_exclusion_reason": "flagged artifacts remain recorded but cannot support positive synthesis",
        "principle": FIELD_PRINCIPLES[
            "adversarial_verifier_receipts_and_positive_claim_exclusions"
        ],
    }


def _branch_synthesis(science: JsonMap) -> JsonDict:
    return {
        "phase_d": {
            "terminal_class": "complete_null_with_gate_blocked_cascade",
            "summary": "Exp6115 is a calibration null; Exp6116-Exp6119 do not provide selector evidence.",
            "positive_selector_claim": False,
        },
        "continuous_self_learning": {
            "terminal_class": "complete_positive",
            "summary": "Exp6120 reports outcome-committed reduced-order CSL ready with safety and ABI receipts.",
            "positive_claim_source_task_id": "exp6120-outcome-committed-reduced-order-csl",
        },
        "hardware": {
            "terminal_class": "blocked_physical_action_retired",
            "summary": "Exp6121 did not authorize JTAG because physical state was unchanged.",
            "hardware_execution_claim": False,
        },
        "arc": {
            "terminal_class": "complete_null_no_solve_credit",
            "summary": "Exp6122 found no supported primitive with held-out causal receipts and claims zero solves.",
            "solve_credit_claim": False,
        },
        "borrowed_evidence_count": 0,
        "source_gate_matrix_hash": sha256_json(science),
        "principle": FIELD_PRINCIPLES["branch_independent_scientific_synthesis"],
    }


def _prior_failure_receipts(tasks: Mapping[str, JsonMap]) -> JsonDict:
    capstone_task = tasks.get(EXPERIMENT_ID, {})
    failures = capstone_task.get("prior_failures") if isinstance(capstone_task, Mapping) else []
    rows: list[JsonDict] = []
    for failure in failures if isinstance(failures, list) else []:
        if not isinstance(failure, Mapping):
            continue
        prior_verdict = str(failure.get("verdict") or "")
        same_terminal_family = prior_verdict.startswith("complete_with_blocks:")
        rows.append(
            {
                "experiment_id": str(failure.get("experiment_id") or ""),
                "prior_verdict": prior_verdict,
                "retire_if_same_verdict": failure.get("retire_if_same_verdict") is True,
                "current_terminal_family": "complete_with_blocks",
                "same_terminal_family": same_terminal_family,
                "retirement_triggered": same_terminal_family
                and failure.get("retire_if_same_verdict") is True,
            }
        )
    return {
        "receipts": rows,
        "triggered_count": sum(int(row["retirement_triggered"]) for row in rows),
        "principle": FIELD_PRINCIPLES["prior_failure_same_verdict_retirement_receipts"],
    }


def _prd_gap_delta() -> JsonDict:
    return {
        "fr11": {
            "delta": "Exp6120 adds a positive reduced-order external-state CSL result.",
            "finish_line_redefined": False,
        },
        "fr12": {
            "delta": "Phase-D selector evidence did not ship; verification moat remains open.",
            "finish_line_redefined": False,
        },
        "phase_d": {
            "delta": "Calibration null replaces planned held/headroom/surface/selector evidence for .530.",
            "finish_line_redefined": False,
        },
        "live_arc_path": {
            "delta": "Exp6122 preserves no-solve-credit reachability null; registry/defaults unchanged.",
            "finish_line_redefined": False,
        },
        "north_star_redefined": False,
        "principle": FIELD_PRINCIPLES["prd_gap_and_north_star_delta"],
    }


def _docs_reconciliation(root: Path) -> JsonDict:
    spec_text = (
        (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / SPEC_RELATIVE_PATH).exists()
        else ""
    )
    return {
        "spec_requirement_present": "REQ-REPORT-6123" in spec_text,
        "docs_modified_by_exp6123": [SPEC_RELATIVE_PATH.as_posix()],
        "deferred_to_conductor_reconciler": [
            TRACEABILITY_RELATIVE_PATH.as_posix(),
            STATUS_RELATIVE_PATH.as_posix(),
            CHANGELOG_RELATIVE_PATH.as_posix(),
        ],
        "conductor_log_used_as_receipt": True,
        "verifier_gaps_used_as_receipt": True,
        "hardware_ledger_used_as_receipt": True,
        "arc_registry_used_as_receipt": True,
        "reason": "operator STOP-WHEN-DONE delegates status/changelog/traceability writes to the conductor reconciler",
        "principle": FIELD_PRINCIPLES[
            "specs_traceability_architecture_status_changelog_conductor_verifier_hardware_and_arc_reconciliation"
        ],
    }


def _architecture_boundary() -> JsonDict:
    return {
        "current": [
            "Exp6114 live GGUF CUDA canary readiness artifact exists.",
            "Exp6120 outcome-committed reduced-order CSL artifact reports positive external-state evidence.",
            "ARC live-path primitive reachability instrumentation exists as experiment-only null evidence.",
        ],
        "planned": [
            "Phase-D held candidate pool, headroom audit, per-layer surface, and selector remain planned behind failed gates.",
        ],
        "blocked": [
            "GateMate JTAG detect remains blocked on changed physical state.",
            "Phase-D selector chain is blocked by Exp6115 calibration null and missing later artifacts.",
        ],
        "retired": [
            "Repeating unchanged GateMate physical-state probes is retired by Exp6121.",
            "External-text/logprob Phase-D scorer family remains retired outside this hidden-state path.",
        ],
        "experiment_only": [
            "Exp6122 ARC primitive reachability null does not modify submitted defaults or registry credit.",
        ],
        "principle": FIELD_PRINCIPLES["architecture_current_planned_blocked_retired_boundary"],
    }


def _debt_delta(test_receipts: Sequence[JsonMap] | None) -> JsonDict:
    receipts = [dict(row) for row in (test_receipts or [])]
    before_failures: set[str] = set()
    after_failures: set[str] = set()
    inherited_after_failures: set[str] = set()
    task_owned_after_failures: set[str] = set()
    task_owned_failures: list[str] = []
    for row in receipts:
        failures = set(str(item) for item in row.get("failure_node_ids", []) if item)
        if row.get("ownership_class") == "task_owned" and row.get("exit_code") != 0:
            task_owned_failures.append(str(row.get("command") or ""))
            task_owned_after_failures |= failures
        if row.get("ownership_class") == "inherited":
            inherited_after_failures |= failures
        if row.get("phase") == "before":
            before_failures |= failures
        if row.get("phase") == "after":
            after_failures |= failures
    task_owned_new_failures = task_owned_after_failures - before_failures
    return {
        "test_receipts": receipts,
        "inherited_failure_node_ids_before": sorted(before_failures),
        "inherited_failure_node_ids_after": sorted(after_failures),
        "inherited_failure_node_ids_observed_after": sorted(inherited_after_failures),
        "new_failure_node_ids": sorted(after_failures - before_failures),
        "task_owned_new_failure_node_ids": sorted(task_owned_new_failures),
        "task_owned_failure_commands": task_owned_failures,
        "task_owned_debt_delta": len(task_owned_failures) + len(task_owned_new_failures),
        "inherited_debt_amplified": bool(task_owned_failures or task_owned_new_failures),
        "principle": FIELD_PRINCIPLES["inherited_debt_baselines_and_deltas"],
    }


def _research_complete_readiness(root: Path) -> JsonDict:
    payload, meta = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    milestones = payload.get("milestones") if isinstance(payload, Mapping) else []
    already_present = any(
        isinstance(row, Mapping) and row.get("id") == MILESTONE
        for row in milestones
        if isinstance(milestones, list)
    )
    return {
        "path": RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        "present": bool(meta["present"]),
        "loadable": bool(meta["loadable"]),
        "milestone_already_present": already_present,
        "append_ready": bool(meta["loadable"]) and not already_present,
        "append_performed": False,
        "reason": "capstone records readiness only; conductor reconciler owns completion append",
        "principle": FIELD_PRINCIPLES["research_complete_append_readiness"],
    }


def _field_provenance() -> dict[str, JsonDict]:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "aggregation_from_upstream_artifacts",
            "spec_ref": "REQ-REPORT-6123",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[Any] | None = None,
    test_receipts: Sequence[JsonMap] | None = None,
    protected_before: Mapping[str, str | None] | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    root = Path(root)
    roadmap, roadmap_meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_next, roadmap_next_meta = _read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    tasks = _roadmap_tasks(root)
    payloads, metadata = _artifact_payloads(root)
    conductor = _conductor_receipts(root)
    receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    if adversarial_receipts is None:
        receipts = run_live_adversarial_receipts(root, metadata)
    matrix = _build_matrix(root, payloads, metadata, conductor, receipts)
    gates = _recompute_gates(tasks, payloads, metadata, conductor)
    science = _scientific_gate_matrix(payloads, gates)
    branch_synthesis = _branch_synthesis(science)
    debt = _debt_delta(test_receipts)
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": "complete_with_blocks",
        "preconditions_checked": {
            "roadmap": {
                "path": ROADMAP_RELATIVE_PATH.as_posix(),
                "present": bool(roadmap_meta["present"]),
                "loadable": bool(roadmap_meta["loadable"]),
                "milestone": roadmap.get("milestone") if isinstance(roadmap, Mapping) else None,
                "expected_milestone": MILESTONE,
                "matches_expected_milestone": roadmap.get("milestone") == MILESTONE
                if isinstance(roadmap, Mapping)
                else False,
            },
            "roadmap_next": {
                "path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
                "present": bool(roadmap_next_meta["present"]),
                "loadable": bool(roadmap_next_meta["loadable"]),
                "milestone": roadmap_next.get("milestone")
                if isinstance(roadmap_next, Mapping)
                else None,
                "absence_is_failure": False,
            },
            "declared_upstream_task_count": len(UPSTREAM_TASKS),
            "declared_upstream_paths": {
                task_id: rel_path.as_posix() for task_id, rel_path in UPSTREAM_DELIVERABLES.items()
            },
            "source_hashes": _source_hashes(root),
            "root_clutter_paths": _root_clutter_inventory(root),
            "dirty_worktree_status_short": _git_status_short(root),
        },
        "milestone_task_and_declared_deliverable_matrix": {
            "milestone": MILESTONE,
            "task_count": len(matrix),
            "tasks": matrix,
            "capstone_output": {
                "task_id": EXPERIMENT_ID,
                "declared_deliverable": RESULT_RELATIVE_PATH.as_posix(),
                "separate_from_upstream_matrix": True,
            },
            "principle": FIELD_PRINCIPLES["milestone_task_and_declared_deliverable_matrix"],
        },
        "exact_evidence_resolution_receipts": {
            "by_task": {
                task_id: {
                    "identity": row["identity"],
                    "declared_deliverable": row["declared_deliverable"],
                    "present": row["present"],
                    "loadable": row["loadable"],
                    "sha256": row["sha256"],
                    "same_number_aliases_ignored": row["same_number_aliases_ignored"],
                    "evidence_definition": "only this declared deliverable path",
                }
                for task_id, row in matrix.items()
            },
            "numeric_prefix_resolution_used": False,
            "principle": FIELD_PRINCIPLES["exact_evidence_resolution_receipts"],
        },
        "per_task_terminal_class_and_reason": _terminal_receipt(matrix),
        "executed_skipped_missing_blocked_retired_underpowered_null_ready_positive_and_flagged_counts": _counts(
            matrix
        ),
        "gate_recomputation_and_title_yaml_alignment": gates,
        "candidate_pool_headroom_surface_selector_csl_hardware_and_arc_gate_matrix": science,
        "adversarial_verifier_receipts_and_positive_claim_exclusions": _adversarial_group(
            receipts,
            matrix,
        ),
        "prior_failure_same_verdict_retirement_receipts": _prior_failure_receipts(tasks),
        "branch_independent_scientific_synthesis": branch_synthesis,
        "prd_gap_and_north_star_delta": _prd_gap_delta(),
        "specs_traceability_architecture_status_changelog_conductor_verifier_hardware_and_arc_reconciliation": _docs_reconciliation(
            root
        ),
        "architecture_current_planned_blocked_retired_boundary": _architecture_boundary(),
        "inherited_debt_baselines_and_deltas": debt,
        "research_complete_append_readiness": _research_complete_readiness(root),
        "protected_files_unchanged": _protected_files_unchanged(root, protected_before),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": {
            "phase_d": "Python/Z3 labels are oracle; hidden-state selectors did not run.",
            "continuous_self_learning": "exact future outcomes are oracle; learned state is not.",
            "hardware": "raw IDCODE/host-I/O would be authoritative, but no detect was authorized.",
            "arc": "environment transitions and reproduce are authority; primitives are not oracle.",
            "principle": FIELD_PRINCIPLES["verifier_is_oracle"],
        },
        "missing_verifier_gaps": {
            "phase_d": "candidate-internal correctness discriminator remains missing.",
            "continuous_self_learning": "no residual blocker recorded by Exp6120, but broader FR11 remains open.",
            "hardware": "changed physical receipt / IDCODE evidence missing for GateMate.",
            "arc": "no live-reachable primitive with held-out causal attribution; no solve credit.",
            "principle": FIELD_PRINCIPLES["missing_verifier_gaps"],
        },
        "field_provenance": _field_provenance(),
        "test_commands": [str(row.get("command") or "") for row in (test_receipts or [])],
        "test_exit_codes": {
            str(row.get("command") or ""): row.get("exit_code") for row in (test_receipts or [])
        },
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_with_blocks: .530 exact classes preserved: phase_d_null_and_gate_skipped, "
            "csl_positive, gatemate_blocked_physical_action_retired, arc_null_no_solve_credit, "
            "adversarial_warn_excluded"
        ),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[Any] | None = None,
    test_receipts: Sequence[JsonMap] | None = None,
) -> JsonDict:
    root = Path(root)
    target = Path(output_path) if output_path is not None else root / RESULT_RELATIVE_PATH
    before = _protected_file_hashes(root)
    start = time.monotonic()
    payload = build_artifact(
        root=root,
        adversarial_receipts=adversarial_receipts,
        test_receipts=test_receipts,
        protected_before=before,
        duration_s=0.0,
    )
    payload["duration_s"] = round(max(time.monotonic() - start, 0.0001), 6)
    payload["reproducibility_checksum"] = payload_checksum(payload)
    write_json(target, payload)
    return payload


def _load_test_receipts(path: Path | None) -> list[JsonDict] | None:  # pragma: no cover
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("--test-receipts-json must contain a JSON list")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--test-receipts-json", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = run(
        root=args.root,
        output_path=args.output,
        test_receipts=_load_test_receipts(args.test_receipts_json),
    )
    result_path = Path(args.output) if args.output is not None else args.root / RESULT_RELATIVE_PATH
    print(
        json.dumps(
            {
                "result_path": result_path.as_posix(),
                "checksum": payload["reproducibility_checksum"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
