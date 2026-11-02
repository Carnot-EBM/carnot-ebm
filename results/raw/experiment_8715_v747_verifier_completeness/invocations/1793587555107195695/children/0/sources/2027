"""Exp5796 transition receipt from terminal milestone .516 into .517.

Spec refs: REQ-REPORT-5796, SCENARIO-REPORT-5796,
SCENARIO-REPORT-5796-COLLISION-BLOCK,
SCENARIO-REPORT-5796-IDENTITY-BLOCK,
SCENARIO-REPORT-5796-FIELD-PRINCIPLES.

This module is deliberately a ledger reconciler. It reads local artifacts,
completion history, and conductor rows, then records the exact evidence boundary
for the next milestone. It does not retry science, repair gate-blocked work, or
replace missing declared paths with convenient same-number files.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import yaml

from carnot.experiment_5754_v513_capstone_reconciliation import (
    _read_json_any,
    path_sha256,
    payload_checksum,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5796_transition_v517.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXPERIMENT = "experiment_5796_transition_v517"
EXPERIMENT_ID = "exp5796-transition-v517"
MILESTONE_FROM = "2026.07.516"
MILESTONE_TO = "2026.07.517"
NEXT_TASK_RANGE = "exp5796-exp5808"
RUN_DATE = "2026-07-22"
RANDOM_SEED = 5796
SCHEMA = "carnot.experiment_5796.transition_v517.v1"
INFERENCE_SUBSTRATE = "local_exact_artifact_and_conductor_reconciliation_no_llm"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-5796",
    "SCENARIO-REPORT-5796",
    "SCENARIO-REPORT-5796-COLLISION-BLOCK",
    "SCENARIO-REPORT-5796-IDENTITY-BLOCK",
    "SCENARIO-REPORT-5796-FIELD-PRINCIPLES",
)

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5782-transition-v516": Path("results/experiment_5782_transition_v516.json"),
    "exp5783-v516-source-delta-ingestion": Path(
        "results/experiment_5783_v516_source_delta_ingestion.json"
    ),
    "exp5784-evidence-index-terminal-qualification": Path(
        "results/experiment_5784_evidence_index_terminal_qualification.json"
    ),
    "exp5785-hardness-surface-prospective-fixture": Path(
        "results/experiment_5785_hardness_surface_fixture.json"
    ),
    "exp5786-sota-hardness-controlled-constraint-stream": Path(
        "results/experiment_5786_sota_constraint_stream.json"
    ),
    "exp5787-validation-gated-constraint-skill-ab": Path(
        "results/experiment_5787_validation_gated_constraint_skill_ab.json"
    ),
    "exp5788-constraint-skill-transfer-audit": Path(
        "results/experiment_5788_constraint_skill_transfer_audit.json"
    ),
    "exp5789-constraint-skill-shadow-adapter": Path(
        "results/experiment_5789_constraint_skill_shadow_adapter.json"
    ),
    "exp5790-arc-world-model-admission-contract": Path(
        "results/experiment_5790_arc_world_model_admission_contract.json"
    ),
    "exp5791-arc-sota-independent-hypothesis-panel": Path(
        "results/experiment_5791_arc_sota_independent_hypothesis_panel.json"
    ),
    "exp5792-arc-calibration-only-selector": Path(
        "results/experiment_5792_arc_calibration_only_selector.json"
    ),
    "exp5793-arc-live-world-model-ab": Path("results/experiment_5793_arc_live_world_model_ab.json"),
    "exp5794-hardware-terminal-action-receipt": Path(
        "results/experiment_5794_hardware_terminal_action_receipt.json"
    ),
    "exp5795-v516-capstone-reconciliation": Path(
        "results/experiment_5795_v516_capstone_reconciliation.json"
    ),
}
EXPECTED_TASK_IDS = tuple(TASK_ARTIFACT_PATHS)

TASK_TITLES: dict[str, str] = {
    "exp5782-transition-v516": "Transition terminal .515 evidence and allocate collision-free .516 identities",
    "exp5783-v516-source-delta-ingestion": "Time-windowed literature freshness receipt",
    "exp5784-evidence-index-terminal-qualification": "Qualify the existing exact-deliverable index with terminal test receipts and gate replay",
    "exp5785-hardness-surface-prospective-fixture": "Gated on Exp5784 readiness: build a sealed hardness- and surface-controlled exact fixture",
    "exp5786-sota-hardness-controlled-constraint-stream": "Gated on Exp5785 fixture readiness: run the three-family prospective exact constraint stream",
    "exp5787-validation-gated-constraint-skill-ab": "Gated on Exp5786 clean drift headroom: run continuous typed-constraint skill learning A/B",
    "exp5788-constraint-skill-transfer-audit": "Causal future-family holdout of versioned rule state",
    "exp5789-constraint-skill-shadow-adapter": "Gated on Exp5788 transfer: wire a disabled typed-constraint shadow adapter with exact restart and rollback",
    "exp5790-arc-world-model-admission-contract": "Pivotal-dynamics accreditation contract for immutable simulators",
    "exp5791-arc-sota-independent-hypothesis-panel": "Gated on Exp5790 admission readiness: run a matched three-family independent single-shot ARC hypothesis panel",
    "exp5792-arc-calibration-only-selector": "Frozen calibration chooser over immutable simulator candidates",
    "exp5793-arc-live-world-model-ab": "Gated on Exp5792 selector benefit: measure selected-world-model influence on held-out live E3",
    "exp5794-hardware-terminal-action-receipt": "Board-state hash ledger and operator handoff packet",
    "exp5795-v516-capstone-reconciliation": "Reconcile .516 evidence, phase telemetry, specs, ops, and next controls",
}

CONDUCTOR_TITLE_PATTERNS: dict[str, str] = {
    "exp5782-transition-v516": "Transition terminal .515 evidence and allocate col",
    "exp5783-v516-source-delta-ingestion": "Time-windowed literature freshness receipt",
    "exp5784-evidence-index-terminal-qualification": "Qualify the existing exact-deliverable index with",
    "exp5785-hardness-surface-prospective-fixture": "Gated on Exp5784 readiness: build a sealed hardnes",
    "exp5786-sota-hardness-controlled-constraint-stream": "Gated on Exp5785 fixture readiness: run the three-",
    "exp5787-validation-gated-constraint-skill-ab": "Gated on Exp5786 clean drift headroom: run continu",
    "exp5788-constraint-skill-transfer-audit": "Causal future-family holdout of versioned rule sta",
    "exp5789-constraint-skill-shadow-adapter": "Gated on Exp5788 transfer: wire a disabled typed-c",
    "exp5790-arc-world-model-admission-contract": "Pivotal-dynamics accreditation contract for immuta",
    "exp5791-arc-sota-independent-hypothesis-panel": "Gated on Exp5790 admission readiness: run a matche",
    "exp5792-arc-calibration-only-selector": "Frozen calibration chooser over immutable simulato",
    "exp5793-arc-live-world-model-ab": "Gated on Exp5792 selector benefit: measure selecte",
    "exp5794-hardware-terminal-action-receipt": "Board-state hash ledger and operator handoff packe",
    "exp5795-v516-capstone-reconciliation": "Reconcile .516 evidence, phase telemetry, specs, o",
}

POSITIVE_RESULT_TASK_IDS = (
    "exp5782-transition-v516",
    "exp5784-evidence-index-terminal-qualification",
    "exp5785-hardness-surface-prospective-fixture",
    "exp5790-arc-world-model-admission-contract",
    "exp5794-hardware-terminal-action-receipt",
)
SCIENTIFIC_NULL_TASK_IDS = ("exp5783-v516-source-delta-ingestion",)
NEGATIVE_RESULT_TASK_IDS = ("exp5786-sota-hardness-controlled-constraint-stream",)
BLOCKED_PRECONDITION_TASK_IDS: tuple[str, ...] = ()
BLOCKED_GATE_TASK_IDS = (
    "exp5787-validation-gated-constraint-skill-ab",
    "exp5789-constraint-skill-shadow-adapter",
    "exp5793-arc-live-world-model-ab",
)
FAILED_DELIVERY_TASK_IDS = ("exp5791-arc-sota-independent-hypothesis-panel",)
MISSING_TASK_IDS = (
    "exp5788-constraint-skill-transfer-audit",
    "exp5792-arc-calibration-only-selector",
)
NO_SOLVE_TASK_IDS = ("exp5790-arc-world-model-admission-contract",)
OPERATIONAL_TASK_IDS = ("exp5795-v516-capstone-reconciliation",)

ACTIVE_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5796-transition-v517": RESULT_RELATIVE_PATH,
    "exp5797-v517-source-delta-ingestion": Path(
        "results/experiment_5797_v517_source_delta_ingestion.json"
    ),
    "exp5798-sota-answer-channel-diagnostic": Path(
        "results/experiment_5798_sota_answer_channel_diagnostic.json"
    ),
    "exp5799-sota-answer-channel-canary": Path(
        "results/experiment_5799_sota_answer_channel_canary.json"
    ),
}
ACTIVE_TASK_IDS = tuple(ACTIVE_TASK_ARTIFACT_PATHS)

NEXT_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    **ACTIVE_TASK_ARTIFACT_PATHS,
    "exp5800-channel-qualified-constraint-stream": Path(
        "results/experiment_5800_channel_qualified_constraint_stream.json"
    ),
    "exp5801-future-validated-constraint-skill-ab": Path(
        "results/experiment_5801_future_validated_constraint_skill_ab.json"
    ),
    "exp5802-constraint-skill-endurance": Path(
        "results/experiment_5802_constraint_skill_endurance.json"
    ),
    "exp5803-constraint-skill-ood-audit": Path(
        "results/experiment_5803_constraint_skill_ood_audit.json"
    ),
    "exp5804-arc-bootstrap-safe-sota-panel": Path(
        "results/experiment_5804_arc_bootstrap_safe_sota_panel.json"
    ),
    "exp5805-arc-immutable-selector": Path("results/experiment_5805_arc_immutable_selector.json"),
    "exp5806-arc-live-heldout-world-model-ab": Path(
        "results/experiment_5806_arc_live_heldout_world_model_ab.json"
    ),
    "exp5807-self-learning-microkernel-handoff": Path(
        "results/experiment_5807_self_learning_microkernel_handoff.json"
    ),
    "exp5808-v517-capstone-reconciliation": Path(
        "results/experiment_5808_v517_capstone_reconciliation.json"
    ),
}
NEXT_TASK_IDS = tuple(NEXT_TASK_ARTIFACT_PATHS)

PROTECTED_FILE_PATHS = (ROADMAP_RELATIVE_PATH, CONDUCTOR_RELATIVE_PATH)
PLAN_REFERENCE_PATHS = (ROADMAP_RELATIVE_PATH, VNEXT_RELATIVE_PATH)
COLLISION_TEXT_PATHS = (
    RESEARCH_COMPLETE_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/python -c \"import pathlib, yaml; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); p=pathlib.Path('research-roadmap-next.yaml'); yaml.safe_load(p.read_text()) if p.exists() else None; yaml.safe_load(pathlib.Path('research-complete.yaml').read_text())\"",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5796_transition_v517.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5796_transition_v517.py -m pytest tests/python/test_experiment_5796_transition_v517.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5796_transition_v517.py --fail-under=100",
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/check_spec_coverage.py", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/root_clutter_sweep.py", "exit_code": None, "status": "not_run"},
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Identifies the versioned Exp5796 transition artifact schema.",
    "experiment": "Names the local experiment slug without relying on paths.",
    "experiment_id": "Binds this receipt to the conductor task id.",
    "status": "Bare terminal transition state derived from explicit precondition checks.",
    "run_date": "Records the operator-specified transition date as a fixed value.",
    "random_seed": "Deterministic metadata for checksum stability; no stochastic run occurs.",
    "spec_refs": "Anchors the artifact to REQ-REPORT-5796 and its scenarios.",
    "result_path": "Names the emitted deliverable path.",
    "field_principles": "Maps every top-level artifact field to its evidence boundary.",
    "preconditions_checked": "Records exact inputs and resource checks before mutation or claims.",
    "milestone_from": "Names the terminal milestone whose conductor evidence is archived.",
    "milestone_to": "Names the milestone receiving the archived evidence.",
    "canonical_identity_contract": "Defines canonical evidence as milestone, task id, and declared deliverable.",
    "declared_deliverable_matrix": "Lists every V516 task with its exact declared deliverable.",
    "canonical_artifact_hashes": "Hashes existing declared artifacts and records missing declared paths.",
    "same_number_alias_groups": "Shows numeric-prefix candidates as aliases that never become canonical.",
    "artifact_selection_policy": "Must equal exact_declared_deliverable to forbid glob or mtime selection.",
    "conductor_outcomes": "Preserves conductor execution authority, including delivery failures and gate skips.",
    "positive_result_task_ids": "Records V516 tasks with positive terminal evidence.",
    "scientific_null_task_ids": "Records complete no-delta evidence without treating it as a failure.",
    "negative_result_task_ids": "Records complete negative stream-readiness evidence without hiding execution.",
    "blocked_precondition_task_ids": "Reserved for primary blocked-precondition classes; Exp5791 is primary failed-delivery here.",
    "blocked_gate_task_ids": "Records gate-blocked V516 tasks without calling them nulls.",
    "failed_delivery_task_ids": "Records bootstrap-ordered output delivery failures from conductor evidence.",
    "missing_task_ids": "Records declared V516 deliverables that are intentionally absent.",
    "retired_task_ids": "Carries retired prior scopes from the V516 capstone without making them V516 results.",
    "no_solve_task_ids": "Records ARC work with no solve credit.",
    "operational_task_ids": "Records completed transition or capstone operations outside science outcome classes.",
    "archived_task_ids": "Lists exactly the fourteen V516 task ids carried forward.",
    "research_complete_append_count": "Records the exactly-once archive receipt; zero when V516 is already present.",
    "duplicate_history_diagnostics": "Reports duplicate history without deleting, sorting, or rewriting it.",
    "collision_scan": "Shows the Exp5796-Exp5808 namespace scan and allowed plan references.",
    "next_task_range": "Records the destination task interval as exp5796-exp5808.",
    "next_range_collision_count": "Bare scalar namespace safety gate for downstream allocation.",
    "docs_reconciled": "Records transition-owned reconciliation mode without touching ops docs.",
    "research_roadmap_unchanged": "Bare boolean must remain true because active roadmap mutation is forbidden.",
    "conductor_unchanged": "Bare boolean must remain true by operator instruction.",
    "inference_substrate": "This transition uses exact local artifacts and conductor rows only; no LLM is involved.",
    "test_commands": "Verification commands are preserved exactly.",
    "test_exit_codes": "Observed verification exits are recorded without relabeling failures.",
    "reproducibility_checksum": "Stable checksum detects artifact drift.",
    "honest_verdict": "Terminal summary starts with complete: or blocked: and does not promote gate blocks.",
}


def _read_yaml_with_meta(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "parsed": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        return {}, meta
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        meta["error"] = str(exc)
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = f"expected mapping, got {type(payload).__name__}"
        return {}, meta
    meta["parsed"] = True
    return payload, meta


def _roadmap_summary(path: Path) -> JsonDict:
    payload, meta = _read_yaml_with_meta(path)
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    task_ids = [
        str(row["id"])
        for row in tasks
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    ]
    deliverables = [
        str(row["deliverable"])
        for row in tasks
        if isinstance(row, Mapping) and isinstance(row.get("deliverable"), str)
    ]
    return {
        **meta,
        "milestone": payload.get("milestone")
        if isinstance(payload.get("milestone"), str)
        else None,
        "task_ids": task_ids,
        "deliverables": deliverables,
    }


def _research_complete_payload(root: Path) -> JsonDict:
    payload, _meta = _read_yaml_with_meta(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    return payload


def _research_complete_blocks(root: Path, milestone: str = MILESTONE_FROM) -> list[JsonDict]:
    milestones = _research_complete_payload(root).get("milestones")
    if not isinstance(milestones, list):
        return []
    return [
        block for block in milestones if isinstance(block, dict) and block.get("id") == milestone
    ]


def _task_signature(block: JsonMap) -> tuple[tuple[str, str], ...]:
    tasks = block.get("tasks")
    if not isinstance(tasks, list):
        return ()
    rows: list[tuple[str, str]] = []
    for row in tasks:
        if isinstance(row, Mapping) and isinstance(row.get("id"), str):
            deliverable = row.get("deliverable")
            rows.append((str(row["id"]), str(deliverable) if isinstance(deliverable, str) else ""))
    return tuple(rows)


def _duplicate_task_conflicts(tasks: Sequence[Any]) -> list[str]:
    deliverables_by_task: dict[str, set[str]] = defaultdict(set)
    for row in tasks:
        if isinstance(row, Mapping) and isinstance(row.get("id"), str):
            deliverable = row.get("deliverable")
            deliverables_by_task[str(row["id"])].add(
                str(deliverable) if isinstance(deliverable, str) else ""
            )
    return sorted(task_id for task_id, values in deliverables_by_task.items() if len(values) > 1)


def _synthetic_task_rows() -> list[JsonDict]:
    return [
        {
            "id": task_id,
            "title": TASK_TITLES[task_id],
            "deliverable": rel_path.as_posix(),
            "result": "append-if-absent exact transition receipt",
        }
        for task_id, rel_path in TASK_ARTIFACT_PATHS.items()
    ]


def _declared_deliverable_matrix(root: Path) -> tuple[list[JsonDict], JsonDict, list[str]]:
    blocks = _research_complete_blocks(root)
    signatures = [_task_signature(block) for block in blocks]
    unique_signatures = set(signatures)
    append_count = 0 if blocks else 1
    stats: JsonDict = {
        "research_complete_milestone_from_block_count": len(blocks),
        "unique_declared_deliverable_block_count": len(unique_signatures),
        "declared_deliverables_unambiguous": len(unique_signatures) <= 1,
        "completion_block_source": "research_complete" if blocks else "synthetic_append_if_absent",
    }
    failures: list[str] = []
    if len(unique_signatures) > 1:
        failures.append("ambiguous_research_complete_declared_task_blocks")

    selected_tasks = blocks[0].get("tasks") if blocks else _synthetic_task_rows()
    task_rows = selected_tasks if isinstance(selected_tasks, list) else []
    conflicts = _duplicate_task_conflicts(task_rows)
    if conflicts:
        failures.append(f"duplicate_task_id_conflicts={conflicts}")
    by_task: dict[str, JsonMap] = {
        str(row["id"]): row
        for row in task_rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    declared_ids = (
        tuple(task_id for task_id, _deliverable in next(iter(unique_signatures), ()))
        if blocks
        else EXPECTED_TASK_IDS
    )
    if declared_ids != EXPECTED_TASK_IDS:
        failures.append(f"declared_task_ids_mismatch={list(declared_ids)}")

    matrix: list[JsonDict] = []
    mismatches: list[str] = []
    for task_id in EXPECTED_TASK_IDS:
        row = by_task.get(task_id, {})
        declared = row.get("deliverable")
        expected = TASK_ARTIFACT_PATHS[task_id].as_posix()
        declared_path = declared if isinstance(declared, str) else ""
        if declared_path != expected:
            mismatches.append(f"{task_id}:{declared_path or '<missing>'}!={expected}")
        matrix.append(
            {
                "identity": [MILESTONE_FROM, task_id, declared_path or expected],
                "milestone": MILESTONE_FROM,
                "task_id": task_id,
                "title": row.get("title") if isinstance(row.get("title"), str) else TASK_TITLES[task_id],
                "declared_deliverable": declared_path or expected,
                "research_complete_result": row.get("result")
                if isinstance(row.get("result"), str)
                else "",
                "selection_policy": ARTIFACT_SELECTION_POLICY,
            }
        )
    if mismatches:
        failures.append(f"declared_deliverable_mismatch={mismatches}")
    stats["research_complete_append_count"] = append_count
    return matrix, stats, failures


def _payload_status(payload: JsonMap, metadata: JsonMap, task_id: str) -> str:
    if metadata.get("exists") is False:
        return "missing"
    if metadata.get("loadable") is False:
        return "malformed"
    if task_id == "exp5791-arc-sota-independent-hypothesis-panel":
        return "blocked-precondition"
    if payload.get("schema") == "blocked_gate_check_v1" or payload.get("blocked_at_layer"):
        return "blocked-gate"
    status = payload.get("status")
    verdict = payload.get("honest_verdict")
    if status == "blocked" or (isinstance(verdict, str) and verdict.startswith("blocked:")):
        return "blocked"
    if status == "complete" or (isinstance(verdict, str) and verdict.startswith("complete:")):
        return "complete"
    return str(status) if isinstance(status, str) and status else "unknown"


def _parse_conductor_log(path_or_root: Path) -> list[JsonDict]:
    path = path_or_root / CONDUCTOR_LOG_RELATIVE_PATH if path_or_root.is_dir() else path_or_root
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "timestamp": parts[0],
                "title": parts[1],
                "outcome": parts[2],
                "detail": parts[3],
                "line": line,
            }
        )
    return rows


def _conductor_outcomes(root: Path) -> tuple[dict[str, JsonDict], list[str]]:
    rows = _parse_conductor_log(root)
    outcomes: dict[str, JsonDict] = {}
    missing: list[str] = []
    for task_id, pattern in CONDUCTOR_TITLE_PATTERNS.items():
        matches = [row for row in rows if pattern in str(row["title"])]
        if not matches:
            missing.append(task_id)
        latest = matches[-1] if matches else {}
        delivery_failures = [
            row for row in matches if "artifact_not_updated_past_bootstrap" in str(row["detail"])
        ]
        gate_blocks = [row for row in matches if row.get("outcome") == "GATE_BLOCK"]
        outcomes[task_id] = {
            "outcome": latest.get("outcome", "UNKNOWN"),
            "source": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            "latest_evidence_line": latest.get("line", ""),
            "evidence_lines": [str(row["line"]) for row in matches],
            "attempt_count": len(matches),
            "delivery_failure_count": len(delivery_failures),
            "delivery_failure_reason": "artifact_not_updated_past_bootstrap"
            if delivery_failures
            else None,
            "gate_block_count": len(gate_blocks),
            "gate_block_reason": latest.get("detail") if gate_blocks else None,
        }
    return outcomes, missing


def _canonical_artifact_hashes(
    root: Path,
    matrix: Sequence[JsonMap],
    conductor_outcomes: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for item in matrix:
        task_id = str(item["task_id"])
        rel_path = Path(str(item["declared_deliverable"]))
        payload, metadata = _read_json_any(root / rel_path)
        status = _payload_status(payload, metadata, task_id)
        non_artifact_authorized = (
            status == "missing"
            and conductor_outcomes.get(task_id, {}).get("outcome") == "GATE_BLOCK"
        )
        rows[task_id] = {
            "identity": [MILESTONE_FROM, task_id, rel_path.as_posix()],
            "path": rel_path.as_posix(),
            "present": bool(metadata.get("exists")),
            "loadable": bool(metadata.get("loadable")),
            "sha256": metadata.get("sha256"),
            "status": status,
            "honest_verdict": payload.get("honest_verdict")
            if isinstance(payload.get("honest_verdict"), str)
            else "",
            "selected_by": ARTIFACT_SELECTION_POLICY,
            "non_artifact_outcome_authorized": non_artifact_authorized,
            "error": metadata.get("error"),
        }
    return rows


def _same_number_alias_groups(
    root: Path,
    canonical_hashes: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    groups: dict[str, JsonDict] = {}
    for task_id in EXPECTED_TASK_IDS:
        number_match = re.match(r"exp(\d+)", task_id)
        if not number_match or task_id not in canonical_hashes:
            continue
        number = number_match.group(1)
        canonical = canonical_hashes[task_id]
        canonical_path = Path(str(canonical["path"]))
        candidates = sorted((root / "results").glob(f"experiment_{number}*.json"))
        aliases: list[JsonDict] = []
        for candidate in candidates:
            rel_path = candidate.relative_to(root)
            if rel_path == canonical_path:
                continue
            payload, metadata = _read_json_any(candidate)
            aliases.append(
                {
                    "path": rel_path.as_posix(),
                    "present": bool(metadata.get("exists")),
                    "loadable": bool(metadata.get("loadable")),
                    "sha256": metadata.get("sha256"),
                    "status": _payload_status(payload, metadata, task_id),
                    "honest_verdict": payload.get("honest_verdict")
                    if isinstance(payload.get("honest_verdict"), str)
                    else "",
                    "role": "same_number_alias",
                }
            )
        groups[number] = {
            "experiment_number": number,
            "canonical": {
                "task_id": task_id,
                "path": canonical["path"],
                "present": canonical["present"],
                "sha256": canonical["sha256"],
                "status": canonical["status"],
                "role": "canonical_declared_deliverable",
            },
            "aliases": aliases,
            "selection_policy": ARTIFACT_SELECTION_POLICY,
        }
    return groups


def _next_range_tokens() -> tuple[str, ...]:
    tokens: list[str] = []
    for task_id, rel_path in NEXT_TASK_ARTIFACT_PATHS.items():
        number = re.match(r"exp(\d+)", task_id)
        if number:
            tokens.append(number.group(0))
            tokens.append(f"experiment_{number.group(1)}")
        tokens.append(task_id)
        tokens.append(rel_path.as_posix())
    return tuple(dict.fromkeys(tokens))


def _text_has_next_range(text: str) -> bool:
    return any(token in text for token in _next_range_tokens())


def _collision_scan(root: Path) -> JsonDict:
    collisions: list[JsonDict] = []
    allowed_plan_references: list[JsonDict] = []
    for rel_path in PLAN_REFERENCE_PATHS:
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if _text_has_next_range(text):
            allowed_plan_references.append(
                {"path": rel_path.as_posix(), "kind": "planned_v517_reference"}
            )
    for rel_path in COLLISION_TEXT_PATHS:
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if _text_has_next_range(text):
            collisions.append(
                {"path": rel_path.as_posix(), "kind": "preexisting_content_reference"}
            )
    results_dir = root / "results"
    for number in range(5796, 5809):
        for candidate in sorted(results_dir.glob(f"experiment_{number}*.json")):
            rel_path = candidate.relative_to(root)
            if rel_path == RESULT_RELATIVE_PATH:
                continue
            collisions.append({"path": rel_path.as_posix(), "kind": "preexisting_result_file"})
    collisions = sorted(collisions, key=lambda row: (str(row["path"]), str(row["kind"])))
    return {
        "next_task_ids": list(NEXT_TASK_IDS),
        "next_declared_deliverables": [
            NEXT_TASK_ARTIFACT_PATHS[task_id].as_posix() for task_id in NEXT_TASK_IDS
        ],
        "allowed_plan_references": allowed_plan_references,
        "preexisting_collisions": collisions,
        "preexisting_collision_count": len(collisions),
        "collision_free": not collisions,
    }


def _resource_receipts(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    mem_available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1])
                break
    return {
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "mem_available_kib": mem_available,
    }


def _duplicate_history_diagnostics(root: Path) -> JsonDict:
    milestones = _research_complete_payload(root).get("milestones")
    blocks = milestones if isinstance(milestones, list) else []
    by_milestone: dict[str, list[JsonMap]] = defaultdict(list)
    for block in blocks:
        if isinstance(block, Mapping) and isinstance(block.get("id"), str):
            by_milestone[str(block["id"])].append(block)
    duplicates: list[JsonDict] = []
    for milestone, milestone_blocks in sorted(by_milestone.items()):
        if len(milestone_blocks) <= 1:
            continue
        duplicates.append(
            {
                "milestone": milestone,
                "block_count": len(milestone_blocks),
                "unique_block_signature_count": len(
                    {_task_signature(block) for block in milestone_blocks}
                ),
                "mutation": "preserved_read_only",
            }
        )
    from_blocks = by_milestone.get(MILESTONE_FROM, [])
    return {
        "history_mutation_policy": "read_only_no_dedup_sort_or_rewrite",
        "milestone_from_block_count": len(from_blocks),
        "milestone_from_unique_signature_count": len(
            {_task_signature(block) for block in from_blocks}
        ),
        "duplicate_milestone_blocks": duplicates,
        "duplicate_history_block_count": sum(row["block_count"] - 1 for row in duplicates),
    }


def _retired_task_ids(root: Path) -> list[str]:
    payload, metadata = _read_json_any(root / TASK_ARTIFACT_PATHS["exp5795-v516-capstone-reconciliation"])
    if not metadata.get("loadable"):
        return []
    values = payload.get("retired_task_ids")
    return [str(item) for item in values] if isinstance(values, list) else []


def _git_modified(root: Path, rel_path: Path) -> bool:  # pragma: no cover - live repo check
    result = subprocess.run(
        ["git", "status", "--short", "--", rel_path.as_posix()],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return bool(result.stdout.strip())


def _protected_files(
    root: Path,
    modification_overrides: Mapping[Path, bool] | None,
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for rel_path in PROTECTED_FILE_PATHS:
        if modification_overrides is not None and rel_path in modification_overrides:
            modified = bool(modification_overrides[rel_path])
            source = "test_override"
        else:  # pragma: no cover - live artifact generation uses git status
            modified = _git_modified(root, rel_path)
            source = "git_status"
        rows[rel_path.as_posix()] = {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
            "modified_by_exp5796": modified,
            "check_source": source,
        }
    return rows


def _input_hashes(root: Path, canonical_hashes: Mapping[str, JsonMap]) -> JsonDict:
    paths = (
        RESEARCH_COMPLETE_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        VNEXT_RELATIVE_PATH,
    )
    return {
        "source_files": {
            rel_path.as_posix(): {
                "present": (root / rel_path).exists(),
                "sha256": path_sha256(root / rel_path),
            }
            for rel_path in paths
        },
        "declared_deliverables": {
            task_id: {"path": row["path"], "sha256": row["sha256"], "present": row["present"]}
            for task_id, row in canonical_hashes.items()
        },
    }


def _test_exit_codes(tests_run: Sequence[JsonMap]) -> JsonDict:
    return {str(row.get("command")): row.get("exit_code") for row in tests_run}


def _load_tests_run(path: Path | None) -> list[JsonDict]:
    if path is None:
        return [dict(row) for row in DEFAULT_TESTS_RUN]
    payload = json.loads(path.read_text(encoding="utf-8"))  # pragma: no cover - CLI convenience
    if not isinstance(payload, list):  # pragma: no cover - CLI convenience
        raise ValueError("tests-run JSON must be a list")
    return [dict(row) for row in payload]  # pragma: no cover - CLI convenience


def _outcome_class(task_id: str) -> str:
    if task_id in POSITIVE_RESULT_TASK_IDS:
        return "complete-positive"
    if task_id in SCIENTIFIC_NULL_TASK_IDS:
        return "complete-null"
    if task_id in NEGATIVE_RESULT_TASK_IDS:
        return "complete-negative"
    if task_id in BLOCKED_GATE_TASK_IDS:
        return "blocked-gate"
    if task_id in FAILED_DELIVERY_TASK_IDS:
        return "failed-delivery"
    if task_id in MISSING_TASK_IDS:
        return "missing"
    if task_id in OPERATIONAL_TASK_IDS:
        return "complete-operational"
    return "unclassified"


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
) -> JsonDict:
    roadmap_active = _roadmap_summary(root / ROADMAP_RELATIVE_PATH)
    roadmap_next = _roadmap_summary(root / ROADMAP_NEXT_RELATIVE_PATH)
    declared_matrix, complete_stats, matrix_failures = _declared_deliverable_matrix(root)
    conductor_rows, missing_conductor_task_ids = _conductor_outcomes(root)
    canonical_hashes = _canonical_artifact_hashes(root, declared_matrix, conductor_rows)
    alias_groups = _same_number_alias_groups(root, canonical_hashes)
    collision_scan = _collision_scan(root)
    duplicate_history = _duplicate_history_diagnostics(root)
    protected_files = _protected_files(root, modification_overrides)

    for task_id, artifact in canonical_hashes.items():
        conductor_rows[task_id]["artifact_status"] = artifact["status"]
        conductor_rows[task_id]["artifact_path"] = artifact["path"]
        conductor_rows[task_id]["terminal_artifact_honest_verdict"] = artifact["honest_verdict"]
        conductor_rows[task_id]["terminal_artifact_sha256"] = artifact["sha256"]
        conductor_rows[task_id]["outcome_class"] = _outcome_class(task_id)

    research_roadmap_unchanged = not protected_files[ROADMAP_RELATIVE_PATH.as_posix()][
        "modified_by_exp5796"
    ]
    conductor_unchanged = not protected_files[CONDUCTOR_RELATIVE_PATH.as_posix()][
        "modified_by_exp5796"
    ]
    active_task_ids = roadmap_active["task_ids"]
    missing_or_malformed = [
        task_id
        for task_id, row in canonical_hashes.items()
        if row["status"] in {"missing", "malformed"} and task_id not in MISSING_TASK_IDS
    ]

    failed_preconditions = list(matrix_failures)
    if not roadmap_active["parsed"]:
        failed_preconditions.append("active_roadmap_unparseable")
    if roadmap_active["milestone"] != MILESTONE_TO:
        failed_preconditions.append(f"active_roadmap_milestone={roadmap_active['milestone']!r}")
    if not set(active_task_ids).issubset(set(NEXT_TASK_IDS)):
        failed_preconditions.append(f"active_roadmap_task_ids={active_task_ids}")
    if roadmap_next["present"] and not roadmap_next["parsed"]:
        failed_preconditions.append("next_roadmap_unparseable")
    if missing_or_malformed:
        failed_preconditions.append(
            f"missing_or_malformed_declared_deliverables={missing_or_malformed}"
        )
    if missing_conductor_task_ids:
        failed_preconditions.append(f"missing_conductor_outcomes={missing_conductor_task_ids}")
    if (
        conductor_rows["exp5791-arc-sota-independent-hypothesis-panel"][
            "delivery_failure_count"
        ]
        != 3
    ):
        failed_preconditions.append("exp5791_delivery_failure_count_not_3")
    if collision_scan["preexisting_collision_count"]:
        failed_preconditions.append(
            f"next_range_collision_count={collision_scan['preexisting_collision_count']}"
        )
    if not research_roadmap_unchanged:
        failed_preconditions.append("research_roadmap_modified")
    if not conductor_unchanged:
        failed_preconditions.append("research_conductor_modified")

    status = "blocked" if failed_preconditions else "complete"
    run_rows = [dict(row) for row in (tests_run if tests_run is not None else DEFAULT_TESTS_RUN)]
    append_count = int(complete_stats["research_complete_append_count"])
    docs_mode = (
        "already_archived_preserving_duplicate_history_no_rewrite"
        if append_count == 0
        else "append_v516_completion_once_if_emit_is_called"
    )

    matrix_with_outcomes: list[JsonDict] = []
    for row in declared_matrix:
        task_id = str(row["task_id"])
        merged = dict(row)
        merged["canonical_artifact_path"] = canonical_hashes[task_id]["path"]
        merged["canonical_artifact_sha256"] = canonical_hashes[task_id]["sha256"]
        merged["canonical_artifact_status"] = canonical_hashes[task_id]["status"]
        merged["conductor_outcome"] = conductor_rows[task_id]["outcome"]
        merged["conductor_latest_evidence_line"] = conductor_rows[task_id]["latest_evidence_line"]
        merged["outcome_class"] = _outcome_class(task_id)
        matrix_with_outcomes.append(merged)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": {},
        "preconditions_checked": {
            "roadmaps": {"active": roadmap_active, "next": roadmap_next},
            "input_hashes": _input_hashes(root, canonical_hashes),
            "declared_deliverable_count": len(declared_matrix),
            "canonical_artifact_count": len(canonical_hashes),
            "canonical_hash_count": sum(1 for row in canonical_hashes.values() if row["sha256"]),
            "same_number_alias_group_count": len(alias_groups),
            "next_range_collision_count": collision_scan["preexisting_collision_count"],
            "resource_receipts": _resource_receipts(root),
            "research_roadmap_unchanged": research_roadmap_unchanged,
            "conductor_unchanged": conductor_unchanged,
            **complete_stats,
            "failed_preconditions": failed_preconditions,
        },
        "milestone_from": MILESTONE_FROM,
        "milestone_to": MILESTONE_TO,
        "canonical_identity_contract": {
            "identity_tuple": ["milestone", "task_id", "declared_deliverable"],
            "canonical_path_rule": "exact declared_deliverable only",
            "numeric_prefix_matches": "aliases_only",
            "selection_policy": ARTIFACT_SELECTION_POLICY,
        },
        "declared_deliverable_matrix": matrix_with_outcomes,
        "canonical_artifact_hashes": canonical_hashes,
        "same_number_alias_groups": alias_groups,
        "artifact_selection_policy": ARTIFACT_SELECTION_POLICY,
        "conductor_outcomes": conductor_rows,
        "positive_result_task_ids": list(POSITIVE_RESULT_TASK_IDS),
        "scientific_null_task_ids": list(SCIENTIFIC_NULL_TASK_IDS),
        "negative_result_task_ids": list(NEGATIVE_RESULT_TASK_IDS),
        "blocked_precondition_task_ids": list(BLOCKED_PRECONDITION_TASK_IDS),
        "blocked_gate_task_ids": list(BLOCKED_GATE_TASK_IDS),
        "failed_delivery_task_ids": list(FAILED_DELIVERY_TASK_IDS),
        "missing_task_ids": list(MISSING_TASK_IDS),
        "retired_task_ids": _retired_task_ids(root),
        "no_solve_task_ids": list(NO_SOLVE_TASK_IDS),
        "operational_task_ids": list(OPERATIONAL_TASK_IDS),
        "archived_task_ids": list(EXPECTED_TASK_IDS),
        "research_complete_append_count": append_count,
        "duplicate_history_diagnostics": duplicate_history,
        "collision_scan": collision_scan,
        "next_task_range": NEXT_TASK_RANGE,
        "next_range_collision_count": collision_scan["preexisting_collision_count"],
        "docs_reconciled": {
            "mode": docs_mode,
            "research_complete_append_count": append_count,
            "research_complete_milestone_from_block_count": complete_stats[
                "research_complete_milestone_from_block_count"
            ],
            "files_modified": [],
            "transition_owned_files": [
                SPEC_RELATIVE_PATH.as_posix(),
                "python/carnot/experiment_5796_transition_v517.py",
                "tests/python/test_experiment_5796_transition_v517.py",
                RESULT_RELATIVE_PATH.as_posix(),
            ],
        },
        "research_roadmap_unchanged": research_roadmap_unchanged,
        "conductor_unchanged": conductor_unchanged,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": [str(row.get("command")) for row in run_rows],
        "test_exit_codes": _test_exit_codes(run_rows),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "blocked: exp5796 transition preconditions failed: " + "; ".join(failed_preconditions)
            if failed_preconditions
            else (
                "complete: archived terminal .516 evidence by exact declared deliverables "
                "into .517; positive/null/negative/gate/missing/no-solve classes preserved; "
                "next_range_collision_count=0; research_complete_append_count="
                f"{append_count}"
            )
        ),
    }
    missing_principles = [field for field in artifact if field not in FIELD_PRINCIPLES]
    if missing_principles:
        raise KeyError(f"missing field principles: {missing_principles}")
    artifact["field_principles"] = {field: FIELD_PRINCIPLES[field] for field in artifact}
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def emit_report(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
) -> JsonDict:
    artifact = build_report(
        root, tests_run=tests_run, modification_overrides=modification_overrides
    )
    write_json(output_path or root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    args = parser.parse_args(argv)
    emit_report(args.root, output_path=args.output, tests_run=_load_tests_run(args.tests_run_json))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
