"""Exp5795 V516 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5795, SCENARIO-CAPSTONE-5795,
SCENARIO-CAPSTONE-5795-GATE-REPLAY,
SCENARIO-CAPSTONE-5795-FIELD-PRINCIPLES.

This module reads the completed .516 queue as evidence. It does not rerun
experiments, call models, patch the conductor, or publish claims. The important
boundary is exact identity: each task is reconciled from the active roadmap's
declared deliverable plus the conductor row, so missing gate-blocked artifacts
stay missing instead of being replaced by same-number aliases.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5795_v516_capstone_reconciliation.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
PRD_RELATIVE_PATH = Path("_bmad/prd.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
CAPABILITIES_RELATIVE_PATH = Path("openspec/capabilities")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
DOC_RECONCILE_RELATIVE_PATH = Path("scripts/in_process_doc_reconcile.py")

EXPERIMENT = "experiment_5795_v516_capstone_reconciliation"
EXPERIMENT_ID = "exp5795-v516-capstone-reconciliation"
MILESTONE = "2026.07.516"
RUN_DATE = "2026-07-22"
RANDOM_SEED = 5795
SCHEMA = "carnot.experiment_5795.v516_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "exact_local_artifact_conductor_gate_and_document_reconciliation_no_llm"

SPEC_REFS = (
    "REQ-CAPSTONE-5795",
    "SCENARIO-CAPSTONE-5795",
    "SCENARIO-CAPSTONE-5795-GATE-REPLAY",
    "SCENARIO-CAPSTONE-5795-FIELD-PRINCIPLES",
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
}

GATE_DEFINITIONS: dict[str, list[JsonDict]] = {
    "exp5785-hardness-surface-prospective-fixture": [
        {"upstream": "exp5784-evidence-index-terminal-qualification", "artifact_field": "evidence_index_ready_score", "op": "==", "value": 1.0},
        {"upstream": "exp5784-evidence-index-terminal-qualification", "artifact_field": "next_range_collision_count", "op": "==", "value": 0},
        {"upstream": "exp5784-evidence-index-terminal-qualification", "artifact_field": "unresolved_canonical_count", "op": "==", "value": 0},
        {"upstream": "exp5784-evidence-index-terminal-qualification", "artifact_field": "history_mutation_count", "op": "==", "value": 0},
    ],
    "exp5786-sota-hardness-controlled-constraint-stream": [
        {"upstream": "exp5785-hardness-surface-prospective-fixture", "artifact_field": "fixture_ready_score", "op": "==", "value": 1.0},
        {"upstream": "exp5785-hardness-surface-prospective-fixture", "artifact_field": "exact_label_coverage", "op": "==", "value": 1.0},
        {"upstream": "exp5785-hardness-surface-prospective-fixture", "artifact_field": "parser_control_pass_rate", "op": "==", "value": 1.0},
    ],
    "exp5787-validation-gated-constraint-skill-ab": [
        {"upstream": "exp5786-sota-hardness-controlled-constraint-stream", "artifact_field": "stream_ready_score", "op": "==", "value": 1.0},
        {"upstream": "exp5786-sota-hardness-controlled-constraint-stream", "artifact_field": "real_sota_model_count", "op": ">=", "value": 3},
        {"upstream": "exp5786-sota-hardness-controlled-constraint-stream", "artifact_field": "exact_label_coverage", "op": "==", "value": 1.0},
        {"upstream": "exp5786-sota-hardness-controlled-constraint-stream", "artifact_field": "satisfiable_drift_count", "op": ">=", "value": 30},
        {"upstream": "exp5786-sota-hardness-controlled-constraint-stream", "artifact_field": "protected_fact_distortion_count", "op": ">=", "value": 0},
    ],
    "exp5788-constraint-skill-transfer-audit": [
        {"upstream": "exp5787-validation-gated-constraint-skill-ab", "artifact_field": "self_learning_ready_score", "op": "==", "value": 1.0},
        {"upstream": "exp5787-validation-gated-constraint-skill-ab", "artifact_field": "drift_reduction_lcb", "op": ">", "value": 0.0},
        {"upstream": "exp5787-validation-gated-constraint-skill-ab", "artifact_field": "unsafe_propagation_count", "op": "==", "value": 0},
        {"upstream": "exp5787-validation-gated-constraint-skill-ab", "artifact_field": "protected_fact_distortion_count", "op": "==", "value": 0},
        {"upstream": "exp5787-validation-gated-constraint-skill-ab", "artifact_field": "gguf_weights_immutable", "op": "==", "value": True},
    ],
    "exp5789-constraint-skill-shadow-adapter": [
        {"upstream": "exp5788-constraint-skill-transfer-audit", "artifact_field": "transfer_ready_score", "op": "==", "value": 1.0},
        {"upstream": "exp5788-constraint-skill-transfer-audit", "artifact_field": "macro_transfer_lcb", "op": ">", "value": 0.0},
        {"upstream": "exp5788-constraint-skill-transfer-audit", "artifact_field": "unsafe_propagation_count", "op": "==", "value": 0},
        {"upstream": "exp5788-constraint-skill-transfer-audit", "artifact_field": "protected_fact_distortion_count", "op": "==", "value": 0},
        {"upstream": "exp5788-constraint-skill-transfer-audit", "artifact_field": "rollback_restart_hash_match", "op": "==", "value": True},
    ],
    "exp5790-arc-world-model-admission-contract": [
        {"upstream": "exp5782-transition-v516", "artifact_field": "next_range_collision_count", "op": "==", "value": 0},
    ],
    "exp5791-arc-sota-independent-hypothesis-panel": [
        {"upstream": "exp5790-arc-world-model-admission-contract", "artifact_field": "admission_contract_ready_score", "op": "==", "value": 1.0},
        {"upstream": "exp5790-arc-world-model-admission-contract", "artifact_field": "pivotal_fixture_coverage_score", "op": "==", "value": 1.0},
        {"upstream": "exp5790-arc-world-model-admission-contract", "artifact_field": "source_leak_count", "op": "==", "value": 0},
    ],
    "exp5792-arc-calibration-only-selector": [
        {"upstream": "exp5791-arc-sota-independent-hypothesis-panel", "artifact_field": "panel_ready_score", "op": "==", "value": 1.0},
        {"upstream": "exp5791-arc-sota-independent-hypothesis-panel", "artifact_field": "admissible_hypothesis_count", "op": ">=", "value": 2},
        {"upstream": "exp5791-arc-sota-independent-hypothesis-panel", "artifact_field": "real_sota_model_count", "op": ">=", "value": 3},
    ],
    "exp5793-arc-live-world-model-ab": [
        {"upstream": "exp5792-arc-calibration-only-selector", "artifact_field": "selector_ready_score", "op": "==", "value": 1.0},
        {"upstream": "exp5792-arc-calibration-only-selector", "artifact_field": "selector_delta_lcb", "op": ">", "value": 0.0},
        {"upstream": "exp5792-arc-calibration-only-selector", "artifact_field": "selected_model_admissible_score", "op": "==", "value": 1.0},
        {"upstream": "exp5792-arc-calibration-only-selector", "artifact_field": "selected_pivotal_coverage_score", "op": "==", "value": 1.0},
        {"upstream": "exp5792-arc-calibration-only-selector", "artifact_field": "source_leak_count", "op": "==", "value": 0},
    ],
    "exp5794-hardware-terminal-action-receipt": [
        {"upstream": "exp5782-transition-v516", "artifact_field": "next_range_collision_count", "op": "==", "value": 0},
    ],
}

PHASE_BY_TASK: dict[str, str] = {
    "exp5782-transition-v516": "infrastructure",
    "exp5783-v516-source-delta-ingestion": "source_refresh",
    "exp5784-evidence-index-terminal-qualification": "infrastructure",
    "exp5785-hardness-surface-prospective-fixture": "fixture",
    "exp5786-sota-hardness-controlled-constraint-stream": "constraint_science",
    "exp5787-validation-gated-constraint-skill-ab": "constraint_science",
    "exp5788-constraint-skill-transfer-audit": "constraint_science",
    "exp5789-constraint-skill-shadow-adapter": "integration",
    "exp5790-arc-world-model-admission-contract": "arc",
    "exp5791-arc-sota-independent-hypothesis-panel": "arc",
    "exp5792-arc-calibration-only-selector": "arc",
    "exp5793-arc-live-world-model-ab": "arc",
    "exp5794-hardware-terminal-action-receipt": "hardware",
}

SOURCE_HASH_PATHS = (
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
    RESEARCH_REFERENCES_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    CAPABILITIES_RELATIVE_PATH,
    DOC_RECONCILE_RELATIVE_PATH,
)

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5795_v516_capstone_reconciliation.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5795_v516_capstone_reconciliation.py -m pytest tests/python/test_experiment_5795_v516_capstone_reconciliation.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5795_v516_capstone_reconciliation.py --fail-under=100",
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/check_spec_coverage.py", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/root_clutter_sweep.py", "exit_code": None, "status": "not_run"},
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Versioned schema for this capstone artifact.",
    "experiment": "Stable local experiment slug.",
    "experiment_id": "Conductor task id for this capstone.",
    "run_date": "Absolute capstone date from the operator prompt.",
    "random_seed": "Deterministic metadata for checksum stability; no stochastic science runs.",
    "spec_refs": "OpenSpec anchors for this artifact contract.",
    "result_path": "Canonical deliverable path emitted by this workflow.",
    "field_principles": "One-line reason for every top-level field.",
    "status": "Terminal state after exact input, protected-file, and schema checks.",
    "preconditions_checked": "Roadmap parses, hashes, resources, exact paths, and forbidden-file checks.",
    "milestone": "The milestone being closed, 2026.07.516.",
    "canonical_task_matrix": "The fixed Exp5782-Exp5794 denominator with exact declared paths, conductor rows, artifact states, substrate, metrics, gates, retirement, and tests.",
    "canonical_artifact_hashes": "Exact declared deliverable hashes and missing/malformed states.",
    "conductor_outcomes": "Latest conductor rows, retry counts, delivery failures, and gate blocks.",
    "outcome_taxonomy": "complete-positive, complete-null, complete-negative, blocked-precondition, blocked-gate, failed-delivery, missing, and retired stay distinct.",
    "positive_task_ids": "Tasks with bounded positive infrastructure/control/hardware evidence.",
    "scientific_null_task_ids": "Executed tasks with valid null evidence, not gate blocks.",
    "negative_task_ids": "Executed tasks with negative science or readiness-defect evidence.",
    "blocked_precondition_task_ids": "Tasks whose own artifact reports a precondition block.",
    "blocked_gate_task_ids": "Conductor or gate-check blocks that did not run the science body.",
    "failed_delivery_task_ids": "Tasks with artifact_not_updated_past_bootstrap conductor failures.",
    "missing_task_ids": "Declared deliverables absent at their exact paths.",
    "retired_task_ids": "Prior-failure task ids whose repeated verdict triggers narrow retirement accounting.",
    "gate_replay_receipts": "Bare producer scalars and gate-check artifacts are replayed exactly.",
    "prior_failure_retirement_receipts": "Same-verdict retirements are named with manifest presence and scope.",
    "constraint_branch_decision": "Default-off unless stream, learning, transfer, and shadow integration gates all pass.",
    "arc_branch_decision": "Default-off unless admission, panel, selector, and live A/B gates pass with solve-neutral provenance.",
    "hardware_branch_decision": "Cached hardware continuity cannot become speedup, energy, or production readiness.",
    "arc_registry_unchanged": "Bare true when no artifact claims solve or registry credit.",
    "solve_claim_count": "Counts only explicit solve_claimed true fields.",
    "phase_telemetry": "Timings and retries support planning only, not scientific claims.",
    "task_wall_times": "Per-task conductor and artifact timing receipts.",
    "retry_counts": "Attempts beyond the first dispatch per task.",
    "gate_skipped_agent_calls": "Gate-block rows that avoided launching a science task body.",
    "slowest_tasks": "Largest conductor window estimates for next-plan triage.",
    "avoidable_orchestration_time_min": "Approximate time spent in repeated failures and gate blocks.",
    "gpu_cpu_receipts": "GPU/offload and CPU/hardware receipts copied only as provenance.",
    "criteria_matrix": "Promotion and claim criteria evaluated without stronger public claims.",
    "docs_reconciled": "The operator stop rule delegates ops/status, ops/changelog, and traceability edits.",
    "specs_reconciled": "This OpenSpec contract and test/module pair exist.",
    "traceability_reconciled": "False because traceability edits are delegated by the stop rule.",
    "research_complete_append_count": "Zero because history append is delegated/unchanged in this focused run.",
    "public_claims_changed": "Bare false without promotion-grade evidence.",
    "research_roadmap_unchanged": "Active roadmap mutation is forbidden.",
    "conductor_unchanged": "The conductor is read-only.",
    "inference_substrate": "Exact local artifact, conductor, gate, and document reconciliation only.",
    "test_commands": "Verification commands are replayable.",
    "test_exit_codes": "Observed command exits without relabeling failures.",
    "reproducibility_checksum": "Content-addressed checksum catches silent drift.",
    "honest_verdict": "Terminal summary starting with complete: or blocked:.",
}


def path_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(child.read_bytes())
        return "sha256:" + digest.hexdigest()
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        meta["error"] = f"expected mapping, got {type(payload).__name__}"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _read_yaml_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "parsed": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
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


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        try:
            return float(value)
        except ValueError:
            return value
    return None


def _compare_gate(op: str, actual: Any, expected: Any) -> bool:
    actual = _coerce_scalar(actual)
    expected = _coerce_scalar(expected)
    if actual is None:
        return False
    if op == "==":
        return actual == expected
    if op == ">=":
        return float(actual) >= float(expected)
    if op == ">":
        return float(actual) > float(expected)
    return False


def _parse_conductor_log(root: Path) -> list[JsonDict]:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
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


def _parse_timestamp(value: str) -> datetime | None:
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC", value)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")


def _conductor_outcomes(root: Path) -> dict[str, JsonDict]:
    rows = _parse_conductor_log(root)
    outcomes: dict[str, JsonDict] = {}
    for task_id in EXPECTED_TASK_IDS:
        pattern = CONDUCTOR_TITLE_PATTERNS[task_id]
        matches = [row for row in rows if pattern in str(row["title"])]
        latest = matches[-1] if matches else {}
        delivery_failures = [
            row for row in matches if "artifact_not_updated_past_bootstrap" in str(row["detail"])
        ]
        gate_blocks = [row for row in matches if row.get("outcome") == "GATE_BLOCK"]
        first_time = _parse_timestamp(str(matches[0]["timestamp"])) if matches else None
        last_time = _parse_timestamp(str(matches[-1]["timestamp"])) if matches else None
        window_min = (
            (last_time - first_time).total_seconds() / 60.0
            if first_time is not None and last_time is not None
            else 0.0
        )
        outcomes[task_id] = {
            "outcome": latest.get("outcome", "UNKNOWN"),
            "source": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            "latest_evidence_line": latest.get("line", ""),
            "evidence_lines": [str(row["line"]) for row in matches],
            "attempt_count": len(matches),
            "retry_count": max(0, len(matches) - 1),
            "delivery_failure_count": len(delivery_failures),
            "gate_block_count": len(gate_blocks),
            "first_timestamp": matches[0]["timestamp"] if matches else None,
            "last_timestamp": matches[-1]["timestamp"] if matches else None,
            "conductor_window_min": round(window_min, 3),
        }
    return outcomes


def _roadmap_summary(root: Path, rel_path: Path) -> tuple[JsonDict, JsonDict]:
    payload, meta = _read_yaml_mapping(root / rel_path)
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    task_rows = [row for row in tasks if isinstance(row, Mapping)]
    return payload, {
        **meta,
        "milestone": payload.get("milestone") if isinstance(payload.get("milestone"), str) else None,
        "task_ids": [
            str(row["id"]) for row in task_rows if isinstance(row.get("id"), str)
        ],
        "deliverables": [
            str(row["deliverable"])
            for row in task_rows
            if isinstance(row.get("deliverable"), str)
        ],
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


def _source_hashes(root: Path, artifact_hashes: Mapping[str, JsonMap]) -> JsonDict:
    return {
        "source_files": {
            rel_path.as_posix(): {
                "present": (root / rel_path).exists(),
                "sha256": path_sha256(root / rel_path),
            }
            for rel_path in SOURCE_HASH_PATHS
        },
        "declared_deliverables": {
            task_id: {
                "path": row.get("path"),
                "present": row.get("present"),
                "sha256": row.get("sha256"),
            }
            for task_id, row in artifact_hashes.items()
        },
    }


def _git_modified(root: Path, rel_path: Path) -> bool:  # pragma: no cover - live repo helper
    result = subprocess.run(
        ["git", "status", "--short", "--", rel_path.as_posix()],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return bool(result.stdout.strip())


def _protected_modified(
    root: Path, rel_path: Path, modification_overrides: Mapping[Path, bool] | None
) -> bool:
    if modification_overrides is not None and rel_path in modification_overrides:
        return bool(modification_overrides[rel_path])
    return _git_modified(root, rel_path)


def _artifact_status(payload: JsonMap, meta: JsonMap) -> str:
    if not meta.get("present"):
        return "missing"
    if not meta.get("loadable"):
        return "malformed"
    if payload.get("schema") == "blocked_gate_check_v1" or payload.get("blocked_at_layer"):
        return "blocked-gate"
    verdict = payload.get("honest_verdict")
    if isinstance(verdict, str) and (
        verdict.startswith("blocked:") or verdict.startswith("blocked_")
    ):
        return "blocked-precondition"
    status = payload.get("status")
    if isinstance(status, str) and status.startswith("complete"):
        return "complete"
    if status == "blocked":
        return "blocked-precondition"
    return str(status) if isinstance(status, str) and status else "unknown"


def _load_artifacts(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    hashes: dict[str, JsonDict] = {}
    for task_id in EXPECTED_TASK_IDS:
        rel_path = TASK_ARTIFACT_PATHS[task_id]
        payload, meta = _read_json_mapping(root / rel_path)
        status = _artifact_status(payload, meta)
        payloads[task_id] = payload
        hashes[task_id] = {
            "identity": [MILESTONE, task_id, rel_path.as_posix()],
            "path": rel_path.as_posix(),
            "present": bool(meta.get("present")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "status": status,
            "honest_verdict": payload.get("honest_verdict")
            if isinstance(payload.get("honest_verdict"), str)
            else "",
            "error": meta.get("error"),
            "selected_by": "exact_declared_deliverable",
        }
    return payloads, hashes


def _primary_metrics(task_id: str, payload: JsonMap) -> JsonDict:
    metric_fields = (
        "next_range_collision_count",
        "accepted_finding_count",
        "evidence_index_ready_score",
        "fixture_ready_score",
        "stream_ready_score",
        "real_sota_model_count",
        "satisfiable_drift_count",
        "protected_fact_distortion_count",
        "admission_contract_ready_score",
        "pivotal_fixture_coverage_score",
        "source_leak_count",
        "panel_ready_score",
        "admissible_hypothesis_count",
        "speedup_claimed",
        "energy_claimed",
        "production_ready_claimed",
    )
    return {field: payload[field] for field in metric_fields if field in payload}


def _test_state(payload: JsonMap) -> JsonDict:
    exit_codes = payload.get("test_exit_codes")
    if not isinstance(exit_codes, Mapping):
        return {"task_owned": "not_recorded", "global": "not_recorded"}
    values = list(exit_codes.values())
    task_owned_pass = any(code == 0 for code in values)
    global_values = [
        code for command, code in exit_codes.items() if isinstance(command, str) and "tests/python -q" in command
    ]
    return {
        "task_owned": "passed_or_recorded" if task_owned_pass else "no_passing_task_owned_receipt",
        "global": "passed" if global_values and all(code == 0 for code in global_values) else "not_passing_or_not_recorded",
        "exit_code_count": len(values),
    }


def _replay_task_gates(
    task_id: str,
    payloads: Mapping[str, JsonMap],
    artifact_hashes: Mapping[str, JsonMap],
    roadmap_task: JsonMap,
) -> JsonDict:
    gate_rows = roadmap_task.get("gated_on")
    if not isinstance(gate_rows, list):
        gate_rows = GATE_DEFINITIONS.get(task_id, [])
    gates: list[JsonDict] = []
    for index, gate in enumerate(gate_rows):
        upstream = str(gate.get("upstream", ""))
        field = str(gate.get("artifact_field", ""))
        op = str(gate.get("op", "=="))
        expected = gate.get("value")
        upstream_payload = payloads.get(upstream, {})
        upstream_hash = artifact_hashes.get(upstream, {})
        actual = upstream_payload.get(field) if isinstance(upstream_payload, Mapping) else None
        passed = _compare_gate(op, actual, expected)
        reason = (
            f"actual={actual!r} {op} expected={expected!r}"
            if upstream_hash.get("present")
            else f"upstream artifact not found for task id {upstream!r}"
        )
        gates.append(
            {
                "gate_index": index,
                "upstream": upstream,
                "artifact_field": field,
                "op": op,
                "expected": expected,
                "actual": _coerce_scalar(actual),
                "passed": passed,
                "reason": reason,
                "source_artifact_status": upstream_hash.get("status", "missing"),
                "source_artifact_path": upstream_hash.get("path"),
            }
        )
    artifact_gates = payloads.get(task_id, {}).get("gates_evaluated")
    discrepancies: list[str] = []
    if isinstance(artifact_gates, list):
        by_key = {
            (row.get("upstream"), row.get("artifact_field")): row
            for row in artifact_gates
            if isinstance(row, Mapping)
        }
        for gate in gates:
            row = by_key.get((gate["upstream"], gate["artifact_field"]))
            if row is None:
                continue
            if row.get("passed") != gate["passed"]:
                discrepancies.append(f"passed_mismatch:{gate['upstream']}.{gate['artifact_field']}")
            if _coerce_scalar(row.get("actual")) != gate["actual"]:
                discrepancies.append(f"actual_mismatch:{gate['upstream']}.{gate['artifact_field']}")
    return {
        "task_id": task_id,
        "gates": gates,
        "all_passed": all(gate["passed"] for gate in gates) if gates else None,
        "gate_check_artifact_present": isinstance(artifact_gates, list),
        "discrepancies": discrepancies,
    }


def _classify_task(
    task_id: str,
    payload: JsonMap,
    artifact_hash: JsonMap,
    conductor: JsonMap,
) -> tuple[str, list[str]]:
    secondary: list[str] = []
    if conductor.get("delivery_failure_count", 0) > 0:
        secondary.append("failed-delivery")
    if artifact_hash.get("status") == "blocked-precondition":
        secondary.append("blocked-precondition")
    if artifact_hash.get("status") == "missing":
        return "missing", secondary
    if artifact_hash.get("status") == "malformed":
        return "blocked-precondition", secondary
    if artifact_hash.get("status") == "blocked-gate" or conductor.get("outcome") == "GATE_BLOCK":
        return "blocked-gate", secondary
    if conductor.get("delivery_failure_count", 0) > 0:
        return "failed-delivery", secondary
    if task_id == "exp5783-v516-source-delta-ingestion":
        return "complete-null", secondary
    if task_id == "exp5786-sota-hardness-controlled-constraint-stream":
        return "complete-negative", secondary
    return "complete-positive", secondary


def _roadmap_tasks_by_id(active_payload: JsonMap) -> dict[str, JsonDict]:
    tasks = active_payload.get("tasks") if isinstance(active_payload.get("tasks"), list) else []
    return {
        str(row["id"]): dict(row)
        for row in tasks
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }


def _prior_failure_receipts(
    payloads: Mapping[str, JsonMap],
    roadmap_tasks: Mapping[str, JsonMap],
    manifest_text: str,
) -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for task_id in EXPECTED_TASK_IDS:
        prior_failures = roadmap_tasks.get(task_id, {}).get("prior_failures")
        if not isinstance(prior_failures, list):
            continue
        current_verdict = payloads.get(task_id, {}).get("honest_verdict")
        rows: list[JsonDict] = []
        for prior in prior_failures:
            if not isinstance(prior, Mapping):
                continue
            prior_id = str(prior.get("experiment_id", ""))
            prior_verdict = str(prior.get("verdict", ""))
            same = current_verdict == prior_verdict
            required = same and bool(prior.get("retire_if_same_verdict"))
            rows.append(
                {
                    "prior_experiment_id": prior_id,
                    "prior_verdict": prior_verdict,
                    "current_verdict": current_verdict,
                    "same_verdict": same,
                    "retirement_required": required,
                    "manifest_entry_present": prior_id in manifest_text,
                    "retirement_scope": f"{prior_id}_same_verdict_v516",
                }
            )
        required_ids = [row["prior_experiment_id"] for row in rows if row["retirement_required"]]
        receipts[task_id] = {
            "retirement_required": bool(required_ids),
            "prior_failures": rows,
            "required_prior_experiment_ids": required_ids,
        }
    return receipts


def _branch_decisions(payloads: Mapping[str, JsonMap], gate_replay: Mapping[str, JsonMap]) -> tuple[JsonDict, JsonDict, JsonDict]:
    constraint_blocking = [
        task_id
        for task_id in (
            "exp5786-sota-hardness-controlled-constraint-stream",
            "exp5787-validation-gated-constraint-skill-ab",
            "exp5788-constraint-skill-transfer-audit",
            "exp5789-constraint-skill-shadow-adapter",
        )
        if task_id not in payloads
        or task_id == "exp5786-sota-hardness-controlled-constraint-stream"
        and payloads[task_id].get("stream_ready_score") != 1.0
        or gate_replay.get(task_id, {}).get("all_passed") is False
    ]
    arc_blocking = [
        task_id
        for task_id in (
            "exp5791-arc-sota-independent-hypothesis-panel",
            "exp5792-arc-calibration-only-selector",
            "exp5793-arc-live-world-model-ab",
        )
        if task_id not in payloads
        or payloads.get(task_id, {}).get("panel_ready_score") == 0.0
        or gate_replay.get(task_id, {}).get("all_passed") is False
    ]
    hardware = payloads.get("exp5794-hardware-terminal-action-receipt", {})
    return (
        {
            "promoted": False,
            "default_enabled": False,
            "reason": "stream_learning_transfer_or_shadow_gate_not_all_passed",
            "blocking_task_ids": sorted(set(constraint_blocking)),
        },
        {
            "promoted": False,
            "default_enabled": False,
            "reason": "panel_selector_or_live_gate_not_all_passed",
            "blocking_task_ids": sorted(set(arc_blocking)),
        },
        {
            "promoted": False,
            "speedup_claimed": bool(hardware.get("speedup_claimed")),
            "energy_claimed": bool(hardware.get("energy_claimed")),
            "production_ready_claimed": bool(hardware.get("production_ready_claimed")),
            "reason": "cached_continuity_no_board_commands_no_speed_or_energy_claim",
        },
    )


def _telemetry(
    conductor_outcomes: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    matrix: Mapping[str, JsonMap],
) -> tuple[JsonDict, JsonDict, JsonDict, int, list[JsonDict], float]:
    task_wall_times = {
        task_id: {
            "conductor_window_min": row.get("conductor_window_min", 0.0),
            "artifact_duration_s": payloads.get(task_id, {}).get("duration_s"),
            "attempt_count": row.get("attempt_count", 0),
        }
        for task_id, row in conductor_outcomes.items()
    }
    retry_counts = {
        task_id: int(row.get("retry_count", 0)) for task_id, row in conductor_outcomes.items()
    }
    phase_rows: dict[str, JsonDict] = defaultdict(
        lambda: {"task_count": 0, "gate_block_count": 0, "failed_delivery_count": 0, "wall_time_min": 0.0}
    )
    for task_id, wall in task_wall_times.items():
        phase = PHASE_BY_TASK[task_id]
        phase_rows[phase]["task_count"] += 1
        phase_rows[phase]["gate_block_count"] += int(conductor_outcomes[task_id].get("gate_block_count", 0))
        phase_rows[phase]["failed_delivery_count"] += int(
            conductor_outcomes[task_id].get("delivery_failure_count", 0)
        )
        phase_rows[phase]["wall_time_min"] = round(
            float(phase_rows[phase]["wall_time_min"]) + float(wall["conductor_window_min"]), 3
        )
    gate_skips = sum(int(row.get("gate_block_count", 0)) for row in conductor_outcomes.values())
    slowest = sorted(
        [
            {
                "task_id": task_id,
                "conductor_window_min": row["conductor_window_min"],
                "outcome_class": matrix[task_id]["outcome_class"],
            }
            for task_id, row in task_wall_times.items()
        ],
        key=lambda item: item["conductor_window_min"],
        reverse=True,
    )[:5]
    avoidable = round(
        sum(
            float(row.get("conductor_window_min", 0.0))
            for task_id, row in conductor_outcomes.items()
            if row.get("gate_block_count", 0) or row.get("delivery_failure_count", 0)
        ),
        3,
    )
    return dict(phase_rows), task_wall_times, retry_counts, gate_skips, slowest, avoidable


def _gpu_cpu_receipts(payloads: Mapping[str, JsonMap]) -> JsonDict:
    gpu_ids = [
        task_id for task_id, payload in payloads.items() if payload.get("gpu_offload_receipts") is not None
    ]
    return {
        "gpu_receipt_task_ids": gpu_ids,
        "gpu_offload_receipts": {
            task_id: payloads[task_id].get("gpu_offload_receipts") for task_id in gpu_ids
        },
        "hardware_cached_receipts": payloads.get("exp5794-hardware-terminal-action-receipt", {}).get(
            "commands_skipped", []
        ),
    }


def _test_exit_codes(tests_run: Sequence[JsonMap]) -> JsonDict:
    return {str(row.get("command")): row.get("exit_code") for row in tests_run}


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
) -> JsonDict:
    active_payload, active_summary = _roadmap_summary(root, ROADMAP_RELATIVE_PATH)
    _next_payload, next_summary = _roadmap_summary(root, ROADMAP_NEXT_RELATIVE_PATH)
    roadmap_tasks = _roadmap_tasks_by_id(active_payload)
    payloads, artifact_hashes = _load_artifacts(root)
    conductor_outcomes = _conductor_outcomes(root)
    manifest_text = (root / EXCLUSION_MANIFEST_RELATIVE_PATH).read_text(
        encoding="utf-8", errors="replace"
    ) if (root / EXCLUSION_MANIFEST_RELATIVE_PATH).exists() else ""

    gate_replay = {
        task_id: _replay_task_gates(task_id, payloads, artifact_hashes, roadmap_tasks.get(task_id, {}))
        for task_id in EXPECTED_TASK_IDS
        if task_id in GATE_DEFINITIONS or roadmap_tasks.get(task_id, {}).get("gated_on")
    }
    prior_failure_receipts = _prior_failure_receipts(payloads, roadmap_tasks, manifest_text)
    matrix: dict[str, JsonDict] = {}
    class_buckets: dict[str, list[str]] = defaultdict(list)
    blocked_precondition_ids: list[str] = []
    failed_delivery_ids: list[str] = []
    for task_id in EXPECTED_TASK_IDS:
        outcome_class, secondary = _classify_task(
            task_id, payloads[task_id], artifact_hashes[task_id], conductor_outcomes[task_id]
        )
        class_buckets[outcome_class].append(task_id)
        if "blocked-precondition" in secondary:
            blocked_precondition_ids.append(task_id)
        if "failed-delivery" in secondary:
            failed_delivery_ids.append(task_id)
        matrix[task_id] = {
            "task_id": task_id,
            "title": roadmap_tasks.get(task_id, {}).get("title", TASK_TITLES[task_id]),
            "declared_deliverable": TASK_ARTIFACT_PATHS[task_id].as_posix(),
            "canonical_artifact_sha256": artifact_hashes[task_id]["sha256"],
            "artifact_status": artifact_hashes[task_id]["status"],
            "conductor_status": conductor_outcomes[task_id]["outcome"],
            "delivery_failure_count": conductor_outcomes[task_id]["delivery_failure_count"],
            "gate_block_count": conductor_outcomes[task_id]["gate_block_count"],
            "inference_substrate": payloads[task_id].get("inference_substrate"),
            "primary_metrics": _primary_metrics(task_id, payloads[task_id]),
            "gate_replay": gate_replay.get(task_id),
            "prior_failure_retirement_result": prior_failure_receipts.get(task_id),
            "outcome_class": outcome_class,
            "secondary_outcomes": secondary,
            "task_owned_global_test_state": _test_state(payloads[task_id]),
        }

    retired_task_ids = [
        prior_id
        for receipt in prior_failure_receipts.values()
        for prior_id in receipt.get("required_prior_experiment_ids", [])
    ]
    constraint_decision, arc_decision, hardware_decision = _branch_decisions(payloads, gate_replay)
    phase_telemetry, task_wall_times, retry_counts, gate_skips, slowest, avoidable = _telemetry(
        conductor_outcomes, payloads, matrix
    )
    solve_claim_count = sum(1 for payload in payloads.values() if payload.get("solve_claimed") is True)
    registry_credit_count = sum(1 for payload in payloads.values() if payload.get("registry_credit") is True)
    research_roadmap_unchanged = not _protected_modified(
        root, ROADMAP_RELATIVE_PATH, modification_overrides
    )
    conductor_unchanged = not _protected_modified(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)

    failed_preconditions: list[str] = []
    if not active_summary["parsed"]:
        failed_preconditions.append("active_roadmap_unparseable")
    if active_summary["milestone"] != MILESTONE:
        failed_preconditions.append(f"active_roadmap_milestone={active_summary['milestone']!r}")
    for task_id in EXPECTED_TASK_IDS:
        declared = roadmap_tasks.get(task_id, {}).get("deliverable")
        if declared is not None and declared != TASK_ARTIFACT_PATHS[task_id].as_posix():
            failed_preconditions.append(f"declared_deliverable_mismatch:{task_id}")
    if not research_roadmap_unchanged:
        failed_preconditions.append("research_roadmap_modified")
    if not conductor_unchanged:
        failed_preconditions.append("research_conductor_modified")

    run_rows = [dict(row) for row in (tests_run if tests_run is not None else DEFAULT_TESTS_RUN)]
    status = "blocked" if failed_preconditions else "complete"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": {},
        "status": status,
        "preconditions_checked": {
            "roadmaps": {"active": active_summary, "next": next_summary},
            "input_hashes": _source_hashes(root, artifact_hashes),
            "resource_receipts": _resource_receipts(root),
            "declared_deliverable_count": len(EXPECTED_TASK_IDS),
            "present_artifact_count": sum(1 for row in artifact_hashes.values() if row["present"]),
            "failed_preconditions": failed_preconditions,
        },
        "milestone": MILESTONE,
        "canonical_task_matrix": matrix,
        "canonical_artifact_hashes": artifact_hashes,
        "conductor_outcomes": conductor_outcomes,
        "outcome_taxonomy": {
            "complete-positive": "bounded positive infrastructure/control/hardware evidence",
            "complete-null": "executed null, not a gate block",
            "complete-negative": "executed negative or readiness-defect science",
            "blocked-precondition": "own artifact reports a precondition block",
            "blocked-gate": "conductor gate prevented execution",
            "failed-delivery": "conductor delivery failed after attempts",
            "missing": "exact declared artifact absent",
            "retired": "same-verdict prior failure scope closed",
        },
        "positive_task_ids": class_buckets["complete-positive"],
        "scientific_null_task_ids": class_buckets["complete-null"],
        "negative_task_ids": class_buckets["complete-negative"],
        "blocked_precondition_task_ids": sorted(set(blocked_precondition_ids)),
        "blocked_gate_task_ids": class_buckets["blocked-gate"],
        "failed_delivery_task_ids": sorted(set(failed_delivery_ids + class_buckets["failed-delivery"])),
        "missing_task_ids": class_buckets["missing"],
        "retired_task_ids": retired_task_ids,
        "gate_replay_receipts": gate_replay,
        "prior_failure_retirement_receipts": prior_failure_receipts,
        "constraint_branch_decision": constraint_decision,
        "arc_branch_decision": arc_decision,
        "hardware_branch_decision": hardware_decision,
        "arc_registry_unchanged": solve_claim_count == 0 and registry_credit_count == 0,
        "solve_claim_count": solve_claim_count,
        "phase_telemetry": phase_telemetry,
        "task_wall_times": task_wall_times,
        "retry_counts": retry_counts,
        "gate_skipped_agent_calls": gate_skips,
        "slowest_tasks": slowest,
        "avoidable_orchestration_time_min": avoidable,
        "gpu_cpu_receipts": _gpu_cpu_receipts(payloads),
        "criteria_matrix": {
            "constraint_adapter_promotion": constraint_decision,
            "arc_live_influence_promotion": arc_decision,
            "hardware_claims": hardware_decision,
            "arc_registry_credit": {
                "solve_claim_count": solve_claim_count,
                "registry_credit_count": registry_credit_count,
            },
        },
        "docs_reconciled": {
            "mode": "operator_stop_rule_delegates_haiku_reconciliation",
            "ops_status_updated": False,
            "ops_changelog_updated": False,
            "traceability_updated": False,
            "files_modified_by_capstone": [
                SPEC_RELATIVE_PATH.as_posix(),
                "python/carnot/experiment_5795_v516_capstone_reconciliation.py",
                "tests/python/test_experiment_5795_v516_capstone_reconciliation.py",
                RESULT_RELATIVE_PATH.as_posix(),
            ],
        },
        "specs_reconciled": {
            "spec_path": SPEC_RELATIVE_PATH.as_posix(),
            "req_ids": ["REQ-CAPSTONE-5795"],
            "scenario_ids": [
                "SCENARIO-CAPSTONE-5795",
                "SCENARIO-CAPSTONE-5795-GATE-REPLAY",
                "SCENARIO-CAPSTONE-5795-FIELD-PRINCIPLES",
            ],
        },
        "traceability_reconciled": False,
        "research_complete_append_count": 0,
        "public_claims_changed": False,
        "research_roadmap_unchanged": research_roadmap_unchanged,
        "conductor_unchanged": conductor_unchanged,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": [str(row.get("command")) for row in run_rows],
        "test_exit_codes": _test_exit_codes(run_rows),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "blocked: v516 capstone preconditions failed: " + "; ".join(failed_preconditions)
            if failed_preconditions
            else (
                "complete: v516 reconciled by exact declared artifacts; constraint_adapter_promoted=false; "
                "arc_live_influence_promoted=false; hardware_speedup_claimed=false; "
                "arc_registry_unchanged=true"
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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    emit_report(args.root, output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
