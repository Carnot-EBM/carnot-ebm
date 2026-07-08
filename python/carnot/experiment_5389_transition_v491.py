"""Exp 5389: archive .490 truth and emit the .491 transition context.

Spec refs: REQ-REPORT-5389, SCENARIO-REPORT-5389,
SCENARIO-REPORT-5389-BLOCKED-INPUT.

This module is record-only. It reads the completed .490 capstone, the .491
roadmap context, and the conductor caution around Exp5383, then writes the
first .491 execution artifact. It does not activate or edit roadmaps.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5389_transition_v491.json")
CAPSTONE_RELATIVE_PATH = Path("results/experiment_5388_capstone_v490.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5389_transition_v491"
EXPERIMENT_ID = "exp5389-v491-transition-and-archive"
MILESTONE = "2026.07.491"
PRIOR_MILESTONE = "2026.07.490"
SCHEMA = "carnot.experiment_5389_transition_v491.v1"
RANDOM_SEED = 5389
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")

PRIOR_GATE_FIELDS = (
    "structured_methodology_receipt_ready",
    "structured_protocol_clean",
    "constraint_tax_panel_ready",
    "budget_memory_corrigendum_clean",
    "continuous_self_learning_real_workflow_ready",
    "continuous_self_learning_requirement_satisfied",
    "overwrite_guidance_scale_ready",
    "pbit_boundary_overwrite_ready",
    "arc_new_level_banked",
    "hardware_hash_chained_receipt_ready",
    "hardware_speedup_claim",
    "future_token_signal_allowed",
)

PRIOR_BLOCKER_FIELDS = (
    "solver_tautology",
    "ARC",
    "token_feature",
    "hardware",
)

EXPECTED_TASK_IDS = [
    "exp5389-v491-transition-and-archive",
    "exp5390-v491-sota-source-delta",
    "exp5391-v491-constraint-tax-scaleup-fixtures",
    "exp5392-v491-formal-encoding-safety-fixture",
    "exp5393-v491-overwrite-guidance-tautology-corrigendum",
    "exp5394-v491-gated-overwrite-pbit-ablation",
    "exp5395-v491-influence-share-verifier-budget-router",
    "exp5396-v491-memory-guard-raw-episode-retention",
    "exp5397-v491-arc-blob-salience-live-path",
    "exp5398-v491-hardware-evidence-graph-repeatability",
    "exp5399-v491-kan-dynamic-counterexample-certificate",
    "exp5400-v491-evidence-table-and-prd-gap-analysis",
    "exp5401-v491-capstone",
]

EXPECTED_PHASE_NAMES = [
    "Phase 0 - Source Delta and Expanded Local Fixtures",
    "Phase 1 - Solver Corrigendum and Gated P-bit Ablation",
    "Phase 2 - Continuous Self-Learning and Live ARC Salience",
    "Phase 3 - Evidence Surfaces, Certificates, and Capstone",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "Honest terminal state; complete only when the .490 capstone, .491 roadmap "
        "document, executable .491 task source, and no-edit guards are present."
    ),
    "milestone": (
        "Must equal 2026.07.491 so this artifact cannot be confused with the completed "
        ".490 capstone."
    ),
    "prior_milestone": (
        "Must equal 2026.07.490 because all inherited gates come from the completed "
        "prior milestone."
    ),
    "prior_capstone_path": "Names the exact .490 capstone used as the source of truth.",
    "prior_gate_summary": (
        "Copies only the requested .490 gate booleans without laundering flagged solver, "
        "no-bank ARC, no-speedup hardware, or closed token-feature lanes."
    ),
    "prior_blockers": (
        "Summarizes solver-tautology, ARC, token-feature, and hardware blockers for "
        "downstream gating."
    ),
    "roadmap_next_present": (
        "Bare boolean proving whether the literal pre-staged next-roadmap file existed "
        "at execution time."
    ),
    "roadmap_doc_present": (
        "Bare boolean proving the vNEXT roadmap document existed for task-range and "
        "phase extraction."
    ),
    "planned_task_count": (
        "Counts the .491 tasks in the executable roadmap source used by this transition."
    ),
    "planned_task_ids": (
        "Ordered task ids preserve the Exp5389-5401 conductor execution range for downstream gates."
    ),
    "downstream_gate_expectations": (
        "Lists the structured, self-learning, solver, ARC, token, and hardware "
        "expectations .491 tasks must honor."
    ),
    "active_roadmap_modified": (
        "Must remain false because Exp5389 is record-only and must not edit the active roadmap."
    ),
    "conductor_modified": (
        "Must remain false because Exp5389 must not edit scripts/research_conductor.py."
    ),
    "honest_verdict": (
        "One-line terminal summary that distinguishes a clean transition from a "
        "missing-input block."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone",
    "prior_milestone",
    "prior_capstone_path",
    "prior_gate_summary",
    "prior_blockers",
    "roadmap_next_present",
    "roadmap_doc_present",
    "planned_task_count",
    "planned_task_ids",
    "downstream_gate_expectations",
    "active_roadmap_modified",
    "conductor_modified",
    "honest_verdict",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "spec_refs",
    "result_path",
    "field_principles",
    "duration_s",
    "random_seed",
    "inference_substrate",
    "planned_phase_names",
    "roadmap_doc_task_range",
    "planned_task_source",
    "failed_preconditions",
    "preconditions_checked",
    "cited_upstream_artifacts",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)

SPEC_REFS = [
    "REQ-REPORT-5389",
    "SCENARIO-REPORT-5389",
    "SCENARIO-REPORT-5389-BLOCKED-INPUT",
]

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    CAPSTONE_RELATIVE_PATH,
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    CONDUCTOR_LOG_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

DOWNSTREAM_GATE_EXPECTATIONS: JsonDict = {
    "structured": {
        "expectation": (
            "Exp5391 and Exp5392 must build on .490 clean structured receipts with "
            "deterministic final authority, local SOTA GGUF evidence when models run, "
            "and zero unsafe false accepts."
        ),
        "requires_prior_structured_protocol_clean": True,
        "requires_prior_constraint_tax_panel_ready": True,
        "deterministic_verifier_final_authority": True,
        "cpu_only_legacy_headline_allowed": False,
    },
    "self_learning": {
        "expectation": (
            "Exp5395 may route verifier budget from the .490 real workflow only with "
            "quality preserved, rollback, stale/poison controls, raw evidence, and no "
            "model-weight mutation; Exp5396 must retain raw episodes."
        ),
        "requires_prior_real_workflow_ready": True,
        "requires_budget_memory_corrigendum_clean": True,
        "no_model_weight_mutation_required": True,
        "raw_episode_retention_required": True,
    },
    "solver": {
        "expectation": (
            "Exp5393 must repair Exp5383 from row-level solver evidence before Exp5394 "
            "uses overwrite or p-bit boundary claims; solver authority and fallback "
            "completeness remain mandatory."
        ),
        "source_flagged_artifact": "results/experiment_5383_overwrite_guidance_scale_validity_v490.json",
        "requires_exp5393_corrigendum_clean_before_exp5394": True,
        "solver_authoritative": True,
        "forced_hint_trust_allowed": False,
    },
    "ARC": {
        "expectation": (
            "Exp5397 can bank a new level only through live agent self-discovery with "
            "registry precheck, no offline BFS, no outer-loop reverse engineering, and "
            "no per-game adapter."
        ),
        "prior_arc_new_level_banked": False,
        "solve_provenance_required": "live_agent_self_discovery",
        "offline_reproduce_required_for_banked_level": True,
    },
    "token": {
        "expectation": (
            "Token/internal-feature energy remains closed unless logits, hidden states, "
            "attention, or intermediate-depth exits have clean backend provenance."
        ),
        "future_token_signal_allowed_from_prior": False,
        "required_backend_features": [
            "logits",
            "hidden_states",
            "attention",
            "intermediate_depth_exits",
        ],
    },
    "hardware": {
        "expectation": (
            "Exp5398 may extend hash-linked receipts and repeatability evidence, but "
            "hardware_speedup_claim stays false unless repeated board-local timing "
            "supports it."
        ),
        "prior_hash_chained_receipt_ready": True,
        "speedup_claim_allowed_from_prior": False,
        "requires_authenticated_repeatability_for_speedup": True,
        "kv260_host_mmcblk_evidence_allowed": False,
    },
}


def value_of(value: Any) -> Any:
    """Return the machine value from older principle-wrapped artifacts."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def path_sha256(path: Path) -> str | None:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, {"exists": True, "loadable": False, "error": "malformed_json"}
    if not isinstance(parsed, dict):
        return {}, {"exists": True, "loadable": False, "error": "not_json_object"}
    return parsed, {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": path_sha256(path),
    }


def read_yaml_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}, {"exists": True, "loadable": False, "error": "malformed_yaml"}
    if not isinstance(parsed, dict):
        return {}, {"exists": True, "loadable": False, "error": "not_yaml_object"}
    return parsed, {
        "exists": True,
        "loadable": True,
        "error": None,
        "sha256": path_sha256(path),
    }


def extract_phase_names(document_text: str) -> list[str]:
    return [
        line.removeprefix("### ").strip()
        for line in document_text.splitlines()
        if line.startswith("### Phase ")
    ]


def extract_task_range(document_text: str) -> str | None:
    match = re.search(r"^\*\*Task range:\*\*\s*(.+)$", document_text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def extract_roadmap_tasks(roadmap: JsonMap) -> list[str]:
    tasks = roadmap.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [str(row["id"]) for row in tasks if isinstance(row, Mapping) and "id" in row]


def empty_prior_gate_summary() -> JsonDict:
    return {field: None for field in PRIOR_GATE_FIELDS}


def extract_prior_gate_summary(capstone: JsonMap) -> JsonDict:
    return {field: value_of(capstone.get(field)) for field in PRIOR_GATE_FIELDS}


def _phase_evidence(capstone: JsonMap, lane: str) -> JsonDict:
    rows = capstone.get("phase_summaries")
    if not isinstance(rows, list):
        rows = capstone.get("phase_outcomes")
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("lane") == lane:
            evidence = row.get("evidence")
            return dict(evidence) if isinstance(evidence, Mapping) else {}
    return {}


def _status_name(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value_of(value.get("status"))
    return value_of(value)


def _mapping_bool(value: Any, key: str) -> bool | None:
    if not isinstance(value, Mapping) or key not in value:
        return None
    return bool(value_of(value.get(key)))


def _critical_tautology_from_rows(rows: Any) -> bool:
    pending = rows if isinstance(rows, list) else []
    return any(
        isinstance(row, Mapping)
        and str(row.get("severity", "")).lower() == "critical"
        and str(row.get("kind", "")).upper() == "TAUTOLOGY"
        for row in pending
    )


def conductor_flagged_exp5383_tautology(conductor_log_text: str) -> bool:
    return any(
        "TAUTOLOGY" in line and ("5383" in line or "overwrite" in line.lower())
        for line in conductor_log_text.splitlines()
    )


def build_prior_blockers(capstone: JsonMap, conductor_log_text: str) -> JsonDict:
    solver = _phase_evidence(capstone, "solver_guidance")
    arc = _phase_evidence(capstone, "arc_geometric_salience")
    hardware = _phase_evidence(capstone, "hardware")
    token = _phase_evidence(capstone, "token_backend")
    kv260 = hardware.get("kv260_status")
    return {
        "solver_tautology": {
            "source": "Exp5383",
            "blocked": True,
            "overwrite_guidance_scale_ready": value_of(
                capstone.get("overwrite_guidance_scale_ready")
            )
            is True,
            "artifact_flagged_adversarial": value_of(solver.get("flagged_adversarial")) is True,
            "critical_tautology_flagged": _critical_tautology_from_rows(
                solver.get("corrigendum_pending")
            ),
            "conductor_flagged_tautology": conductor_flagged_exp5383_tautology(
                conductor_log_text
            ),
            "corrigendum_required": True,
            "downstream_requirement": "recompute_from_row_level_solver_evidence_before_exp5394",
        },
        "ARC": {
            "source": "Exp5385",
            "blocked": value_of(capstone.get("arc_new_level_banked")) is not True,
            "arc_new_level_banked": value_of(capstone.get("arc_new_level_banked")) is True,
            "failure_mode": value_of(arc.get("failure_mode")),
            "live_attempt_count": value_of(arc.get("live_attempt_count")),
            "offline_reproduced": value_of(arc.get("offline_reproduced")) is True,
            "solve_provenance": value_of(arc.get("solve_provenance")),
        },
        "token_feature": {
            "source": "Exp5387",
            "blocked": value_of(capstone.get("future_token_signal_allowed")) is not True,
            "future_token_signal_allowed": value_of(capstone.get("future_token_signal_allowed"))
            is True,
            "backend_reopen_allowed": value_of(token.get("backend_reopen_allowed")) is True,
            "no_live_signal_claim": value_of(token.get("no_live_signal_claim")) is True,
            "logits_available": value_of(token.get("logits_available")) is True,
            "hidden_states_available": value_of(token.get("hidden_states_available")) is True,
            "attention_available": value_of(token.get("attention_available")) is True,
            "intermediate_depth_exits_available": value_of(
                token.get("intermediate_depth_exits_available")
            )
            is True,
        },
        "hardware": {
            "source": "Exp5386",
            "blocked": value_of(capstone.get("hardware_speedup_claim")) is not True,
            "hardware_hash_chained_receipt_ready": value_of(
                capstone.get("hardware_hash_chained_receipt_ready")
            )
            is True,
            "hardware_speedup_claim": value_of(capstone.get("hardware_speedup_claim")) is True,
            "kv260_reachability": _status_name(kv260),
            "kv260_ssh_reachable": _mapping_bool(kv260, "ssh_reachable"),
            "polar_fire_status": _status_name(hardware.get("polar_fire_status")),
            "gatemate_status": _status_name(hardware.get("gatemate_status")),
            "repeatability_evidence_present": value_of(
                hardware.get("repeatability_evidence_present")
            )
            is True,
        },
    }


def git_path_modified(root: Path, relative_path: Path) -> bool:
    if not (root / ".git").exists():
        return False
    result = subprocess.run(
        ("git", "status", "--short", "--", str(relative_path)),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _modification_status(
    root: Path,
    relative_path: Path,
    overrides: Mapping[Path | str, bool] | None,
) -> bool:
    if overrides is None:
        return git_path_modified(root, relative_path)
    if relative_path in overrides:
        return bool(overrides[relative_path])
    return bool(overrides.get(str(relative_path), git_path_modified(root, relative_path)))


def _yaml_milestone(roadmap: JsonMap) -> str | None:
    milestone = roadmap.get("milestone")
    return str(milestone) if milestone is not None else None


def capstone_failures(capstone: JsonMap) -> list[str]:
    if not capstone:
        return ["capstone_missing_or_unloadable"]
    failures: list[str] = []
    capstone_milestone = value_of(capstone.get("milestone"))
    capstone_status = value_of(capstone.get("status"))
    capstone_verdict = value_of(capstone.get("honest_verdict"))
    if capstone_milestone != PRIOR_MILESTONE:
        failures.append(
            f"capstone_milestone_expected_{PRIOR_MILESTONE}_observed_{capstone_milestone}"
        )
    if capstone_status != "complete":
        failures.append(f"capstone_status_expected_complete_observed_{capstone_status}")
    if not isinstance(capstone_verdict, str) or not capstone_verdict.startswith(TERMINAL_PREFIXES):
        failures.append("capstone_honest_verdict_missing_terminal_prefix")
    return failures


def cited_upstream_artifacts(root: Path) -> list[JsonDict]:
    return [
        {
            "relative_path": str(relative_path),
            "exists": (root / relative_path).exists(),
            "sha256": path_sha256(root / relative_path),
            "read_only": True,
        }
        for relative_path in SOURCE_CONTEXT_PATHS
    ]


def precondition_summary(
    *,
    root: Path,
    capstone_meta: JsonMap,
    active_roadmap: JsonMap,
    active_meta: JsonMap,
    next_roadmap: JsonMap,
    next_meta: JsonMap,
    active_roadmap_modified: bool,
    conductor_modified: bool,
) -> JsonDict:
    doc_path = root / VNEXT_RELATIVE_PATH
    doc_text = doc_path.read_text(encoding="utf-8", errors="replace") if doc_path.exists() else ""
    active_milestone = _yaml_milestone(active_roadmap)
    next_milestone = _yaml_milestone(next_roadmap)
    executable_milestone = next_milestone if next_milestone == MILESTONE else active_milestone
    return {
        "capstone_present": capstone_meta.get("exists") is True,
        "capstone_loadable": capstone_meta.get("loadable") is True,
        "roadmap_doc_present": doc_path.exists(),
        "roadmap_doc_names_milestone": MILESTONE in doc_text,
        "active_roadmap_present": active_meta.get("exists") is True,
        "active_roadmap_loadable": active_meta.get("loadable") is True,
        "active_roadmap_milestone": active_milestone,
        "roadmap_next_present": next_meta.get("exists") is True,
        "roadmap_next_loadable": next_meta.get("loadable") is True,
        "roadmap_next_milestone": next_milestone,
        "roadmap_next_names_milestone": next_milestone == MILESTONE,
        "executable_roadmap_milestone": executable_milestone,
        "active_roadmap_modified": active_roadmap_modified,
        "conductor_modified": conductor_modified,
        "phase_names": extract_phase_names(doc_text),
        "task_range": extract_task_range(doc_text),
        "checked_paths": [str(path) for path in SOURCE_CONTEXT_PATHS],
    }


def select_task_source(
    *,
    active_roadmap: JsonMap,
    next_roadmap: JsonMap,
    preconditions: JsonMap,
) -> tuple[Path, list[str]]:
    if preconditions.get("roadmap_next_names_milestone") is True:
        return ROADMAP_NEXT_RELATIVE_PATH, extract_roadmap_tasks(next_roadmap)
    return ROADMAP_RELATIVE_PATH, extract_roadmap_tasks(active_roadmap)


def failed_preconditions(
    *,
    capstone_failure_rows: Sequence[str],
    preconditions: JsonMap,
    planned_task_ids: Sequence[str],
) -> list[str]:
    failures = list(capstone_failure_rows)
    if preconditions.get("roadmap_doc_names_milestone") is not True:
        failures.append(f"roadmap_doc_missing_or_mismatch_{MILESTONE}")
    if preconditions.get("executable_roadmap_milestone") != MILESTONE:
        failures.append(
            f"executable_roadmap_milestone_expected_{MILESTONE}_observed_"
            f"{preconditions.get('executable_roadmap_milestone')}"
        )
    if (
        preconditions.get("roadmap_next_present") is True
        and preconditions.get("roadmap_next_names_milestone") is not True
    ):
        failures.append(
            f"roadmap_next_milestone_expected_{MILESTONE}_observed_"
            f"{preconditions.get('roadmap_next_milestone')}"
        )
    if not planned_task_ids:
        failures.append("planned_task_ids_missing")
    elif list(planned_task_ids) != EXPECTED_TASK_IDS:
        failures.append("planned_task_ids_mismatch")
    if preconditions.get("phase_names") != EXPECTED_PHASE_NAMES:
        failures.append("roadmap_phase_names_missing_or_mismatch")
    if preconditions.get("active_roadmap_modified") is True:
        failures.append("active_roadmap_modified")
    if preconditions.get("conductor_modified") is True:
        failures.append("conductor_modified")
    return failures


def build_honest_verdict(
    *,
    status: str,
    roadmap_next_present: bool,
    planned_task_source: str,
    failures: Sequence[str],
) -> str:
    if status == "complete":
        return (
            "complete: .490 truth table archived and .491 task range recorded from "
            f"{planned_task_source}; roadmap_next_present="
            f"{str(roadmap_next_present).lower()}; solver flagged, ARC no-bank, "
            "token closed, hardware no-speedup; no active roadmap or conductor edit."
        )
    first_failure = failures[0] if failures else "unknown"
    return (
        "blocked_transition_v491: required transition input failed "
        f"({first_failure}); roadmap_next_present={str(roadmap_next_present).lower()}; "
        "no active roadmap or conductor edit."
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260708",
    duration_s: float | None = None,
    modification_status: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    start = time.perf_counter()
    capstone, capstone_meta = read_json_mapping(root / CAPSTONE_RELATIVE_PATH)
    active_roadmap, active_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    next_roadmap, next_meta = read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    active_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_status)
    conductor_dirty = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_status)
    conductor_log_path = root / CONDUCTOR_LOG_RELATIVE_PATH
    conductor_log_text = (
        conductor_log_path.read_text(encoding="utf-8", errors="replace")
        if conductor_log_path.exists()
        else ""
    )
    preconditions = precondition_summary(
        root=root,
        capstone_meta=capstone_meta,
        active_roadmap=active_roadmap,
        active_meta=active_meta,
        next_roadmap=next_roadmap,
        next_meta=next_meta,
        active_roadmap_modified=active_modified,
        conductor_modified=conductor_dirty,
    )
    task_source, task_ids = select_task_source(
        active_roadmap=active_roadmap,
        next_roadmap=next_roadmap,
        preconditions=preconditions,
    )
    failures = failed_preconditions(
        capstone_failure_rows=capstone_failures(capstone),
        preconditions=preconditions,
        planned_task_ids=task_ids,
    )
    status = "complete" if not failures else "blocked"
    roadmap_next_present = bool(preconditions["roadmap_next_present"])
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - start),
            6,
        ),
        "random_seed": RANDOM_SEED,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "status": status,
        "milestone": MILESTONE,
        "prior_milestone": PRIOR_MILESTONE,
        "prior_capstone_path": str(CAPSTONE_RELATIVE_PATH),
        "prior_gate_summary": (
            extract_prior_gate_summary(capstone) if capstone else empty_prior_gate_summary()
        ),
        "prior_blockers": build_prior_blockers(capstone, conductor_log_text),
        "roadmap_next_present": roadmap_next_present,
        "roadmap_doc_present": bool(preconditions["roadmap_doc_present"]),
        "planned_task_count": len(task_ids),
        "planned_task_ids": list(task_ids),
        "planned_phase_names": list(preconditions["phase_names"]),
        "roadmap_doc_task_range": preconditions["task_range"],
        "planned_task_source": str(task_source),
        "downstream_gate_expectations": json.loads(json.dumps(DOWNSTREAM_GATE_EXPECTATIONS)),
        "active_roadmap_modified": active_modified,
        "conductor_modified": conductor_dirty,
        "failed_preconditions": failures,
        "preconditions_checked": preconditions,
        "cited_upstream_artifacts": cited_upstream_artifacts(root),
        "honest_verdict": build_honest_verdict(
            status=status,
            roadmap_next_present=roadmap_next_present,
            planned_task_source=str(task_source),
            failures=failures,
        ),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if payload.get("schema") != SCHEMA:
        raise ValueError("schema mismatch")
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles mismatch")
    status = payload["status"]
    if status not in {"complete", "blocked"}:
        raise ValueError("status must be complete or blocked")
    if payload["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    if payload["prior_milestone"] != PRIOR_MILESTONE:
        raise ValueError("prior_milestone mismatch")
    if payload["prior_capstone_path"] != str(CAPSTONE_RELATIVE_PATH):
        raise ValueError("prior_capstone_path mismatch")
    prior_gate_summary = payload["prior_gate_summary"]
    if not isinstance(prior_gate_summary, Mapping) or set(prior_gate_summary) != set(
        PRIOR_GATE_FIELDS
    ):
        raise ValueError("prior_gate_summary must contain the required .490 gate fields")
    prior_blockers = payload["prior_blockers"]
    if not isinstance(prior_blockers, Mapping) or set(prior_blockers) != set(
        PRIOR_BLOCKER_FIELDS
    ):
        raise ValueError("prior_blockers must contain the required .490 blocker fields")
    for field in ("roadmap_next_present", "roadmap_doc_present"):
        if not isinstance(payload[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    planned_task_ids = payload["planned_task_ids"]
    if not isinstance(planned_task_ids, list) or not all(
        isinstance(task_id, str) for task_id in planned_task_ids
    ):
        raise ValueError("planned_task_ids must be a list of strings")
    if status == "complete" and planned_task_ids != EXPECTED_TASK_IDS:
        raise ValueError("planned_task_ids mismatch")
    if payload["planned_task_count"] != len(planned_task_ids):
        raise ValueError("planned_task_count must match planned_task_ids")
    if status == "complete" and payload["planned_task_count"] != len(EXPECTED_TASK_IDS):
        raise ValueError("planned_task_count mismatch")
    if payload["downstream_gate_expectations"] != DOWNSTREAM_GATE_EXPECTATIONS:
        raise ValueError("downstream_gate_expectations mismatch")
    if payload["active_roadmap_modified"] is not False:
        raise ValueError("active_roadmap_modified must be false")
    if payload["conductor_modified"] is not False:
        raise ValueError("conductor_modified must be false")
    honest_verdict = payload["honest_verdict"]
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    failures = payload["failed_preconditions"]
    if status == "complete" and failures:
        raise ValueError("complete status cannot carry failed preconditions")
    if status == "blocked" and not failures:
        raise ValueError("blocked status must carry failed preconditions")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("reproducibility_checksum mismatch")


def run(
    *,
    root: Path = REPO_ROOT,
    run_date: str = "20260708",
    duration_s: float | None = None,
) -> Path:
    artifact = build_artifact(root=root, run_date=run_date, duration_s=duration_s)
    out_path = root / RESULT_RELATIVE_PATH
    write_json(out_path, artifact)
    return out_path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-date", default="20260708")
    args = parser.parse_args(argv)
    print(run(root=args.root, run_date=args.run_date))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
