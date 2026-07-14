"""Exp5613 transition receipt from milestone .506 into .507.

Spec refs: REQ-REPORT-5613, SCENARIO-REPORT-5613,
SCENARIO-REPORT-5613-DEPENDENCY-MAP,
SCENARIO-REPORT-5613-FIELD-PRINCIPLES.

This module performs no model inference, ARC solving, KAN learning, or sampler
benchmarking. It reads the terminal `.506` evidence, records which facts are
safe dependencies for `.507`, and keeps blocked, retired, and adversarially
flagged scopes visibly bounded. That explicit ledger prevents a later roadmap
or capstone from accidentally treating a useful null result as a promotion.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    extract_roadmap_tasks,
    path_sha256,
    payload_checksum,
    read_yaml_mapping,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5613_transition_v507.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")

EXPERIMENT = "experiment_5613_transition_v507"
EXPERIMENT_ID = "exp5613-transition-v507"
PREVIOUS_MILESTONE = "2026.07.506"
CURRENT_MILESTONE = "2026.07.507"
PREVIOUS_TASK_RANGE = "exp5603-exp5612"
CURRENT_TASK_RANGE = "exp5613-exp5624"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5613
SCHEMA = "carnot.experiment_5613.transition_v507.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-REPORT-5613",
    "SCENARIO-REPORT-5613",
    "SCENARIO-REPORT-5613-DEPENDENCY-MAP",
    "SCENARIO-REPORT-5613-FIELD-PRINCIPLES",
)

EXP5603_TRANSITION_PATH = Path("results/experiment_5603_transition_v506.json")
EXP5604_SOURCE_PATH = Path("results/experiment_5604_v506_source_delta_ingestion.json")
EXP5605_ENVELOPE_PATH = Path("results/experiment_5605_raw_response_evidence_envelope.json")
EXP5606_PANEL_PATH = Path("results/experiment_5606_clean_sota_solve_verify_evidence_panel.json")
EXP5607_EXTENSION_PATH = Path(
    "results/experiment_5607_property_template_exact_residual_extension.json"
)
EXP5608_KAN_PATH = Path("results/experiment_5608_kan_longitudinal_self_learning.json")
EXP5609_ARC_FILTER_PATH = Path("results/experiment_5609_arc_filter_intermediate_invariance_ab.json")
EXP5610_ARC_LEVEL_PATH = Path("results/experiment_5610_arc_live_self_discovery_levelup_v506.json")
EXP5611_CDLS_PATH = Path("results/experiment_5611_cdls_matched_sampler_crossover.json")
EXP5612_CAPSTONE_PATH = Path("results/experiment_5612_v506_capstone_reconciliation.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5603-transition-v506": EXP5603_TRANSITION_PATH,
    "exp5604-v506-source-delta-ingestion": EXP5604_SOURCE_PATH,
    "exp5605-raw-response-evidence-envelope": EXP5605_ENVELOPE_PATH,
    "exp5606-clean-sota-solve-verify-evidence-panel": EXP5606_PANEL_PATH,
    "exp5607-property-template-exact-residual-extension": EXP5607_EXTENSION_PATH,
    "exp5608-kan-longitudinal-self-learning": EXP5608_KAN_PATH,
    "exp5609-arc-filter-intermediate-invariance-ab": EXP5609_ARC_FILTER_PATH,
    "exp5610-arc-live-self-discovery-levelup-v506": EXP5610_ARC_LEVEL_PATH,
    "exp5611-cdls-matched-sampler-crossover": EXP5611_CDLS_PATH,
    "exp5612-v506-capstone-reconciliation": EXP5612_CAPSTONE_PATH,
}
UPSTREAM_ARTIFACT_PATHS = tuple(TASK_ARTIFACT_PATHS.values())

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

EXPECTED_TASK_IDS = [
    "exp5613-transition-v507",
    "exp5614-v507-source-delta-ingestion",
    "exp5615-native-llamacpp-cuda-runtime-certificate",
    "exp5616-exact-nonstationary-constraint-stream",
    "exp5617-kan-critical-task-duration-map",
    "exp5618-predictive-window-kan-self-learning",
    "exp5619-arc-forward-inverse-transition-cycle",
    "exp5620-arc-cycle-guarded-live-update-ab",
    "exp5621-arc-live-self-discovery-levelup-v507",
    "exp5622-cdls-exact-kernel-audit",
    "exp5623-cdls-multiseed-cpu-cuda-crossover",
    "exp5624-v507-capstone-reconciliation",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "One-line annotations for every required headline and gate field.",
    "artifacts_read": "claims trace to files",
    "terminal_findings": "only observed outcomes cross milestones",
    "promoted_substrates": "clean prerequisites are explicit",
    "retired_scopes": "failed chains stay closed",
    "adversarial_flags_preserved": "flagged artifacts are not upgraded",
    "current_task_range": "IDs do not collide",
    "dependency_map": "gates are auditable",
    "inference_substrate": "no new inference occurred",
    "reproducibility_checksum": "the transition is stable",
    "honest_verdict": "terminal status starts complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "status",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "previous_milestone",
    "current_milestone",
    "previous_task_range",
    "current_task_range",
    "current_task_collision_check",
    "artifact_metadata",
    "source_context",
    "source_context_missing",
    "missing_artifacts",
    "malformed_artifacts",
    "roadmap_task_ids",
    "roadmap_task_count",
    "roadmap_doc_task_range",
    "protected_file_checks",
    "preconditions_checked",
    "failed_preconditions",
    "tests_run",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
    *REQUIRED_ARTIFACT_FIELDS,
)
LIST_FIELDS = (
    "artifacts_read",
    "promoted_substrates",
    "retired_scopes",
    "adversarial_flags_preserved",
    "source_context",
    "source_context_missing",
    "missing_artifacts",
    "malformed_artifacts",
    "roadmap_task_ids",
    "protected_file_checks",
    "failed_preconditions",
    "tests_run",
)
DICT_FIELDS = (
    "field_principles",
    "terminal_findings",
    "current_task_collision_check",
    "artifact_metadata",
    "dependency_map",
    "preconditions_checked",
)
BOOL_FIELDS = ("roadmap_yaml_unchanged", "conductor_unchanged")

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5613_transition_v507.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5613_transition_v507.py -m pytest "
            "tests/python/test_experiment_5613_transition_v507.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5613_transition_v507.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
)


def _read_text(root: Path, rel_path: Path) -> str:
    path = root / rel_path
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _task_range_from_text(text: str) -> str | None:
    match = re.search(r"Exp\s*(\d+)\s*[-\u2013]\s*Exp?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"exp{match.group(1)}-exp{match.group(2)}"
    compact = re.search(r"exp(\d+)\s*[-\u2013]\s*exp?(\d+)", text, flags=re.IGNORECASE)
    return f"exp{compact.group(1)}-exp{compact.group(2)}" if compact else None


def _read_json_any(path: Path) -> tuple[JsonDict, JsonDict]:
    metadata: JsonDict = {
        "exists": path.exists(),
        "loadable": False,
        "json_type": None,
        "sha256": path_sha256(path),
    }
    if not path.exists():
        metadata["error"] = "missing"
        return {}, metadata
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        metadata.update({"error": "malformed_json", "line": exc.lineno, "column": exc.colno})
        return {}, metadata
    metadata["json_type"] = type(parsed).__name__
    if not isinstance(parsed, Mapping):
        metadata["error"] = "not_json_object"
        return {}, metadata
    metadata.update({"loadable": True, "error": None})
    return dict(parsed), metadata


def _read_source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        exists = path.exists()
        records.append(
            {
                "path": rel_path.as_posix(),
                "exists": exists,
                "read_only": True,
                "sha256": path_sha256(path),
            }
        )
        if not exists:
            missing.append(rel_path.as_posix())
    return records, missing


def read_artifacts(root: Path) -> tuple[dict[str, JsonDict], JsonDict, list[JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: JsonDict = {}
    found: list[JsonDict] = []
    path_to_task = {path: task_id for task_id, path in TASK_ARTIFACT_PATHS.items()}
    for rel_path in UPSTREAM_ARTIFACT_PATHS:
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        metadata[rel] = meta
        payloads[rel] = payload
        if meta.get("exists") and meta.get("loadable"):
            found.append(
                {
                    "path": rel,
                    "task_id": path_to_task[rel_path],
                    "sha256": meta.get("sha256"),
                    "status": _status_for_payload(payload, meta),
                    "honest_verdict": _verdict(payload) or None,
                }
            )
    return payloads, metadata, found


def _payload(artifacts: Mapping[str, JsonMap], rel_path: Path) -> JsonMap:
    value = artifacts.get(rel_path.as_posix(), {})
    return value if isinstance(value, Mapping) else {}


def _verdict(payload: JsonMap) -> str:
    verdict = payload.get("honest_verdict")
    return str(verdict) if verdict is not None else ""


def _is_gate_skip(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    blocked_at_layer = str(payload.get("blocked_at_layer") or "").lower()
    return bool(
        payload.get("schema") == "blocked_gate_check_v1"
        or verdict == "blocked_gate_check_failed"
        or ("gate" in blocked_at_layer and str(payload.get("status") or "").lower() == "blocked")
        or (payload.get("gate_check_summary") and str(payload.get("status") or "").lower() == "blocked")
    )


def _is_blocked(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    status = str(payload.get("status") or "").lower()
    return bool(status == "blocked" or verdict.startswith("blocked:") or verdict.startswith("blocked_"))


def _is_complete(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    status = str(payload.get("status") or "").lower()
    return bool(status == "complete" or verdict.startswith("complete:"))


def _status_for_payload(payload: JsonMap, meta: JsonMap) -> str:
    if not meta.get("exists"):
        return "missing"
    if not meta.get("loadable"):
        return "malformed"
    if payload.get("flagged_adversarial"):
        return "flagged"
    if _is_gate_skip(payload):
        return "gate_skipped"
    if _is_blocked(payload):
        return "blocked"
    if _is_complete(payload):
        return "complete"
    return "unknown"


def _int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


def _float(payload: JsonMap, field: str) -> float:
    value = payload.get(field)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _all_true(value: Any) -> bool:
    return bool(isinstance(value, Mapping) and value and all(item is True for item in value.values()))


def _extract_exp_number(value: str) -> int | None:
    match = re.search(r"(?:exp|experiment_)(\d+)", value)
    return int(match.group(1)) if match else None


def _completed_task_ids(root: Path) -> list[str]:
    text = _read_text(root, RESEARCH_COMPLETE_RELATIVE_PATH)
    return sorted(set(re.findall(r"\bid:\s*(exp\d+(?:[-_][A-Za-z0-9_]+)*)", text)))


def _collision_check(root: Path, roadmap_task_ids: Sequence[str]) -> JsonDict:
    completed = _completed_task_ids(root)
    current_numbers = set(range(5613, 5625))
    colliding = [
        task_id
        for task_id in completed
        if (number := _extract_exp_number(task_id)) in current_numbers
    ]
    highest = max((_extract_exp_number(task_id) or 0 for task_id in completed), default=0)
    return {
        "current_task_range": CURRENT_TASK_RANGE,
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "roadmap_task_ids": list(roadmap_task_ids),
        "completed_ids_checked_count": len(completed),
        "highest_completed_id": f"exp{highest}" if highest else None,
        "colliding_ids": colliding,
        "collision_free": not colliding,
        "range_matches_roadmap": list(roadmap_task_ids) == list(EXPECTED_TASK_IDS),
    }


def terminal_findings(artifacts: Mapping[str, JsonMap], metadata: JsonMap) -> dict[str, JsonDict]:
    findings: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        rel = rel_path.as_posix()
        payload = _payload(artifacts, rel_path)
        meta = metadata.get(rel, {})
        findings[task_id] = {
            "artifact_path": rel,
            "status": _status_for_payload(payload, meta),
            "honest_verdict": _verdict(payload) or None,
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "corrigendum_pending": payload.get("corrigendum_pending", []),
            "sha256": meta.get("sha256"),
            "metadata_error": meta.get("error"),
            "supports_positive_claim": _status_for_payload(payload, meta) == "complete",
        }
    return findings


def _response_envelope_clean(exp5605: JsonMap, status: str) -> bool:
    return bool(
        status == "complete"
        and exp5605.get("envelope_ready")
        and exp5605.get("raw_payloads_preserved")
        and _float(exp5605, "lossless_replay_rate") >= 1.0
        and _int(exp5605, "semantic_false_accept_count") == 0
        and exp5605.get("parser_version_replay_passed")
        and exp5605.get("payload_corruption_rejected")
    )


def _kan_clean(exp5608: JsonMap, status: str) -> bool:
    return bool(
        status == "complete"
        and exp5608.get("continuous_self_learning_task")
        and exp5608.get("kan_longitudinal_ready")
        and _all_true(exp5608.get("promotion_gate"))
        and _int(exp5608, "unsafe_false_accept_count") == 0
        and exp5608.get("rollback_positive_control")
        and exp5608.get("delayed_regression_passed")
        and exp5608.get("no_model_weight_mutation")
        and _int(exp5608, "llm_calls") == 0
        and not exp5608.get("llm_weight_training")
    )


def promoted_substrates(
    artifacts: Mapping[str, JsonMap],
    findings: Mapping[str, JsonMap],
) -> list[JsonDict]:
    exp5605 = _payload(artifacts, EXP5605_ENVELOPE_PATH)
    exp5608 = _payload(artifacts, EXP5608_KAN_PATH)
    rows: list[JsonDict] = []
    if _response_envelope_clean(
        exp5605,
        str(findings["exp5605-raw-response-evidence-envelope"]["status"]),
    ):
        rows.append(
            {
                "key": "lossless_response_envelope",
                "source_artifacts": [EXP5605_ENVELOPE_PATH.as_posix()],
                "prerequisite_for": ["exp5615-native-llamacpp-cuda-runtime-certificate"],
                "evidence": {
                    "response_rows_written": exp5605.get("response_rows_written"),
                    "lossless_replay_rate": exp5605.get("lossless_replay_rate"),
                    "semantic_false_accept_count": exp5605.get("semantic_false_accept_count"),
                    "truncation_controls_detected": exp5605.get("truncation_controls_detected"),
                },
                "claim_boundary": "runtime evidence transport only; no solve-versus-verify accuracy claim",
            }
        )
    if _kan_clean(exp5608, str(findings["exp5608-kan-longitudinal-self-learning"]["status"])):
        rows.append(
            {
                "key": "active_spline_kan",
                "source_artifacts": [EXP5608_KAN_PATH.as_posix()],
                "prerequisite_for": [
                    "exp5617-kan-critical-task-duration-map",
                    "exp5618-predictive-window-kan-self-learning",
                ],
                "evidence": {
                    "promotion_gate": exp5608.get("promotion_gate", {}),
                    "forward_transfer_delta": exp5608.get("forward_transfer_delta"),
                    "backward_retention_delta": exp5608.get("backward_retention_delta"),
                    "forgetting_delta": exp5608.get("forgetting_delta"),
                    "kan_weights_mutated": bool(exp5608.get("kan_weights_mutated")),
                    "poison_update_disposition": exp5608.get("poison_update_disposition", {}),
                },
                "claim_boundary": "bounded KAN-component adaptation only; no LLM weight mutation",
            }
        )
    return rows


def retired_scopes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    exp5606 = _payload(artifacts, EXP5606_PANEL_PATH)
    exp5607 = _payload(artifacts, EXP5607_EXTENSION_PATH)
    exp5609 = _payload(artifacts, EXP5609_ARC_FILTER_PATH)
    exp5611 = _payload(artifacts, EXP5611_CDLS_PATH)
    decisions = exp5609.get("filter_promotion_decisions")
    inert = decisions.get("inert_click", {}) if isinstance(decisions, Mapping) else {}
    history = decisions.get("object_history", {}) if isinstance(decisions, Mapping) else {}
    return [
        {
            "key": "solve_versus_verify_panel",
            "closed": True,
            "source_artifacts": [EXP5606_PANEL_PATH.as_posix()],
            "reason": "CUDA offload was unauthenticated and parser failure stayed total.",
            "evidence": {
                "panel_complete": bool(exp5606.get("panel_complete")),
                "gpu_offload_authenticated": bool(exp5606.get("gpu_offload_authenticated")),
                "maximum_parser_failure_rate": exp5606.get("maximum_parser_failure_rate"),
                "solve_verify_asymmetry_supported": bool(
                    exp5606.get("solve_verify_asymmetry_supported")
                ),
            },
        },
        {
            "key": "exact_residual_extension_chain",
            "closed": True,
            "source_artifacts": [EXP5607_EXTENSION_PATH.as_posix()],
            "reason": "The gate skipped because Exp5606 supplied no clean residual panel.",
            "evidence": {"gate_check_summary": exp5607.get("gate_check_summary")},
        },
        {
            "key": "arc_inert_click_filter",
            "closed": True,
            "source_artifacts": [EXP5609_ARC_FILTER_PATH.as_posix()],
            "reason": str(inert.get("reason") or inert.get("decision") or "no downstream improvement"),
            "evidence": {
                "decision": inert.get("decision"),
                "reachable": bool(inert.get("reachable")),
                "downstream_improved": bool(inert.get("downstream_improved")),
            },
        },
        {
            "key": "arc_object_history_filter",
            "closed": True,
            "source_artifacts": [EXP5609_ARC_FILTER_PATH.as_posix()],
            "reason": str(history.get("reason") or history.get("decision") or "no downstream improvement"),
            "evidence": {
                "decision": history.get("decision"),
                "reachable": bool(history.get("reachable")),
                "downstream_improved": bool(history.get("downstream_improved")),
            },
        },
        {
            "key": "unmatched_cdls_crossover",
            "closed": True,
            "source_artifacts": [EXP5611_CDLS_PATH.as_posix()],
            "reason": "One seed and zero quality-matched pairs cannot support a crossover claim.",
            "evidence": {
                "seeds": exp5611.get("seeds", []),
                "successful_matched_pairs": _int(exp5611, "successful_matched_pairs"),
                "crossover_claim_allowed": bool(exp5611.get("crossover_claim_allowed")),
                "board_speedup_claimed": bool(exp5611.get("board_speedup_claimed")),
            },
            "forward_boundary": "exact kernel audit may proceed; this unmatched timing claim stays closed",
        },
    ]


def adversarial_flags_preserved(findings: Mapping[str, JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task_id, row in findings.items():
        if not row.get("flagged_adversarial"):
            continue
        pending = row.get("corrigendum_pending")
        flags = pending if isinstance(pending, list) else []
        rows.append(
            {
                "task_id": task_id,
                "artifact_path": row.get("artifact_path"),
                "status": row.get("status"),
                "flag_kinds": [str(flag.get("kind")) for flag in flags if isinstance(flag, Mapping)],
                "upgraded_to_clean": False,
                "claim_boundary": "flagged evidence may be cited as a boundary, not as a clean substrate",
            }
        )
    return rows


def dependency_map() -> JsonDict:
    return {
        "native_runtime_certification": {
            "chain": "native-runtime certification",
            "tasks": ["exp5615-native-llamacpp-cuda-runtime-certificate"],
            "upstream_clean_substrate": "lossless_response_envelope",
            "closed_scope_not_reopened": "solve_versus_verify_panel",
            "gates": [
                {"upstream": "exp5605-raw-response-evidence-envelope", "field": "envelope_ready", "op": "==", "value": True},
                {"upstream": "exp5615-native-llamacpp-cuda-runtime-certificate", "field": "runtime_certificate_ready_score", "op": "==", "value": 1.0},
            ],
        },
        "kan_drift_to_predictive_controller": {
            "chain": "exact drift fixture->duration map->predictive KAN",
            "tasks": [
                "exp5616-exact-nonstationary-constraint-stream",
                "exp5617-kan-critical-task-duration-map",
                "exp5618-predictive-window-kan-self-learning",
            ],
            "upstream_clean_substrate": "active_spline_kan",
            "gates": [
                {"upstream": "exp5616-exact-nonstationary-constraint-stream", "field": "fixture_ready_score", "op": "==", "value": 1.0},
                {"upstream": "exp5617-kan-critical-task-duration-map", "field": "nondegenerate_switch_cases", "op": ">", "value": 1},
                {"upstream": "exp5618-predictive-window-kan-self-learning", "field": "continuous_self_learning_ready", "op": "==", "value": True},
            ],
        },
        "arc_transition_cycle_to_level_attempt": {
            "chain": "ARC transition-cycle prototype->live A/B->unconditional level attempt",
            "tasks": [
                "exp5619-arc-forward-inverse-transition-cycle",
                "exp5620-arc-cycle-guarded-live-update-ab",
                "exp5621-arc-live-self-discovery-levelup-v507",
            ],
            "closed_scopes_not_reopened": ["arc_inert_click_filter", "arc_object_history_filter"],
            "gates": [
                {"upstream": "exp5619-arc-forward-inverse-transition-cycle", "field": "cycle_verifier_positive_control_rate", "op": ">=", "value": "preregistered_threshold"},
                {"upstream": "exp5620-arc-cycle-guarded-live-update-ab", "field": "live_branch_promotion_score", "op": "==", "value": 1.0, "advisory_only": True},
                {"upstream": "exp5621-arc-live-self-discovery-levelup-v507", "field": "live_attempt_executed", "op": "==", "value": True, "unconditional": True},
            ],
        },
        "cdls_exactness_to_crossover": {
            "chain": "cDLS exactness->gated crossover",
            "tasks": [
                "exp5622-cdls-exact-kernel-audit",
                "exp5623-cdls-multiseed-cpu-cuda-crossover",
            ],
            "closed_scope_not_reopened": "unmatched_cdls_crossover",
            "gates": [
                {"upstream": "exp5622-cdls-exact-kernel-audit", "field": "kernel_audit_ready_score", "op": "==", "value": 1.0},
                {"upstream": "exp5623-cdls-multiseed-cpu-cuda-crossover", "field": "crossover_claim_allowed", "op": "==", "value": True},
            ],
        },
        "capstone_reconciliation": {
            "chain": "Exp5613-Exp5623->Exp5624 reconciliation",
            "tasks": [*EXPECTED_TASK_IDS[:-1], "exp5624-v507-capstone-reconciliation"],
            "gate": "Exp5624 cannot upgrade blocked, skipped, flagged, development-proxy, or unmatched evidence.",
        },
    }


def _protected_file_checks(
    root: Path,
    *,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[JsonDict]:
    return [
        {
            "path": ROADMAP_RELATIVE_PATH.as_posix(),
            "exists": (root / ROADMAP_RELATIVE_PATH).exists(),
            "git_status_clean": not roadmap_modified,
            "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
        },
        {
            "path": CONDUCTOR_RELATIVE_PATH.as_posix(),
            "exists": (root / CONDUCTOR_RELATIVE_PATH).exists(),
            "git_status_clean": not conductor_modified,
            "sha256": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
        },
    ]


def _failed_preconditions(
    missing_artifacts: Sequence[str],
    malformed_artifacts: Sequence[str],
    collision: JsonMap,
    *,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failures = [f"missing_artifact:{path}" for path in missing_artifacts]
    failures.extend(f"malformed_artifact:{path}" for path in malformed_artifacts)
    failures.extend(f"current_task_id_collision:{task_id}" for task_id in collision.get("colliding_ids", []))
    if not collision.get("range_matches_roadmap"):
        failures.append("roadmap_task_range_mismatch")
    if roadmap_modified:
        failures.append("research-roadmap.yaml_modified")
    if conductor_modified:
        failures.append("scripts/research_conductor.py_modified")
    return failures


def _honest_verdict(status: str, failures: Sequence[str]) -> str:
    if status == "complete":
        return (
            "complete: archived .506 terminal evidence into .507 dependency map; "
            "current_task_range=exp5613-exp5624; response_envelope_promoted=True; "
            "active_spline_kan_promoted=True; solve_verify_panel_closed=True; "
            "exact_residual_extension_closed=True; arc_filters_retired=True; "
            "cdls_unmatched_crossover_closed=True."
        )
    first = failures[0] if failures else "unknown"
    return f"blocked: .507 transition receipt failed precondition {first}."


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, metadata, artifacts_read = read_artifacts(root)
    source_context, source_context_missing = _read_source_context(root)
    roadmap, _roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    roadmap_doc_task_range = _task_range_from_text(_read_text(root, VNEXT_RELATIVE_PATH))
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    missing_artifacts = [
        path.as_posix()
        for path in UPSTREAM_ARTIFACT_PATHS
        if not metadata[path.as_posix()].get("exists")
    ]
    malformed_artifacts = [
        path.as_posix()
        for path in UPSTREAM_ARTIFACT_PATHS
        if metadata[path.as_posix()].get("exists") and not metadata[path.as_posix()].get("loadable")
    ]
    findings = terminal_findings(artifacts, metadata)
    collision = _collision_check(root, roadmap_task_ids)
    failures = _failed_preconditions(
        missing_artifacts,
        malformed_artifacts,
        collision,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )
    status = "complete" if not failures else "blocked"
    tests = [dict(row) if isinstance(row, Mapping) else {"command": str(row)} for row in tests_run]
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "previous_milestone": PREVIOUS_MILESTONE,
        "current_milestone": CURRENT_MILESTONE,
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "current_task_range": CURRENT_TASK_RANGE,
        "current_task_collision_check": collision,
        "artifact_metadata": metadata,
        "source_context": source_context,
        "source_context_missing": source_context_missing,
        "missing_artifacts": missing_artifacts,
        "malformed_artifacts": malformed_artifacts,
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_task_count": len(roadmap_task_ids),
        "roadmap_doc_task_range": roadmap_doc_task_range,
        "protected_file_checks": _protected_file_checks(
            root,
            roadmap_modified=roadmap_modified,
            conductor_modified=conductor_modified,
        ),
        "preconditions_checked": {
            "upstream_artifacts_expected": len(UPSTREAM_ARTIFACT_PATHS),
            "upstream_artifacts_read": len(artifacts_read),
            "roadmap_task_count": len(roadmap_task_ids),
            "roadmap_doc_task_range": roadmap_doc_task_range,
            "research_complete_checked": (root / RESEARCH_COMPLETE_RELATIVE_PATH).exists(),
            "collision_free": bool(collision.get("collision_free")),
            "roadmap_yaml_unchanged": not roadmap_modified,
            "conductor_unchanged": not conductor_modified,
        },
        "failed_preconditions": failures,
        "tests_run": tests,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "field_principles": dict(FIELD_PRINCIPLES),
        "artifacts_read": artifacts_read,
        "terminal_findings": findings,
        "promoted_substrates": promoted_substrates(artifacts, findings),
        "retired_scopes": retired_scopes(artifacts),
        "adversarial_flags_preserved": adversarial_flags_preserved(findings),
        "dependency_map": dependency_map(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(status, failures),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(field)
    for field in LIST_FIELDS:
        if field in payload and not isinstance(payload[field], list):
            errors.append(field)
    for field in DICT_FIELDS:
        if field in payload and not isinstance(payload[field], Mapping):
            errors.append(field)
    for field in BOOL_FIELDS:
        if field in payload and not isinstance(payload[field], bool):
            errors.append(field)
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    terminal = payload.get("terminal_findings")
    if isinstance(terminal, Mapping):
        if not set(TASK_ARTIFACT_PATHS).issubset(terminal):
            errors.append("terminal_findings")
    promoted = payload.get("promoted_substrates")
    if payload.get("status") == "complete" and isinstance(promoted, list):
        keys = {str(row.get("key")) for row in promoted if isinstance(row, Mapping)}
        if not {"lossless_response_envelope", "active_spline_kan"} <= keys:
            errors.append("promoted_substrates")
    retired = payload.get("retired_scopes")
    if isinstance(retired, list):
        keys = {str(row.get("key")) for row in retired if isinstance(row, Mapping)}
        if not {
            "solve_versus_verify_panel",
            "exact_residual_extension_chain",
            "arc_inert_click_filter",
            "arc_object_history_filter",
            "unmatched_cdls_crossover",
        } <= keys:
            errors.append("retired_scopes")
    flags = payload.get("adversarial_flags_preserved")
    if payload.get("status") == "complete" and isinstance(flags, list):
        ids = {str(row.get("task_id")) for row in flags if isinstance(row, Mapping)}
        if not {
            "exp5604-v506-source-delta-ingestion",
            "exp5610-arc-live-self-discovery-levelup-v506",
            "exp5611-cdls-matched-sampler-crossover",
        } <= ids:
            errors.append("adversarial_flags_preserved")
    if payload.get("current_task_range") != CURRENT_TASK_RANGE:
        errors.append("current_task_range")
    collision = payload.get("current_task_collision_check")
    if (
        payload.get("status") == "complete"
        and isinstance(collision, Mapping)
        and collision.get("collision_free") is not True
    ):
        errors.append("current_task_collision_check")
    dependency = payload.get("dependency_map")
    if not isinstance(dependency, Mapping) or not {
        "native_runtime_certification",
        "kan_drift_to_predictive_controller",
        "arc_transition_cycle_to_level_attempt",
        "cdls_exactness_to_crossover",
    } <= set(dependency):
        errors.append("dependency_map")
    if payload.get("roadmap_yaml_unchanged") is not True:
        errors.append("roadmap_yaml_unchanged")
    if payload.get("conductor_unchanged") is not True:
        errors.append("conductor_unchanged")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    checksum = payload.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum")
    honest_verdict = payload.get("honest_verdict")
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    return sorted(set(errors))


def write_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = build_report(
        root=root,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - validation failure paths are covered without raising.
        raise ValueError(f"invalid Exp5613 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5613 artifact")
    args = parser.parse_args(argv)
    artifact = write_report() if args.write else build_report()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
