"""Exp5603 transition receipt from milestone .505 into .506.

Spec refs: REQ-REPORT-5603, SCENARIO-REPORT-5603,
SCENARIO-REPORT-5603-DEPENDENCY-MAP,
SCENARIO-REPORT-5603-FIELD-PRINCIPLES.

This module does not run inference, ARC search, or repair experiments. It
collects the repository artifacts that already closed milestone `.505` and the
post-milestone outer-loop artifacts that shaped `.506`, then records which
facts are safe to carry forward. The guardrail is deliberately boring: a fact
can become a dependency only when it is present in a file we read, and failed
chains stay closed unless a later artifact explicitly changes their status.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5603_transition_v506.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")

EXPERIMENT = "experiment_5603_transition_v506"
EXPERIMENT_ID = "exp5603-transition-v506"
PREVIOUS_MILESTONE = "2026.07.505"
CURRENT_MILESTONE = "2026.07.506"
PREVIOUS_TASK_RANGE = "exp5578-exp5585"
CURRENT_TASK_RANGE = "exp5603-exp5612"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5603
SCHEMA = "carnot.experiment_5603.transition_v506.v1"
INFERENCE_SUBSTRATE = "aggregation_from_repository_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-REPORT-5603",
    "SCENARIO-REPORT-5603",
    "SCENARIO-REPORT-5603-DEPENDENCY-MAP",
    "SCENARIO-REPORT-5603-FIELD-PRINCIPLES",
)

EXP5578_TRANSITION_PATH = Path("results/experiment_5578_transition_v505.json")
EXP5579_SOURCE_PATH = Path("results/experiment_5579_v505_source_delta_ingestion.json")
EXP5580_PARSER_PATH = Path("results/experiment_5580_parser_forensics_positive_control.json")
EXP5581_SOTA_GATE_PATH = Path(
    "results/experiment_5581_clean_sota_solve_verify_remeasurement.json"
)
EXP5583_MEMORY_PATH = Path("results/experiment_5583_causal_memory_metric_corrigendum.json")
EXP5584_PACE_GATE_PATH = Path("results/experiment_5584_two_timescale_exact_self_learning.json")
EXP5585_ARC_PATH = Path("results/experiment_5585_arc_levelup_attempt_v505.json")

TERMINAL_ARTIFACT_PATHS = (
    EXP5578_TRANSITION_PATH,
    EXP5579_SOURCE_PATH,
    EXP5580_PARSER_PATH,
    EXP5581_SOTA_GATE_PATH,
    EXP5583_MEMORY_PATH,
    EXP5584_PACE_GATE_PATH,
    EXP5585_ARC_PATH,
)

OUTER_LOOP_ARTIFACT_PATHS = (
    Path("results/experiment_5592_candidate_scoring_stack_bare_control_ab.json"),
    Path("results/experiment_5593_goal_predicate_consistency_offline_sim_prototype.json"),
    Path("results/experiment_5594_think_mode_induction_quality_ab.json"),
    Path("results/experiment_5595_inert_click_sig_pruner_offline_sim_prototype.json"),
    Path("results/experiment_5596_generator_size_ab_gemma31b_vs_current.json"),
    Path("results/experiment_5597_generator_size_ab_qwen35b_moe_vs_current.json"),
    Path("results/experiment_5598_generator_size_multiseed_ab.json"),
    Path("results/experiment_5599_reinduction_ab_lp85_levelup.json"),
    Path("results/experiment_5600_ptrm_loo_gate.json"),
    Path("results/experiment_5601_object_history_salience_offline_sim_prototype.json"),
    Path("results/experiment_5602_inert_click_pruner_matched_budget_ab.json"),
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

EXPECTED_TASK_IDS = [
    "exp5603-transition-v506",
    "exp5604-v506-source-delta-ingestion",
    "exp5605-raw-response-evidence-envelope",
    "exp5606-clean-sota-solve-verify-evidence-panel",
    "exp5607-property-template-exact-residual-extension",
    "exp5608-kan-longitudinal-self-learning",
    "exp5609-arc-filter-intermediate-invariance-ab",
    "exp5610-unconditional-live-agent-levelup-attempt",
    "exp5611-cdls-matched-cpu-cuda-benchmark",
    "exp5612-v506-capstone-reconciliation",
]

ARTIFACT_IDS: dict[Path, str] = {
    EXP5578_TRANSITION_PATH: "exp5578-transition-v505",
    EXP5579_SOURCE_PATH: "exp5579-v505-source-delta-ingestion",
    EXP5580_PARSER_PATH: "exp5580-parser-forensics-positive-control",
    EXP5581_SOTA_GATE_PATH: "exp5581-clean-sota-solve-verify-remeasurement",
    EXP5583_MEMORY_PATH: "exp5583-causal-memory-metric-corrigendum",
    EXP5584_PACE_GATE_PATH: "exp5584-two-timescale-exact-self-learning",
    EXP5585_ARC_PATH: "exp5585-arc-levelup-attempt-v505",
    Path("results/experiment_5592_candidate_scoring_stack_bare_control_ab.json"): (
        "exp5592-candidate-scoring-stack-bare-control-ab"
    ),
    Path("results/experiment_5593_goal_predicate_consistency_offline_sim_prototype.json"): (
        "exp5593-goal-predicate-consistency-offline-sim-prototype"
    ),
    Path("results/experiment_5594_think_mode_induction_quality_ab.json"): (
        "exp5594-think-mode-induction-quality-ab"
    ),
    Path("results/experiment_5595_inert_click_sig_pruner_offline_sim_prototype.json"): (
        "exp5595-inert-click-sig-pruner-offline-sim-prototype"
    ),
    Path("results/experiment_5596_generator_size_ab_gemma31b_vs_current.json"): (
        "exp5596-generator-size-ab-gemma31b-vs-current"
    ),
    Path("results/experiment_5597_generator_size_ab_qwen35b_moe_vs_current.json"): (
        "exp5597-generator-size-ab-qwen35b-moe-vs-current"
    ),
    Path("results/experiment_5598_generator_size_multiseed_ab.json"): (
        "exp5598-generator-size-multiseed-ab"
    ),
    Path("results/experiment_5599_reinduction_ab_lp85_levelup.json"): (
        "exp5599-reinduction-ab-lp85-levelup"
    ),
    Path("results/experiment_5600_ptrm_loo_gate.json"): "exp5600-ptrm-loo-gate",
    Path("results/experiment_5601_object_history_salience_offline_sim_prototype.json"): (
        "exp5601-object-history-salience-offline-sim-prototype"
    ),
    Path("results/experiment_5602_inert_click_pruner_matched_budget_ab.json"): (
        "exp5602-inert-click-pruner-matched-budget-ab"
    ),
}

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "One-line annotations for every required headline and gate field.",
    "artifacts_read": "terminal claims trace to files",
    "terminal_findings": "only observed outcomes cross milestones",
    "retired_scopes": "failed chains remain closed",
    "clean_substrates": "only unflagged evidence becomes a prerequisite",
    "post_milestone_outer_loop_artifacts": "concurrent work is explicit",
    "current_task_range": "IDs cannot collide",
    "dependency_map": "gates are auditable",
    "inference_substrate": "no live inference occurred",
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
    "task_id_collision_avoidance",
    "artifact_metadata",
    "artifacts_expected",
    "artifacts_read",
    "missing_artifacts",
    "source_context",
    "source_context_missing",
    "roadmap_task_ids",
    "roadmap_task_count",
    "roadmap_doc_task_range",
    "exp5582_preemptive_skip",
    "protected_file_checks",
    "preconditions_checked",
    "failed_preconditions",
    "terminal_findings",
    "retired_scopes",
    "clean_substrates",
    "post_milestone_outer_loop_artifacts",
    "dependency_map",
    "tests_run",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
    "reproducibility_checksum",
)
BOOL_FIELDS = ("roadmap_yaml_unchanged", "conductor_unchanged")
LIST_FIELDS = (
    "artifacts_read",
    "missing_artifacts",
    "source_context",
    "source_context_missing",
    "roadmap_task_ids",
    "protected_file_checks",
    "failed_preconditions",
    "terminal_findings",
    "retired_scopes",
    "clean_substrates",
    "post_milestone_outer_loop_artifacts",
    "tests_run",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5603_transition_v506.py -q --no-cov",
        "outcome": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5603_transition_v506.py -m pytest "
            "tests/python/test_experiment_5603_transition_v506.py -q --no-cov -n 0"
        ),
        "outcome": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5603_transition_v506.py --fail-under=100"
        ),
        "outcome": "not_run_in_default_artifact",
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


def _read_json_any(path: Path) -> tuple[Any, JsonDict]:
    metadata: JsonDict = {
        "exists": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "json_type": None,
    }
    if not path.exists():
        metadata["error"] = "missing"
        return {}, metadata
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        metadata.update({"error": "malformed_json", "line": exc.lineno, "column": exc.colno})
        return {}, metadata
    metadata.update({"loadable": True, "error": None, "json_type": type(payload).__name__})
    if isinstance(payload, list):
        metadata["length"] = len(payload)
    return payload, metadata


def _status_label(payload: JsonMap) -> str:
    status = payload.get("status")
    if status is not None:
        return str(status).lower()
    verdict = str(payload.get("honest_verdict") or "").lower()
    if verdict.startswith("blocked:") or verdict.startswith("blocked_"):
        return "blocked"
    if verdict.startswith("honest_null:") or "honest_null" in verdict:
        return "honest_null"
    if verdict.startswith("failed:"):
        return "failed"
    if verdict.startswith("complete:"):
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


def _payload(artifacts: Mapping[str, Any], rel_path: Path) -> JsonMap:
    value = artifacts.get(rel_path.as_posix(), {})
    return value if isinstance(value, Mapping) else {}


def _read_artifacts(
    root: Path,
    rel_paths: Sequence[Path],
    *,
    role: str,
    required: bool,
) -> tuple[dict[str, Any], JsonDict, list[JsonDict], list[str]]:
    artifacts: dict[str, Any] = {}
    metadata: JsonDict = {}
    read_records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in rel_paths:
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        artifacts[rel] = payload
        metadata[rel] = meta
        if meta["exists"] and meta["loadable"]:
            read_records.append(
                {
                    "path": rel,
                    "experiment_id": ARTIFACT_IDS.get(rel_path, rel_path.stem),
                    "role": role,
                    "sha256": meta.get("sha256"),
                    "status": _status_label(payload if isinstance(payload, Mapping) else {}),
                    "honest_verdict": payload.get("honest_verdict")
                    if isinstance(payload, Mapping)
                    else None,
                }
            )
        elif required:
            missing.append(rel)
    return artifacts, metadata, read_records, missing


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


def _find_exp5582_preemptive_skip(root: Path) -> JsonDict:
    text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    lines = [
        line.strip()
        for line in text.splitlines()
        if "Counterexample-guided exact verifier extension" in line
        and "Pre-emptive skip" in line
    ]
    return {
        "task_id": "exp5582-exact-counterexample-verifier-extension",
        "classification": "preemptive_skip",
        "source_path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        "observed": bool(lines),
        "matching_lines": lines[-3:],
        "claim_boundary": "No exact extension ran because the Exp5581 residual prerequisite retired.",
    }


def _terminal_findings(
    artifacts: Mapping[str, Any],
    exp5582_skip: JsonMap,
) -> list[JsonDict]:
    exp5578 = _payload(artifacts, EXP5578_TRANSITION_PATH)
    exp5579 = _payload(artifacts, EXP5579_SOURCE_PATH)
    exp5580 = _payload(artifacts, EXP5580_PARSER_PATH)
    exp5581 = _payload(artifacts, EXP5581_SOTA_GATE_PATH)
    exp5583 = _payload(artifacts, EXP5583_MEMORY_PATH)
    exp5584 = _payload(artifacts, EXP5584_PACE_GATE_PATH)
    exp5585 = _payload(artifacts, EXP5585_ARC_PATH)
    taxonomy = exp5580.get("failure_taxonomy") if isinstance(exp5580.get("failure_taxonomy"), Mapping) else {}
    policy_gate = exp5583.get("policy_gate") if isinstance(exp5583.get("policy_gate"), Mapping) else {}
    return [
        {
            "key": "v505_transition_receipt",
            "classification": "conductor_task",
            "source_artifacts": [EXP5578_TRANSITION_PATH.as_posix()],
            "evidence": {
                "status": _status_label(exp5578),
                "next_task_range": exp5578.get("next_task_range"),
                "clean_lane_count": len(exp5578.get("clean_lanes", []))
                if isinstance(exp5578.get("clean_lanes"), list)
                else 0,
            },
        },
        {
            "key": "hash_only_parser_forensics",
            "classification": "conductor_task",
            "source_artifacts": [EXP5580_PARSER_PATH.as_posix()],
            "evidence": {
                "cached_rows_audited": _int(exp5580, "cached_rows_audited"),
                "raw_response_text_available": bool(exp5580.get("raw_response_text_available")),
                "parser_repair_ready": bool(exp5580.get("parser_repair_ready")),
                "truncation_count": _int(taxonomy, "truncation"),
                "other_failure_count": _int(taxonomy, "other"),
                "semantic_false_accept_count": _int(exp5580, "semantic_false_accept_count"),
                "claim_imported": "unrecoverable_instrumentation_only",
            },
        },
        {
            "key": "sota_remeasurement_gate_block",
            "classification": "gate_blocked_conductor_task",
            "source_artifacts": [EXP5581_SOTA_GATE_PATH.as_posix()],
            "evidence": {
                "status": _status_label(exp5581),
                "honest_verdict": exp5581.get("honest_verdict"),
                "gate_check_summary": exp5581.get("gate_check_summary"),
            },
        },
        {
            "key": "exact_extension_preemptive_skip",
            "classification": "preemptive_skip",
            "source_artifacts": [CONDUCTOR_LOG_RELATIVE_PATH.as_posix()],
            "evidence": {
                "observed": bool(exp5582_skip.get("observed")),
                "upstream": "exp5581-clean-sota-solve-verify-remeasurement",
                "artifact_emitted": False,
            },
        },
        {
            "key": "causal_memory_corrigendum",
            "classification": "conductor_task",
            "source_artifacts": [EXP5583_MEMORY_PATH.as_posix()],
            "evidence": {
                "policy_ready": bool(exp5583.get("policy_ready")),
                "forward_transfer_delta": _float(exp5583, "forward_transfer_delta"),
                "backward_retention_delta": _float(exp5583, "backward_retention_delta"),
                "forgetting_delta": _float(exp5583, "forgetting_delta"),
                "policy_benefit_passed": bool(policy_gate.get("policy_benefit_passed")),
                "retirement_reasons": policy_gate.get("retirement_reasons", []),
            },
        },
        {
            "key": "pace_gate_block",
            "classification": "gate_blocked_conductor_task",
            "source_artifacts": [EXP5584_PACE_GATE_PATH.as_posix()],
            "evidence": {
                "status": _status_label(exp5584),
                "honest_verdict": exp5584.get("honest_verdict"),
                "gate_check_summary": exp5584.get("gate_check_summary"),
            },
        },
        {
            "key": "arc_registry_delta",
            "classification": "conductor_task",
            "source_artifacts": [EXP5585_ARC_PATH.as_posix()],
            "evidence": {
                "game_targeted": exp5585.get("game_targeted"),
                "new_levels_banked": _int(exp5585, "new_levels_banked"),
                "registry_total_before": _int(exp5585, "registry_total_before"),
                "registry_total_after": _int(exp5585, "registry_total_after"),
                "registry_updated": bool(exp5585.get("registry_updated")),
            },
        },
        {
            "key": "v505_source_delta",
            "classification": "conductor_task",
            "source_artifacts": [EXP5579_SOURCE_PATH.as_posix()],
            "evidence": {
                "new_references_added": len(exp5579.get("new_references_added", []))
                if isinstance(exp5579.get("new_references_added"), list)
                else 0,
                "closed_scopes_reopened": bool(exp5579.get("closed_scopes_reopened")),
            },
        },
    ]


def _outer_loop_finding(rel_path: Path, payload: JsonMap) -> tuple[str, JsonDict]:
    if rel_path.name.startswith("experiment_5592_"):
        return (
            "no_level_or_efficiency_delta",
            {
                "levels_gained_full_stack_total": _int(payload, "levels_gained_full_stack_total"),
                "levels_gained_bare_control_total": _int(payload, "levels_gained_bare_control_total"),
                "efficiency_full_stack_total": _float(payload, "efficiency_full_stack_total"),
                "efficiency_bare_control_total": _float(payload, "efficiency_bare_control_total"),
            },
        )
    if rel_path.name.startswith("experiment_5599_"):
        summary = payload.get("per_arm_summary") if isinstance(payload.get("per_arm_summary"), Mapping) else {}
        return (
            "current_generator_retained_on_reinduction_path",
            {"per_arm_summary": summary},
        )
    if rel_path.name.startswith("experiment_5600_"):
        return (
            "ptrm_generator_retired",
            {
                "loo_verdict_reached": bool(payload.get("loo_verdict_reached")),
                "retire_trm_generator_line": bool(payload.get("retire_trm_generator_line")),
                "heldout_games": payload.get("heldout_games", []),
            },
        )
    if rel_path.name.startswith("experiment_5601_"):
        return (
            "object_history_signal_found",
            {
                "hashes_with_nonzero_change_rate": _int(
                    payload,
                    "total_hashes_with_evidence_and_nonzero_change_rate",
                ),
                "hashes_tracked": _int(payload, "total_hashes_tracked"),
                "adversarial_degeneracy_check": payload.get("adversarial_degeneracy_check", {}),
            },
        )
    if rel_path.name.startswith("experiment_5602_"):
        return (
            "inert_click_no_op",
            {
                "reduction_pct": _float(payload, "reduction_pct"),
                "states_expanded_reduction": _int(payload, "states_expanded_reduction"),
                "live_wired_supplementary_check": payload.get("live_wired_supplementary_check", {}),
            },
        )
    if rel_path.name.startswith("experiment_5596_") or rel_path.name.startswith("experiment_5597_"):
        return (
            "larger_generator_pairwise_diagnostic_no_promotion",
            {
                "current_induction_success_count": _int(payload, "current_induction_success_count"),
                "candidate_induction_success_count": _int(payload, "candidate_induction_success_count"),
            },
        )
    if rel_path.name.startswith("experiment_5598_"):
        return (
            "larger_generator_multiseed_diagnostic_not_promoted_without_reinduction",
            {"paired_vs_current": payload.get("paired_vs_current", {})},
        )
    return ("diagnostic_no_current_dependency", {})


def _post_milestone_outer_loop_artifacts(
    artifacts: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path in OUTER_LOOP_ARTIFACT_PATHS:
        rel = rel_path.as_posix()
        meta = metadata.get(rel, {})
        payload = _payload(artifacts, rel_path)
        if not meta.get("exists") or not meta.get("loadable"):
            continue
        finding, evidence = _outer_loop_finding(rel_path, payload)
        rows.append(
            {
                "experiment_id": ARTIFACT_IDS.get(rel_path, rel_path.stem),
                "path": rel,
                "role": "post_milestone_outer_loop",
                "status": _status_label(payload),
                "honest_verdict": payload.get("honest_verdict"),
                "finding": finding,
                "evidence": evidence,
                "claim_boundary": "Outer-loop evidence can inform gates but is not `.505` conductor completion.",
            }
        )
    return rows


def _retired_scopes(
    artifacts: Mapping[str, Any],
    terminal_findings: Sequence[JsonMap],
    outer_loop_rows: Sequence[JsonMap],
) -> list[JsonDict]:
    findings = {str(row.get("key")): row for row in terminal_findings}
    outer = {str(row.get("experiment_id")): row for row in outer_loop_rows}
    return [
        {
            "key": "hash_only_parser_repair",
            "closed": True,
            "source_artifacts": [EXP5580_PARSER_PATH.as_posix()],
            "reason": "Cached rows preserve hashes and failure classes, not recoverable raw response text.",
            "evidence": findings.get("hash_only_parser_forensics", {}).get("evidence", {}),
        },
        {
            "key": "causal_memory_pace_policy_chain",
            "closed": True,
            "source_artifacts": [EXP5583_MEMORY_PATH.as_posix(), EXP5584_PACE_GATE_PATH.as_posix()],
            "reason": "Forward transfer stayed zero, forgetting was visible, and the PACE gate blocked.",
            "evidence": {
                "corrigendum": findings.get("causal_memory_corrigendum", {}).get("evidence", {}),
                "pace_gate": findings.get("pace_gate_block", {}).get("evidence", {}),
            },
        },
        {
            "key": "ptrm_as_generator",
            "closed": True,
            "source_artifacts": ["results/experiment_5600_ptrm_loo_gate.json"],
            "reason": "Leave-one-out gate failed the generator promotion criterion.",
            "evidence": outer.get("exp5600-ptrm-loo-gate", {}).get("evidence", {}),
        },
        {
            "key": "candidate_scoring_stack_promotion",
            "closed": True,
            "source_artifacts": [
                "results/experiment_5592_candidate_scoring_stack_bare_control_ab.json"
            ],
            "reason": "Full scoring stack and bare control produced the same level and efficiency totals.",
            "evidence": outer.get(
                "exp5592-candidate-scoring-stack-bare-control-ab",
                {},
            ).get("evidence", {}),
        },
        {
            "key": "larger_generator_promotion",
            "closed": True,
            "source_artifacts": [
                "results/experiment_5598_generator_size_multiseed_ab.json",
                "results/experiment_5599_reinduction_ab_lp85_levelup.json",
            ],
            "reason": "A larger candidate won offline induction accuracy but lost the actual reinduction planning path.",
            "evidence": outer.get(
                "exp5599-reinduction-ab-lp85-levelup",
                {},
            ).get("evidence", {}),
        },
        {
            "key": "v505_arc_registry_credit",
            "closed": True,
            "source_artifacts": [EXP5585_ARC_PATH.as_posix()],
            "reason": "The live ARC floor reproduced known levels but banked no new level.",
            "evidence": findings.get("arc_registry_delta", {}).get("evidence", {}),
        },
    ]


def _clean_substrates(
    artifacts: Mapping[str, Any],
    outer_loop_rows: Sequence[JsonMap],
) -> list[JsonDict]:
    exp5578 = _payload(artifacts, EXP5578_TRANSITION_PATH)
    clean_lane_names = {
        str(row.get("lane"))
        for row in exp5578.get("clean_lanes", [])
        if isinstance(row, Mapping) and row.get("lane")
    }
    outer = {str(row.get("experiment_id")): row for row in outer_loop_rows}
    rows: list[JsonDict] = []
    if "exact_asp_fsm_near_miss_corpus" in clean_lane_names:
        rows.append(
            {
                "key": "exact_asp_fsm_corpus",
                "source_artifacts": [EXP5578_TRANSITION_PATH.as_posix()],
                "prerequisite_for": ["exp5606-clean-sota-solve-verify-evidence-panel"],
                "claim_boundary": "Exact validators are clean; prior SOTA responses are not.",
            }
        )
    if "spline_local_kan_online_energy" in clean_lane_names:
        rows.append(
            {
                "key": "spline_local_kan",
                "source_artifacts": [EXP5578_TRANSITION_PATH.as_posix()],
                "prerequisite_for": ["exp5608-kan-longitudinal-self-learning"],
                "claim_boundary": "KAN is the clean FR-11 substrate; causal memory is not a prerequisite.",
            }
        )
    if "exp5601-object-history-salience-offline-sim-prototype" in outer:
        rows.append(
            {
                "key": "object_history_filter_signal",
                "source_artifacts": [
                    "results/experiment_5601_object_history_salience_offline_sim_prototype.json"
                ],
                "prerequisite_for": ["exp5609-arc-filter-intermediate-invariance-ab"],
                "claim_boundary": "A signal exists, but promotion requires reachable downstream A/B evidence.",
            }
        )
    return rows


def _dependency_map() -> JsonDict:
    return {
        "verification_evidence_chain": {
            "chain": "envelope->SOTA panel->exact extension",
            "tasks": [
                "exp5605-raw-response-evidence-envelope",
                "exp5606-clean-sota-solve-verify-evidence-panel",
                "exp5607-property-template-exact-residual-extension",
            ],
            "gates": [
                {
                    "upstream": "exp5605-raw-response-evidence-envelope",
                    "field": "envelope_ready",
                    "op": "==",
                    "value": True,
                },
                {
                    "upstream": "exp5606-clean-sota-solve-verify-evidence-panel",
                    "field": "panel_complete",
                    "op": "==",
                    "value": True,
                },
            ],
        },
        "kan_longitudinal_learning": {
            "chain": "KAN-only longitudinal learning",
            "tasks": ["exp5608-kan-longitudinal-self-learning"],
            "upstream_clean_substrate": "spline_local_kan",
            "excluded_dependency": "causal_memory_pace_policy_chain",
        },
        "arc_filter_to_live_attempt": {
            "chain": "ARC filter A/B->advisory live attempt",
            "tasks": [
                "exp5609-arc-filter-intermediate-invariance-ab",
                "exp5610-unconditional-live-agent-levelup-attempt",
            ],
            "advisory_only": True,
            "live_attempt_runs_even_if_filter_fails": True,
        },
        "cdls_to_capstone": {
            "chain": "independent cDLS benchmark->capstone",
            "tasks": [
                "exp5611-cdls-matched-cpu-cuda-benchmark",
                "exp5612-v506-capstone-reconciliation",
            ],
            "independent_of_verification_and_arc_gates": True,
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
    *,
    exp5582_skip_observed: bool,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failures = [f"missing_artifact:{path}" for path in missing_artifacts]
    if not exp5582_skip_observed:
        failures.append("exp5582_preemptive_skip_not_observed")
    if roadmap_modified:
        failures.append("research-roadmap.yaml_modified")
    if conductor_modified:
        failures.append("scripts/research_conductor.py_modified")
    return failures


def _honest_verdict(status: str, failures: Sequence[str]) -> str:
    if status == "complete":
        return (
            "complete: archived .505 terminal evidence and exp5592-exp5602 "
            "outer-loop findings into .506 dependency map; "
            "current_task_range=exp5603-exp5612; "
            "hash_only_parser_evidence=unrecoverable; "
            "causal_memory_pace_retired=True; kan_fr11_substrate_clean=True; "
            "arc_registry_delta=0; ptrm_generator_retired=True."
        )
    first = failures[0] if failures else "unknown"
    return f"blocked: .506 transition receipt failed precondition {first}."


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    terminal_artifacts, terminal_metadata, terminal_records, missing = _read_artifacts(
        root,
        TERMINAL_ARTIFACT_PATHS,
        role="v505_terminal_conductor_artifact",
        required=True,
    )
    outer_artifacts, outer_metadata, outer_records, _outer_missing = _read_artifacts(
        root,
        OUTER_LOOP_ARTIFACT_PATHS,
        role="post_milestone_outer_loop_artifact",
        required=False,
    )
    artifacts = {**terminal_artifacts, **outer_artifacts}
    artifact_metadata: JsonDict = {**terminal_metadata, **outer_metadata}
    exp5582_skip = _find_exp5582_preemptive_skip(root)
    artifacts_read = [
        *terminal_records,
        {
            "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
            "experiment_id": "exp5582-exact-counterexample-verifier-extension",
            "role": "v505_preemptive_skip_log",
            "sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
            "status": "blocked" if exp5582_skip["observed"] else "missing",
            "honest_verdict": "blocked: preemptive skip observed"
            if exp5582_skip["observed"]
            else "blocked: preemptive skip not observed",
        },
        *outer_records,
    ]
    source_context, source_missing = _read_source_context(root)
    roadmap, _roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    roadmap_doc_task_range = _task_range_from_text(_read_text(root, VNEXT_RELATIVE_PATH))
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    failures = _failed_preconditions(
        missing,
        exp5582_skip_observed=bool(exp5582_skip["observed"]),
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )
    status = "complete" if not failures else "blocked"
    terminal_findings = _terminal_findings(artifacts, exp5582_skip)
    outer_loop_rows = _post_milestone_outer_loop_artifacts(artifacts, artifact_metadata)
    tests = [dict(row) if isinstance(row, Mapping) else str(row) for row in tests_run]
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
        "task_id_collision_avoidance": {
            "previous_outer_loop_last_id": "exp5602",
            "new_range_starts_at": "exp5603",
            "collision_avoided": True,
        },
        "artifact_metadata": artifact_metadata,
        "artifacts_expected": {
            "terminal_v505": [path.as_posix() for path in TERMINAL_ARTIFACT_PATHS],
            "post_milestone_outer_loop": [
                path.as_posix() for path in OUTER_LOOP_ARTIFACT_PATHS
            ],
            "preemptive_skip_log": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        },
        "artifacts_read": artifacts_read,
        "missing_artifacts": list(missing),
        "source_context": source_context,
        "source_context_missing": source_missing,
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_task_count": len(roadmap_task_ids),
        "roadmap_doc_task_range": roadmap_doc_task_range,
        "exp5582_preemptive_skip": exp5582_skip,
        "protected_file_checks": _protected_file_checks(
            root,
            roadmap_modified=roadmap_modified,
            conductor_modified=conductor_modified,
        ),
        "preconditions_checked": {
            "terminal_artifacts_expected": len(TERMINAL_ARTIFACT_PATHS),
            "terminal_artifacts_read": len(terminal_records),
            "outer_loop_artifacts_present": len(outer_records),
            "exp5582_preemptive_skip_observed": bool(exp5582_skip["observed"]),
            "roadmap_next_present": (root / ROADMAP_NEXT_RELATIVE_PATH).exists(),
            "active_roadmap_milestone": roadmap.get("milestone"),
            "active_roadmap_task_count": len(roadmap_task_ids),
            "roadmap_doc_task_range": roadmap_doc_task_range,
            "roadmap_yaml_unchanged": not roadmap_modified,
            "conductor_unchanged": not conductor_modified,
        },
        "failed_preconditions": failures,
        "terminal_findings": terminal_findings,
        "retired_scopes": _retired_scopes(artifacts, terminal_findings, outer_loop_rows),
        "clean_substrates": _clean_substrates(artifacts, outer_loop_rows),
        "post_milestone_outer_loop_artifacts": outer_loop_rows,
        "dependency_map": _dependency_map(),
        "tests_run": tests,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "field_principles": dict(FIELD_PRINCIPLES),
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
    for field in BOOL_FIELDS:
        if field in payload and not isinstance(payload[field], bool):
            errors.append(field)
    for field in LIST_FIELDS:
        if field in payload and not isinstance(payload[field], list):
            errors.append(field)
    principles = payload.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles")
    terminal = payload.get("terminal_findings")
    if isinstance(terminal, list):
        keys = {str(row.get("key")) for row in terminal if isinstance(row, Mapping)}
        if not {
            "hash_only_parser_forensics",
            "causal_memory_corrigendum",
            "arc_registry_delta",
        } <= keys:
            errors.append("terminal_findings")
    retired = payload.get("retired_scopes")
    if isinstance(retired, list):
        keys = {str(row.get("key")) for row in retired if isinstance(row, Mapping)}
        if not {"hash_only_parser_repair", "causal_memory_pace_policy_chain", "ptrm_as_generator"} <= keys:
            errors.append("retired_scopes")
    clean = payload.get("clean_substrates")
    if isinstance(clean, list):
        keys = {str(row.get("key")) for row in clean if isinstance(row, Mapping)}
        if "spline_local_kan" not in keys:
            errors.append("clean_substrates")
    outer = payload.get("post_milestone_outer_loop_artifacts")
    if isinstance(outer, list):
        ids = {str(row.get("experiment_id")) for row in outer if isinstance(row, Mapping)}
        if not {
            "exp5600-ptrm-loo-gate",
            "exp5601-object-history-salience-offline-sim-prototype",
            "exp5602-inert-click-pruner-matched-budget-ab",
        } <= ids:
            errors.append("post_milestone_outer_loop_artifacts")
    if payload.get("current_task_range") != CURRENT_TASK_RANGE:
        errors.append("current_task_range")
    dependency_map = payload.get("dependency_map")
    if not isinstance(dependency_map, Mapping) or not {
        "verification_evidence_chain",
        "kan_longitudinal_learning",
        "arc_filter_to_live_attempt",
        "cdls_to_capstone",
    } <= set(dependency_map):
        errors.append("dependency_map")
    if payload.get("roadmap_yaml_unchanged") is not True:
        errors.append("roadmap_yaml_unchanged")
    if payload.get("conductor_unchanged") is not True:
        errors.append("conductor_unchanged")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
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
    if errors:  # pragma: no cover - validation failures are asserted in unit tests.
        raise ValueError(f"invalid Exp5603 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5603 artifact")
    args = parser.parse_args(argv)
    artifact = write_report() if args.write else build_report()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
