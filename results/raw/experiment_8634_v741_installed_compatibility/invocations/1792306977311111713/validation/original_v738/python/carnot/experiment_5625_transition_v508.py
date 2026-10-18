"""Exp5625 transition receipt from milestone .507 into .508.

Spec refs: REQ-REPORT-5625, SCENARIO-REPORT-5625,
SCENARIO-REPORT-5625-DEPENDENCY-MAP,
SCENARIO-REPORT-5625-FIELD-PRINCIPLES.

This module is a record-only evidence lock. It reads the terminal `.507`
artifacts and records which facts are safe prerequisites for `.508`. The
important boundary is negative as much as positive: a blocked runtime
certificate, an adversarially flagged ARC proxy, or an unpromoted KAN
controller must stay visible instead of being rounded into a clean dependency.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5625_transition_v508.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")

EXPERIMENT = "experiment_5625_transition_v508"
EXPERIMENT_ID = "exp5625-transition-v508"
PREVIOUS_MILESTONE = "2026.07.507"
CURRENT_MILESTONE = "2026.07.508"
PREVIOUS_TASK_RANGE = "exp5613-exp5624"
CURRENT_TASK_RANGE = "exp5625-exp5635"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5625
SCHEMA = "carnot.experiment_5625.transition_v508.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-REPORT-5625",
    "SCENARIO-REPORT-5625",
    "SCENARIO-REPORT-5625-DEPENDENCY-MAP",
    "SCENARIO-REPORT-5625-FIELD-PRINCIPLES",
)

EXP5613_TRANSITION_PATH = Path("results/experiment_5613_transition_v507.json")
EXP5614_SOURCE_PATH = Path("results/experiment_5614_v507_source_delta_ingestion.json")
EXP5615_RUNTIME_PATH = Path("results/experiment_5615_native_llamacpp_cuda_runtime_certificate.json")
EXP5616_STREAM_PATH = Path("results/experiment_5616_exact_nonstationary_constraint_stream.json")
EXP5617_DURATION_PATH = Path("results/experiment_5617_kan_critical_task_duration_map.json")
EXP5618_KAN_PATH = Path("results/experiment_5618_predictive_window_kan_self_learning.json")
EXP5619_ARC_PROXY_PATH = Path("results/experiment_5619_arc_forward_inverse_transition_cycle.json")
EXP5620_ARC_AB_PATH = Path("results/experiment_5620_arc_cycle_guarded_live_update_ab.json")
EXP5621_ARC_LEVEL_PATH = Path("results/experiment_5621_arc_live_self_discovery_levelup_v507.json")
EXP5622_CDLS_EXACT_PATH = Path("results/experiment_5622_cdls_exact_kernel_audit.json")
EXP5623_CDLS_CROSSOVER_PATH = Path("results/experiment_5623_cdls_multiseed_cpu_cuda_crossover.json")
EXP5624_CAPSTONE_PATH = Path("results/experiment_5624_v507_capstone_reconciliation.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5613-transition-v507": EXP5613_TRANSITION_PATH,
    "exp5614-v507-source-delta-ingestion": EXP5614_SOURCE_PATH,
    "exp5615-native-llamacpp-cuda-runtime-certificate": EXP5615_RUNTIME_PATH,
    "exp5616-exact-nonstationary-constraint-stream": EXP5616_STREAM_PATH,
    "exp5617-kan-critical-task-duration-map": EXP5617_DURATION_PATH,
    "exp5618-predictive-window-kan-self-learning": EXP5618_KAN_PATH,
    "exp5619-arc-forward-inverse-transition-cycle": EXP5619_ARC_PROXY_PATH,
    "exp5620-arc-cycle-guarded-live-update-ab": EXP5620_ARC_AB_PATH,
    "exp5621-arc-live-self-discovery-levelup-v507": EXP5621_ARC_LEVEL_PATH,
    "exp5622-cdls-exact-kernel-audit": EXP5622_CDLS_EXACT_PATH,
    "exp5623-cdls-multiseed-cpu-cuda-crossover": EXP5623_CDLS_CROSSOVER_PATH,
    "exp5624-v507-capstone-reconciliation": EXP5624_CAPSTONE_PATH,
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
    "exp5625-transition-v508",
    "exp5626-v508-source-delta-ingestion",
    "exp5627-online-conformal-kan-qualification",
    "exp5628-conformal-active-spline-kan-csl",
    "exp5629-conformal-kan-independent-audit",
    "exp5630-arc-epistemic-object-probe-prototype",
    "exp5631-arc-epistemic-probe-live-ab",
    "exp5632-arc-live-self-discovery-levelup-v508",
    "exp5633-temperature-exchange-cdls-exact-audit",
    "exp5634-temperature-exchange-cdls-quality",
    "exp5635-v508-capstone-reconciliation",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "One-line annotations for every required headline and gate field.",
    "artifacts_read": "every claim traces to a file",
    "terminal_findings": "only observed outcomes cross milestones",
    "promoted_substrates": "clean prerequisites are explicit",
    "promising_unpromoted_substrates": "Exp5618 is not laundered into a promotion",
    "retired_scopes": "failed chains remain closed",
    "adversarial_flags_preserved": "flags remain visible",
    "current_task_range": "IDs do not collide",
    "dependency_map": "gates are auditable",
    "inference_substrate": "no new inference occurred",
    "reproducibility_checksum": "transition content is stable",
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
    "promising_unpromoted_substrates",
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
        "command": ".venv/bin/pytest tests/python/test_experiment_5625_transition_v508.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5625_transition_v508.py -m pytest "
            "tests/python/test_experiment_5625_transition_v508.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5625_transition_v508.py --fail-under=100"
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


def _nested_total(value: Any) -> int:
    if isinstance(value, Mapping):
        if "total" in value:
            return _nested_total(value["total"])
        return sum(_nested_total(item) for item in value.values())
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _all_true(value: Any) -> bool:
    return bool(isinstance(value, Mapping) and value and all(item is True for item in value.values()))


def _extract_exp_number(value: str) -> int | None:
    match = re.search(r"(?:exp|experiment_)(\d+)", value)
    return int(match.group(1)) if match else None


def _completed_task_ids(root: Path) -> list[str]:
    text = _read_text(root, RESEARCH_COMPLETE_RELATIVE_PATH)
    return sorted(set(re.findall(r"\bid:\s*(exp\d+(?:[-_][A-Za-z0-9_]+)*)", text)))


def _retired_task_ids(root: Path) -> list[str]:
    text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    rows = [line for line in text.splitlines() if "retired" in line.lower()]
    return sorted(set(re.findall(r"\bexp\d+(?:[-_][A-Za-z0-9_]+)*", "\n".join(rows))))


def _collision_check(root: Path, roadmap_task_ids: Sequence[str]) -> JsonDict:
    completed = _completed_task_ids(root)
    retired = _retired_task_ids(root)
    current_numbers = set(range(5625, 5636))
    completed_colliding = [
        task_id
        for task_id in completed
        if (number := _extract_exp_number(task_id)) in current_numbers
    ]
    retired_colliding = [
        task_id
        for task_id in retired
        if (number := _extract_exp_number(task_id)) in current_numbers
    ]
    highest = max((_extract_exp_number(task_id) or 0 for task_id in completed), default=0)
    colliding = sorted(set(completed_colliding + retired_colliding))
    return {
        "current_task_range": CURRENT_TASK_RANGE,
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "roadmap_task_ids": list(roadmap_task_ids),
        "completed_ids_checked_count": len(completed),
        "retired_ids_checked_count": len(retired),
        "highest_completed_id": f"exp{highest}" if highest else None,
        "completed_colliding_ids": completed_colliding,
        "retired_colliding_ids": retired_colliding,
        "colliding_ids": colliding,
        "collision_free": not colliding,
        "range_matches_roadmap": list(roadmap_task_ids) == list(EXPECTED_TASK_IDS),
    }


def terminal_findings(artifacts: Mapping[str, JsonMap], metadata: JsonMap) -> dict[str, JsonDict]:
    capstone = _payload(artifacts, EXP5624_CAPSTONE_PATH)
    promoted = set(capstone.get("promoted_tasks") or [])
    retired = set(capstone.get("retired_tasks") or [])
    blocked = set(capstone.get("blocked_tasks") or [])
    gate_skipped = set(capstone.get("gate_skipped_tasks") or [])
    findings: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        rel = rel_path.as_posix()
        payload = _payload(artifacts, rel_path)
        meta = metadata.get(rel, {})
        status = _status_for_payload(payload, meta)
        findings[task_id] = {
            "artifact_path": rel,
            "status": status,
            "capstone_classification": {
                "promoted": task_id in promoted,
                "retired": task_id in retired,
                "blocked": task_id in blocked,
                "gate_skipped": task_id in gate_skipped,
            },
            "honest_verdict": _verdict(payload) or None,
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "corrigendum_pending": payload.get("corrigendum_pending", []),
            "sha256": meta.get("sha256"),
            "metadata_error": meta.get("error"),
            "supports_positive_claim": status == "complete" and task_id not in retired,
        }
    return findings


def _exact_stream_clean(exp5616: JsonMap, status: str) -> bool:
    return bool(
        status == "complete"
        and _float(exp5616, "fixture_ready_score") >= 1.0
        and _int(exp5616, "dataset_row_count") > 0
        and _int(exp5616, "exact_oracle_label_count") > 0
        and _int(exp5616, "oracle_label_error_count") == 0
        and not exp5616.get("llm_invoked")
        and not exp5616.get("policy_fit")
        and _all_true(exp5616.get("readiness_gates"))
    )


def _cdls_exact_clean(exp5622: JsonMap, status: str) -> bool:
    return bool(
        status == "complete"
        and _float(exp5622, "kernel_audit_ready_score") >= 1.0
        and exp5622.get("correction_applied")
        and exp5622.get("broken_kernel_controls_rejected")
        and _float(exp5622, "transition_row_sum_error_max") <= 1e-12
        and _float(exp5622, "detailed_balance_residual_max") <= 1e-12
        and _float(exp5622, "exact_distribution_tv_max") <= 1e-10
    )


def promoted_substrates(
    artifacts: Mapping[str, JsonMap],
    findings: Mapping[str, JsonMap],
) -> list[JsonDict]:
    exp5616 = _payload(artifacts, EXP5616_STREAM_PATH)
    exp5622 = _payload(artifacts, EXP5622_CDLS_EXACT_PATH)
    rows: list[JsonDict] = []
    if _exact_stream_clean(
        exp5616,
        str(findings["exp5616-exact-nonstationary-constraint-stream"]["status"]),
    ):
        rows.append(
            {
                "key": "exact_nonstationary_constraint_stream",
                "source_artifacts": [EXP5616_STREAM_PATH.as_posix()],
                "prerequisite_for": [
                    "exp5627-online-conformal-kan-qualification",
                    "exp5628-conformal-active-spline-kan-csl",
                    "exp5629-conformal-kan-independent-audit",
                ],
                "evidence": {
                    "fixture_ready_score": exp5616.get("fixture_ready_score"),
                    "dataset_row_count": exp5616.get("dataset_row_count"),
                    "stream_count": exp5616.get("stream_count"),
                    "exact_oracle_label_count": exp5616.get("exact_oracle_label_count"),
                    "oracle_label_error_count": exp5616.get("oracle_label_error_count"),
                },
                "claim_boundary": "exact labeled stream only; no learner promotion by itself",
            }
        )
    if _cdls_exact_clean(exp5622, str(findings["exp5622-cdls-exact-kernel-audit"]["status"])):
        rows.append(
            {
                "key": "corrected_cdls_exact_kernel",
                "source_artifacts": [EXP5622_CDLS_EXACT_PATH.as_posix()],
                "prerequisite_for": [
                    "exp5633-temperature-exchange-cdls-exact-audit",
                    "exp5634-temperature-exchange-cdls-quality",
                ],
                "evidence": {
                    "kernel_audit_ready_score": exp5622.get("kernel_audit_ready_score"),
                    "correction_applied": bool(exp5622.get("correction_applied")),
                    "final_kernel": (exp5622.get("correction_spec") or {}).get("final_kernel")
                    if isinstance(exp5622.get("correction_spec"), Mapping)
                    else None,
                    "transition_row_sum_error_max": exp5622.get("transition_row_sum_error_max"),
                    "detailed_balance_residual_max": exp5622.get("detailed_balance_residual_max"),
                    "exact_distribution_tv_max": exp5622.get("exact_distribution_tv_max"),
                    "energy_histogram_tv_max": exp5622.get("energy_histogram_tv_max"),
                },
                "claim_boundary": "exact corrected kernel only; no timing, crossover, or hardware claim",
            }
        )
    return rows


def promising_unpromoted_substrates(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    exp5617 = _payload(artifacts, EXP5617_DURATION_PATH)
    exp5618 = _payload(artifacts, EXP5618_KAN_PATH)
    capstone = _payload(artifacts, EXP5624_CAPSTONE_PATH)
    decisions = capstone.get("promotion_decisions") if isinstance(capstone, Mapping) else {}
    kan_decision = decisions.get("predictive_window_kan_self_learning", {}) if isinstance(decisions, Mapping) else {}
    return [
        {
            "key": "predictive_window_active_spline_kan",
            "source_artifacts": [
                EXP5617_DURATION_PATH.as_posix(),
                EXP5618_KAN_PATH.as_posix(),
                EXP5624_CAPSTONE_PATH.as_posix(),
            ],
            "promoted": False,
            "terminal_state": "promising_but_unpromoted",
            "blocking_gate": {
                "critical_duration_fit_r2": exp5617.get("critical_duration_fit_r2"),
                "required_critical_duration_fit_r2": 0.5,
                "nondegenerate_switch_case_count": len(exp5617.get("nondegenerate_switch_cases") or []),
                "capstone_decision": kan_decision.get("decision"),
            },
            "evidence": {
                "continuous_self_learning_ready": bool(exp5618.get("continuous_self_learning_ready")),
                "delta_ale_vs_best_fixed": exp5618.get("delta_ale_vs_best_fixed"),
                "unsafe_false_accept_total": _nested_total(exp5618.get("unsafe_false_accept_count")),
                "no_model_weight_mutation": bool(exp5618.get("no_model_weight_mutation")),
                "llm_invoked": bool(exp5618.get("llm_invoked")),
                "llm_weight_training": bool(exp5618.get("llm_weight_training")),
            },
            "claim_boundary": "may seed conformal replication, but cannot be treated as promoted FR-11 evidence",
        }
    ]


def retired_scopes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    exp5613 = _payload(artifacts, EXP5613_TRANSITION_PATH)
    exp5615 = _payload(artifacts, EXP5615_RUNTIME_PATH)
    exp5619 = _payload(artifacts, EXP5619_ARC_PROXY_PATH)
    exp5620 = _payload(artifacts, EXP5620_ARC_AB_PATH)
    exp5623 = _payload(artifacts, EXP5623_CDLS_CROSSOVER_PATH)
    prior_retired = exp5613.get("retired_scopes", []) if isinstance(exp5613, Mapping) else []
    solve_verify = next(
        (row for row in prior_retired if isinstance(row, Mapping) and row.get("key") == "solve_versus_verify_panel"),
        {},
    )
    quality_pairs = exp5623.get("quality_gate_results_by_pair")
    failed_quality_count = len(
        [
            row
            for row in quality_pairs
            if isinstance(row, Mapping) and not bool(row.get("included_in_speedups"))
        ]
    ) if isinstance(quality_pairs, list) else 0
    return [
        {
            "key": "native_runtime_certificate",
            "closed": True,
            "source_artifacts": [EXP5615_RUNTIME_PATH.as_posix(), EXP5624_CAPSTONE_PATH.as_posix()],
            "reason": "Zero of three model runtimes earned the verifier-quality certificate.",
            "evidence": {
                "runtime_certificate_ready_score": exp5615.get("runtime_certificate_ready_score"),
                "models_certified_count": exp5615.get("models_certified_count"),
                "models_certified_denominator": exp5615.get("models_certified_denominator"),
                "native_cuda_ready": (exp5615.get("cuda_build_capability") or {}).get("native_cuda_ready")
                if isinstance(exp5615.get("cuda_build_capability"), Mapping)
                else None,
            },
        },
        {
            "key": "solve_versus_verify_chain",
            "closed": True,
            "source_artifacts": [EXP5613_TRANSITION_PATH.as_posix(), EXP5615_RUNTIME_PATH.as_posix()],
            "reason": "The prior solve-versus-verify panel remains closed, and Exp5615 computed no task accuracy.",
            "evidence": {
                "prior_scope": solve_verify,
                "no_task_accuracy_computed": bool(exp5615.get("no_task_accuracy_computed")),
                "solve_verify_accuracy_inferred": bool(exp5615.get("solve_verify_accuracy_inferred")),
            },
        },
        {
            "key": "arc_transition_cycle_proxy",
            "closed": True,
            "source_artifacts": [EXP5619_ARC_PROXY_PATH.as_posix()],
            "reason": "The proxy was safe by unsafe accepts but over-abstained and failed positive-control utility.",
            "evidence": {
                "flagged_adversarial": bool(exp5619.get("flagged_adversarial")),
                "cycle_verifier_positive_control_rate": exp5619.get("cycle_verifier_positive_control_rate"),
                "valid_transition_accept_rate": exp5619.get("valid_transition_accept_rate"),
                "unsafe_transition_accept_count": exp5619.get("unsafe_transition_accept_count"),
                "inverse_action_accuracy": exp5619.get("inverse_action_accuracy"),
            },
        },
        {
            "key": "arc_cycle_guarded_live_branch",
            "closed": True,
            "source_artifacts": [EXP5620_ARC_AB_PATH.as_posix()],
            "reason": "The guarded live A/B was skipped at the conductor pre-gate.",
            "evidence": {
                "blocked_at_layer": exp5620.get("blocked_at_layer"),
                "gate_check_summary": exp5620.get("gate_check_summary"),
                "gates_evaluated": exp5620.get("gates_evaluated", []),
            },
        },
        {
            "key": "cdls_timing_crossover",
            "closed": True,
            "source_artifacts": [EXP5623_CDLS_CROSSOVER_PATH.as_posix()],
            "reason": "No quality-matched pairs entered speedups, so timing and crossover claims stay closed.",
            "evidence": {
                "crossover_claim_allowed": bool(exp5623.get("crossover_claim_allowed")),
                "board_speedup_claimed": bool(exp5623.get("board_speedup_claimed")),
                "crossover_size": exp5623.get("crossover_size"),
                "successful_matched_pairs_count": len(exp5623.get("successful_matched_pairs") or []),
                "quality_gate_failed_pair_count": failed_quality_count,
            },
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
                "claim_boundary": "flagged evidence remains visible and is never a clean substrate",
            }
        )
    return rows


def dependency_map() -> JsonDict:
    return {
        "conformal_kan_qualification_to_audit": {
            "chain": "conformal qualification->KAN replication->independent audit",
            "tasks": [
                "exp5627-online-conformal-kan-qualification",
                "exp5628-conformal-active-spline-kan-csl",
                "exp5629-conformal-kan-independent-audit",
            ],
            "upstream_clean_substrate": "exact_nonstationary_constraint_stream",
            "promising_unpromoted_input": "predictive_window_active_spline_kan",
            "gates": [
                {
                    "upstream": "exp5627-online-conformal-kan-qualification",
                    "field": "conformal_qualification_ready_score",
                    "op": ">=",
                    "value": 1.0,
                },
                {
                    "upstream": "exp5627-online-conformal-kan-qualification",
                    "field": "worst_group_coverage",
                    "op": ">=",
                    "value": 0.9,
                },
                {
                    "upstream": "exp5627-online-conformal-kan-qualification",
                    "field": "exact_unsafe_accept_count",
                    "op": "==",
                    "value": 0,
                },
                {
                    "upstream": "exp5628-conformal-active-spline-kan-csl",
                    "field": "continuous_self_learning_ready",
                    "op": "==",
                    "value": True,
                },
                {
                    "upstream": "exp5629-conformal-kan-independent-audit",
                    "field": "independent_promotion_ready",
                    "op": "==",
                    "value": True,
                },
            ],
        },
        "epistemic_object_arc_to_level_attempt": {
            "chain": "epistemic object prototype->advisory live A/B->unconditional level attempt",
            "tasks": [
                "exp5630-arc-epistemic-object-probe-prototype",
                "exp5631-arc-epistemic-probe-live-ab",
                "exp5632-arc-live-self-discovery-levelup-v508",
            ],
            "closed_scope_not_reopened": "arc_transition_cycle_proxy",
            "gates": [
                {
                    "upstream": "exp5630-arc-epistemic-object-probe-prototype",
                    "field": "epistemic_object_probe_ready_score",
                    "op": ">=",
                    "value": 1.0,
                },
                {
                    "upstream": "exp5630-arc-epistemic-object-probe-prototype",
                    "field": "corruption_controls_safe",
                    "op": "==",
                    "value": True,
                },
                {
                    "upstream": "exp5631-arc-epistemic-probe-live-ab",
                    "field": "live_ab_promotion_allowed",
                    "op": "==",
                    "value": True,
                    "advisory_only": True,
                },
                {
                    "upstream": "exp5632-arc-live-self-discovery-levelup-v508",
                    "field": "live_attempt_executed",
                    "op": "==",
                    "value": True,
                    "unconditional": True,
                },
            ],
        },
        "temperature_exchange_cdls_to_quality": {
            "chain": "exact temperature exchange->gated quality trial",
            "tasks": [
                "exp5633-temperature-exchange-cdls-exact-audit",
                "exp5634-temperature-exchange-cdls-quality",
            ],
            "upstream_clean_substrate": "corrected_cdls_exact_kernel",
            "closed_scope_not_reopened": "cdls_timing_crossover",
            "gates": [
                {
                    "upstream": "exp5633-temperature-exchange-cdls-exact-audit",
                    "field": "replica_exchange_kernel_ready_score",
                    "op": ">=",
                    "value": 1.0,
                },
                {
                    "upstream": "exp5633-temperature-exchange-cdls-exact-audit",
                    "field": "exact_distribution_tv_max",
                    "op": "<=",
                    "value": "preregistered_exact_tolerance",
                },
                {
                    "upstream": "exp5634-temperature-exchange-cdls-quality",
                    "field": "replica_exchange_quality_promoted",
                    "op": "==",
                    "value": True,
                },
            ],
        },
        "capstone_reconciliation": {
            "chain": "Exp5625-Exp5634->Exp5635 reconciliation",
            "tasks": [*EXPECTED_TASK_IDS[:-1], "exp5635-v508-capstone-reconciliation"],
            "gate": "Exp5635 cannot promote blocked, skipped, development-proxy, flagged, unpromoted, retired, timing, or hardware-speedup evidence.",
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
    failures.extend(
        f"current_task_id_collision_completed:{task_id}"
        for task_id in collision.get("completed_colliding_ids", [])
    )
    failures.extend(
        f"current_task_id_collision_retired:{task_id}"
        for task_id in collision.get("retired_colliding_ids", [])
    )
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
            "complete: archived .507 terminal evidence into .508 dependency map; "
            "current_task_range=exp5625-exp5635; exact_stream_promoted=True; "
            "corrected_cdls_kernel_promoted=True; predictive_kan_promoted=False; "
            "native_runtime_certificate_closed=True; arc_transition_cycle_flagged=True; "
            "cdls_crossover_closed=True."
        )
    first = failures[0] if failures else "unknown"
    return f"blocked: .508 transition receipt failed precondition {first}."


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
            "conductor_log_checked": (root / CONDUCTOR_LOG_RELATIVE_PATH).exists(),
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
        "promising_unpromoted_substrates": promising_unpromoted_substrates(artifacts),
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
        if not {"exact_nonstationary_constraint_stream", "corrected_cdls_exact_kernel"} <= keys:
            errors.append("promoted_substrates")
    promising = payload.get("promising_unpromoted_substrates")
    if payload.get("status") == "complete" and isinstance(promising, list):
        keys = {str(row.get("key")) for row in promising if isinstance(row, Mapping)}
        if "predictive_window_active_spline_kan" not in keys:
            errors.append("promising_unpromoted_substrates")
    retired = payload.get("retired_scopes")
    if isinstance(retired, list):
        keys = {str(row.get("key")) for row in retired if isinstance(row, Mapping)}
        if not {
            "native_runtime_certificate",
            "solve_versus_verify_chain",
            "arc_transition_cycle_proxy",
            "arc_cycle_guarded_live_branch",
            "cdls_timing_crossover",
        } <= keys:
            errors.append("retired_scopes")
    flags = payload.get("adversarial_flags_preserved")
    if payload.get("status") == "complete" and isinstance(flags, list):
        ids = {str(row.get("task_id")) for row in flags if isinstance(row, Mapping)}
        if not {
            "exp5619-arc-forward-inverse-transition-cycle",
            "exp5621-arc-live-self-discovery-levelup-v507",
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
        "conformal_kan_qualification_to_audit",
        "epistemic_object_arc_to_level_attempt",
        "temperature_exchange_cdls_to_quality",
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
        raise ValueError(f"invalid Exp5625 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5625 artifact")
    args = parser.parse_args(argv)
    artifact = write_report() if args.write else build_report()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
