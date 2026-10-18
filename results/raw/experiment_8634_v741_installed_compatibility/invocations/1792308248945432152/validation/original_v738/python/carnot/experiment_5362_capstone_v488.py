"""Exp 5362: V488 capstone decision artifact.

Spec refs: REQ-CAPSTONE-5362, SCENARIO-CAPSTONE-5362,
SCENARIO-CAPSTONE-5362-BLOCKED-MISSING-INPUT,
SCENARIO-CAPSTONE-5362-FIELD-PRINCIPLES.

This module is a milestone-close synthesizer. It reads the local .488 result
artifacts and conductor notes, then keeps blocked, flagged, skipped, honest-null,
and partial hardware outcomes visible. The capstone does not rerun models,
solvers, ARC agents, or hardware probes; its job is to prevent a bounded
diagnostic from becoming a headline quality, internal-energy, ARC-progress, or
hardware-speedup claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5362_capstone_v488.json")
EXPERIMENT = "experiment_5362_capstone_v488"
EXPERIMENT_ID = "exp5362-capstone-v488"
MILESTONE = "2026.07.488"
SCHEMA = "carnot.experiment_5362_capstone_v488.v1"
RUN_DATE = "20260707"
RANDOM_SEED = 5362
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked_")

SPEC_REFS = (
    "REQ-CAPSTONE-5362",
    "SCENARIO-CAPSTONE-5362",
    "SCENARIO-CAPSTONE-5362-BLOCKED-MISSING-INPUT",
    "SCENARIO-CAPSTONE-5362-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": "Prevents `.488` synthesis from being confused with `.487`.",
    "status": "Lets conductor classify the capstone itself.",
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous milestone outcome."
    ),
    "inference_substrate": "Expected value is aggregation_from_upstream_artifacts.",
    "artifacts_read": "Lists which result files actually informed the synthesis.",
    "missing_blocked_flagged_or_skipped_artifacts": (
        "Prevents skipped work from being treated as success."
    ),
    "gate_table": (
        "One row per .488 decision gate with source artifacts, readiness truth, "
        "claim boundary, and imported evidence."
    ),
    "next_milestone_recommendation": ("Names the next decisive gate without overclaiming."),
    "cited_upstream_artifacts": "Makes the synthesis auditable.",
    "tests_run": "Lists local validation for artifact/docs edits.",
}

WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)
BARE_BOOLEAN_FIELDS = (
    "structured_protocol_clean",
    "constraint_tax_panel_ready",
    "tokenprob_feature_rows_ready",
    "carry_token_energy_signal_ready",
    "dependency_provenance_ready",
    "memory_tool_drift_ready",
    "self_learning_scaleup_ready",
    "solver_projection_ready",
    "pbit_schedule_signal_ready",
    "arc_new_level_banked",
    "hardware_speedup_claim",
    "active_roadmap_modified",
    "conductor_modified",
)
REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "milestone",
    "status",
    "run_date",
    "spec_refs",
    "result_path",
    "random_seed",
    "field_principles",
    "reproducibility_checksum",
    *WRAPPED_FIELDS,
    *BARE_BOOLEAN_FIELDS,
)

GATE_ORDER = (
    "source_delta",
    "structured_protocol",
    "constraint_tax_panel",
    "tokenprob_feature_audit",
    "carry_diagnostic",
    "dependency_provenance",
    "memory_tool_drift",
    "self_learning_scaleup",
    "solver_projection",
    "pbit_schedules",
    "arc_level_up",
    "hardware_continuity",
)


@dataclass(frozen=True)
class UpstreamArtifact:
    """One actual V488 upstream result artifact.

    The capstone reads this fixed ledger from disk. Missing or malformed rows
    block the capstone artifact itself because the synthesis would otherwise be
    reasoning over an incomplete close-state. Blocked or flagged payloads are
    still evidence and are recorded rather than rounded up.
    """

    experiment_number: int
    task_id: str
    relative_path: Path
    default_status: str = "missing_or_unreadable"


EXP5349 = UpstreamArtifact(
    5349,
    "exp5349-archive-487-activate-488",
    Path("results/experiment_5349_archive_487_activate_488.json"),
)
EXP5350 = UpstreamArtifact(
    5350,
    "exp5350-sota-source-delta-v488",
    Path("results/experiment_5350_sota_source_delta_v488.json"),
)
EXP5351 = UpstreamArtifact(
    5351,
    "exp5351-trigger-constrain-structured-protocol-v488",
    Path("results/experiment_5351_trigger_constrain_structured_protocol_v488.json"),
)
EXP5352 = UpstreamArtifact(
    5352,
    "exp5352-gated-constraint-tax-tool-action-panel-v488",
    Path("results/experiment_5352_gated_constraint_tax_tool_action_panel_v488.json"),
)
EXP5352_REQUESTED_ALIAS = Path("results/experiment_5352_constraint_tax_tool_action_panel_v488.json")
EXP5353 = UpstreamArtifact(
    5353,
    "exp5353-tokenprob-feature-audit-corrigendum-v488",
    Path("results/experiment_5353_tokenprob_feature_audit_corrigendum_v488.json"),
)
EXP5354 = UpstreamArtifact(
    5354,
    "exp5354-gated-arithmetic-carry-token-energy-v488",
    Path("results/experiment_5354_arithmetic_carry_token_energy_v488.json"),
)
EXP5355 = UpstreamArtifact(
    5355,
    "exp5355-dependency-provenance-self-learning-v488",
    Path("results/experiment_5355_dependency_provenance_self_learning_v488.json"),
)
EXP5356 = UpstreamArtifact(
    5356,
    "exp5356-memory-tool-drift-harness-v488",
    Path("results/experiment_5356_memory_tool_drift_harness_v488.json"),
)
EXP5357 = UpstreamArtifact(
    5357,
    "exp5357-gated-dependency-drift-self-learning-scaleup-v488",
    Path("results/experiment_5357_dependency_drift_self_learning_scaleup_v488.json"),
)
EXP5358 = UpstreamArtifact(
    5358,
    "exp5358-solver-projection-cut-bridge-v488",
    Path("results/experiment_5358_solver_projection_cut_bridge_v488.json"),
)
EXP5359 = UpstreamArtifact(
    5359,
    "exp5359-pbit-schedule-diagnostic-v488",
    Path("results/experiment_5359_pbit_schedule_diagnostic_v488.json"),
)
EXP5360 = UpstreamArtifact(
    5360,
    "exp5360-arc-perception-salience-levelup-attempt-v488",
    Path("results/experiment_5360_arc_perception_salience_levelup_attempt_v488.json"),
)
EXP5361 = UpstreamArtifact(
    5361,
    "exp5361-hardware-continuity-workload-v488",
    Path("results/experiment_5361_hardware_continuity_workload_v488.json"),
)

EXPECTED_ARTIFACTS = (
    EXP5349,
    EXP5350,
    EXP5351,
    EXP5352,
    EXP5353,
    EXP5354,
    EXP5355,
    EXP5356,
    EXP5357,
    EXP5358,
    EXP5359,
    EXP5360,
    EXP5361,
)

CITED_FIELDS_BY_PATH: dict[str, list[str]] = {
    str(EXP5349.relative_path): [
        "status",
        "honest_verdict",
        "roadmap_next_present",
        "active_roadmap_modified",
        "conductor_modified",
    ],
    str(EXP5350.relative_path): [
        "status",
        "honest_verdict",
        "new_actionable_findings_count",
        "executable_plan_change_required",
        "retired_scope_reopened",
    ],
    str(EXP5351.relative_path): [
        "structured_protocol_clean",
        "parse_success_rate",
        "schema_success_rate",
        "final_json_extraction_rate",
        "methodology_duration_s",
        "unsafe_false_accepts",
        "no_quality_claim",
    ],
    str(EXP5352.relative_path): [
        "status",
        "honest_verdict",
        "blocked_at_layer",
        "gate_check_summary",
        "gates_evaluated",
    ],
    str(EXP5353.relative_path): [
        "tokenprob_feature_rows_ready",
        "tokenprob_feature_row_count",
        "per_token_logprob_available",
        "topk_alternatives_available",
        "flagged_adversarial",
        "corrigendum_pending",
        "external_text_scorer_reopened",
    ],
    str(EXP5354.relative_path): [
        "carry_token_energy_signal_ready",
        "feature_complete_rate",
        "correct_vs_perturbed_margin",
        "missing_feature_names",
        "flagged_adversarial",
        "corrigendum_pending",
        "no_broad_hallucination_claim",
    ],
    str(EXP5355.relative_path): [
        "dependency_provenance_ready",
        "dependency_edge_precision",
        "dependency_edge_recall",
        "execution_feedback_attribution_rate",
        "unsafe_false_accepts",
        "no_weight_mutation",
    ],
    str(EXP5356.relative_path): [
        "memory_tool_drift_ready",
        "drift_case_count",
        "clean_selection_accuracy",
        "induced_tool_drift_rate",
        "rollback_recovery_rate",
        "unsafe_false_accepts",
        "no_weight_mutation",
    ],
    str(EXP5357.relative_path): [
        "self_learning_scaleup_ready",
        "multi_session_trace_count",
        "dependency_attribution_rate",
        "drift_detection_rate",
        "quality_delta_vs_always_full",
        "duplicated_metric_pairs",
        "unsafe_false_accepts",
        "no_weight_mutation",
    ],
    str(EXP5358.relative_path): [
        "solver_projection_ready",
        "solver_authoritative",
        "fallback_completeness_rate",
        "projection_success_rate",
        "post_projection_validity_rate",
        "unsafe_false_accepts",
    ],
    str(EXP5359.relative_path): [
        "pbit_schedule_signal_ready",
        "hardware_speedup_claim",
        "fixture_count",
        "schedule_variant_count",
        "false_accept_count",
        "claim_limits",
    ],
    str(EXP5360.relative_path): [
        "status",
        "honest_verdict",
        "target_game",
        "target_level",
        "new_level_banked",
        "offline_reproduced",
        "registry_updated",
        "perception_error_classes",
    ],
    str(EXP5361.relative_path): [
        "hardware_evidence_level",
        "polarfire_workload_validated",
        "authenticated_workload_run",
        "speedup_claim",
        "kv260_status",
        "polarfire_status",
        "gatemate_status",
        "blocked_reason",
    ],
}

NON_BLOCKING_ISSUE_CLASSIFICATIONS = {
    "requested_alias_missing",
    "honest_null",
    "hardware_subgate_blocked",
}


def value_of(value: Any) -> Any:
    """Return the machine value from a principle-wrapped or bare artifact field."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def wrapped(field: str, value: Any) -> JsonDict:
    """Attach the field principle required by the capstone schema."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _sha256(path: Path) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _path_key(source: UpstreamArtifact) -> str:
    return str(source.relative_path)


def _payload_value(payload: JsonMap | None, field: str, default: Any = None) -> Any:
    if payload is None:
        return default
    return value_of(payload.get(field, default))


def _status(payload: JsonMap | None, default: str = "missing_or_unreadable") -> str:
    return str(_payload_value(payload, "status", default))


def _verdict(payload: JsonMap | None) -> str:
    return str(_payload_value(payload, "honest_verdict", ""))


def _exp_id(payload: JsonMap, source: UpstreamArtifact) -> Any:
    return (
        value_of(payload.get("experiment_id"))
        or value_of(payload.get("experiment"))
        or source.task_id
    )


def read_conductor_outcomes(root: Path | str = REPO_ROOT) -> dict[int, list[JsonDict]]:
    """Parse conductor rows for Exp5349 through Exp5361 when the log exists."""

    path = Path(root) / "ops/conductor-log.md"
    outcomes: dict[int, list[JsonDict]] = {
        source.experiment_number: [] for source in EXPECTED_ARTIFACTS
    }
    if not path.exists():
        return outcomes
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line_number, line in enumerate(lines, 1):
        for source in EXPECTED_ARTIFACTS:
            exp_text = f"Exp {source.experiment_number}"
            compact_exp_text = f"Exp{source.experiment_number}"
            if exp_text not in line and compact_exp_text not in line:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 4:
                outcomes[source.experiment_number].append(
                    {
                        "line_number": line_number,
                        "timestamp": cells[0],
                        "status": cells[2],
                        "summary": cells[3],
                    }
                )
    return outcomes


def latest_conductor_outcome(
    outcomes: Mapping[int, Sequence[JsonDict]], experiment_number: int
) -> JsonDict | None:
    rows = list(outcomes.get(experiment_number, ()))
    return rows[-1] if rows else None


def read_upstream_artifacts(
    root: Path | str = REPO_ROOT,
    conductor_outcomes: Mapping[int, Sequence[JsonDict]] | None = None,
) -> tuple[dict[str, JsonDict], list[JsonDict], list[JsonDict]]:
    """Read every actual V488 upstream artifact and preserve unreadable inputs."""

    root_path = Path(root)
    conductor = conductor_outcomes or {}
    payloads: dict[str, JsonDict] = {}
    artifacts_read: list[JsonDict] = []
    missing_or_malformed: list[JsonDict] = []
    for source in EXPECTED_ARTIFACTS:
        path = root_path / source.relative_path
        path_key = _path_key(source)
        if not path.exists():
            missing_or_malformed.append(
                {
                    "experiment_number": source.experiment_number,
                    "task_id": source.task_id,
                    "path": path_key,
                    "classification": "missing",
                    "reason": "missing",
                }
            )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            missing_or_malformed.append(
                {
                    "experiment_number": source.experiment_number,
                    "task_id": source.task_id,
                    "path": path_key,
                    "classification": "malformed",
                    "reason": f"malformed_json:{exc.msg}",
                }
            )
            continue
        if not isinstance(payload, dict):
            missing_or_malformed.append(
                {
                    "experiment_number": source.experiment_number,
                    "task_id": source.task_id,
                    "path": path_key,
                    "classification": "malformed",
                    "reason": "not_json_object",
                }
            )
            continue
        latest = latest_conductor_outcome(conductor, source.experiment_number)
        payloads[path_key] = payload
        artifacts_read.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "path": path_key,
                "experiment_id": _exp_id(payload, source),
                "status": _status(payload, source.default_status),
                "honest_verdict": _verdict(payload),
                "flagged_adversarial": _payload_value(payload, "flagged_adversarial") is True,
                "conductor_outcome": latest,
                "sha256": _sha256(path),
            }
        )
    return payloads, artifacts_read, missing_or_malformed


def _is_blocked(payload: JsonMap | None) -> bool:
    status = _status(payload)
    verdict = _verdict(payload)
    return status == "blocked" or status.startswith("blocked_") or verdict.startswith("blocked_")


def _is_flagged(payload: JsonMap | None, latest: JsonMap | None) -> bool:
    latest_status = str((latest or {}).get("status", ""))
    return _payload_value(payload, "flagged_adversarial") is True or latest_status == "FLAGGED"


def _is_skipped(payload: JsonMap | None, latest: JsonMap | None) -> bool:
    latest_status = str((latest or {}).get("status", ""))
    return (
        latest_status in {"GATE_BLOCK", "SKIP"}
        or _payload_value(payload, "blocked_at_layer") == "conductor_pre_gate"
    )


def _requested_alias_rows(root: Path | str, payloads: Mapping[str, JsonDict]) -> list[JsonDict]:
    alias_path = Path(root) / EXP5352_REQUESTED_ALIAS
    if alias_path.exists():
        return []
    actual_present = _path_key(EXP5352) in payloads
    return [
        {
            "experiment_number": 5352,
            "task_id": EXP5352.task_id,
            "path": str(EXP5352_REQUESTED_ALIAS),
            "classification": "requested_alias_missing",
            "reason": (
                "roadmap prompt requested the non-gated path; actual gated "
                "conductor-pre-gate artifact was read"
            ),
            "actual_path": str(EXP5352.relative_path) if actual_present else None,
        }
    ]


def _hardware_subgate_rows(payloads: Mapping[str, JsonDict]) -> list[JsonDict]:
    payload = _payload(payloads, EXP5361)
    if payload is None:
        return []
    kv260 = _payload_value(payload, "kv260_status", {})
    gatemate = _payload_value(payload, "gatemate_status", {})
    blocked_reason = _payload_value(payload, "blocked_reason", {})
    kv260_blocked = isinstance(kv260, Mapping) and kv260.get("ssh_reachable") is not True
    gatemate_blocked = isinstance(gatemate, Mapping) and str(gatemate.get("status", "")).startswith(
        "blocked_"
    )
    if not kv260_blocked and not gatemate_blocked:
        return []
    return [
        {
            "experiment_number": 5361,
            "task_id": EXP5361.task_id,
            "path": str(EXP5361.relative_path),
            "classification": "hardware_subgate_blocked",
            "status": _status(payload),
            "honest_verdict": _verdict(payload),
            "blocked_reason": blocked_reason,
        }
    ]


def _arc_null_rows(payloads: Mapping[str, JsonDict]) -> list[JsonDict]:
    payload = _payload(payloads, EXP5360)
    if payload is None or _payload_value(payload, "new_level_banked") is True:
        return []
    return [
        {
            "experiment_number": 5360,
            "task_id": EXP5360.task_id,
            "path": str(EXP5360.relative_path),
            "classification": "honest_null",
            "status": _status(payload),
            "honest_verdict": _verdict(payload),
            "target_game": _payload_value(payload, "target_game"),
            "target_level": _payload_value(payload, "target_level"),
        }
    ]


def _blocked_flagged_or_skipped_rows(
    *,
    root: Path | str,
    payloads: Mapping[str, JsonDict],
    missing_or_malformed: Sequence[JsonDict],
    conductor_outcomes: Mapping[int, Sequence[JsonDict]],
) -> list[JsonDict]:
    rows = [dict(row) for row in missing_or_malformed]
    rows.extend(_requested_alias_rows(root, payloads))
    missing_paths = {str(row["path"]) for row in missing_or_malformed}
    for source in EXPECTED_ARTIFACTS:
        path_key = _path_key(source)
        if path_key in missing_paths:
            continue
        payload = payloads.get(path_key)
        latest = latest_conductor_outcome(conductor_outcomes, source.experiment_number)
        blocked = _is_blocked(payload)
        flagged = _is_flagged(payload, latest)
        skipped = _is_skipped(payload, latest)
        classification = (
            "conductor_gate_skip"
            if skipped
            else "blocked_and_flagged"
            if blocked and flagged
            else "blocked"
            if blocked
            else "flagged"
            if flagged
            else None
        )
        if classification is None:
            continue
        rows.append(
            {
                "experiment_number": source.experiment_number,
                "task_id": source.task_id,
                "path": path_key,
                "classification": classification,
                "status": _status(payload, source.default_status),
                "honest_verdict": _verdict(payload),
                "conductor_outcome": latest,
                "corrigendum_pending": _payload_value(payload, "corrigendum_pending", []),
                "blocked_at_layer": _payload_value(payload, "blocked_at_layer"),
                "gate_check_summary": _payload_value(payload, "gate_check_summary"),
            }
        )
    rows.extend(_arc_null_rows(payloads))
    rows.extend(_hardware_subgate_rows(payloads))
    return rows


def _issue_by_path(issues: Sequence[JsonMap]) -> dict[str, str]:
    return {
        str(row["path"]): str(row["classification"])
        for row in issues
        if str(row.get("classification")) not in NON_BLOCKING_ISSUE_CLASSIFICATIONS
    }


def _payload(payloads: Mapping[str, JsonDict], source: UpstreamArtifact) -> JsonDict | None:
    return payloads.get(_path_key(source))


def _clean_artifact(
    payloads: Mapping[str, JsonDict],
    issues_by_path: Mapping[str, str],
    source: UpstreamArtifact,
) -> bool:
    return _path_key(source) in payloads and _path_key(source) not in issues_by_path


def _nested_mapping(payload: JsonMap | None, field: str) -> JsonDict:
    value = _payload_value(payload, field, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _gate_table(payloads: Mapping[str, JsonDict], issues: Sequence[JsonMap]) -> list[JsonDict]:
    issues_by_path = _issue_by_path(issues)
    exp5350 = _payload(payloads, EXP5350)
    exp5351 = _payload(payloads, EXP5351)
    exp5352 = _payload(payloads, EXP5352)
    exp5353 = _payload(payloads, EXP5353)
    exp5354 = _payload(payloads, EXP5354)
    exp5355 = _payload(payloads, EXP5355)
    exp5356 = _payload(payloads, EXP5356)
    exp5357 = _payload(payloads, EXP5357)
    exp5358 = _payload(payloads, EXP5358)
    exp5359 = _payload(payloads, EXP5359)
    exp5360 = _payload(payloads, EXP5360)
    exp5361 = _payload(payloads, EXP5361)

    source_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5350)
        and _status(exp5350) == "complete"
        and int(_payload_value(exp5350, "new_actionable_findings_count", -1)) >= 0
        and _payload_value(exp5350, "retired_scope_reopened") is False
        and _payload_value(exp5350, "executable_plan_change_required") is False
    )
    structured_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5351)
        and _payload_value(exp5351, "structured_protocol_clean") is True
    )
    constraint_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5352)
        and _payload_value(exp5352, "constraint_tax_panel_ready") is True
    )
    tokenprob_rows_ready = (
        _path_key(EXP5353) in payloads
        and _payload_value(exp5353, "tokenprob_feature_rows_ready") is True
    )
    carry_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5354)
        and _payload_value(exp5354, "carry_token_energy_signal_ready") is True
        and _payload_value(exp5354, "flagged_adversarial") is not True
    )
    dependency_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5355)
        and _payload_value(exp5355, "dependency_provenance_ready") is True
        and _payload_value(exp5355, "unsafe_false_accepts", 1) == 0
        and _payload_value(exp5355, "no_weight_mutation") is True
    )
    memory_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5356)
        and _payload_value(exp5356, "memory_tool_drift_ready") is True
        and _payload_value(exp5356, "unsafe_false_accepts", 1) == 0
        and _payload_value(exp5356, "no_weight_mutation") is True
    )
    scaleup_source_gate = _payload_value(exp5357, "source_gate", {})
    scaleup_ready = (
        dependency_ready
        and memory_ready
        and _clean_artifact(payloads, issues_by_path, EXP5357)
        and _payload_value(exp5357, "self_learning_scaleup_ready") is True
        and isinstance(scaleup_source_gate, Mapping)
        and scaleup_source_gate.get("all_passed") is True
        and _payload_value(exp5357, "unsafe_false_accepts", 1) == 0
        and _payload_value(exp5357, "no_weight_mutation") is True
        and _payload_value(exp5357, "duplicated_metric_pairs", []) == []
    )
    solver_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5358)
        and _payload_value(exp5358, "solver_projection_ready") is True
        and _payload_value(exp5358, "solver_authoritative") is True
        and _payload_value(exp5358, "fallback_completeness_rate") == 1.0
        and _payload_value(exp5358, "unsafe_false_accepts", 1) == 0
        and _payload_value(exp5358, "readiness_blockers", []) == []
    )
    pbit_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5359)
        and _payload_value(exp5359, "pbit_schedule_signal_ready") is True
        and _payload_value(exp5359, "hardware_speedup_claim") is False
        and _payload_value(exp5359, "false_accept_count", 1) == 0
    )
    arc_new_level_banked = (
        _clean_artifact(payloads, issues_by_path, EXP5360)
        and _payload_value(exp5360, "new_level_banked") is True
        and _payload_value(exp5360, "offline_reproduced") is True
    )
    kv260 = _nested_mapping(exp5361, "kv260_status")
    polarfire = _nested_mapping(exp5361, "polarfire_status")
    gatemate = _nested_mapping(exp5361, "gatemate_status")
    hardware_ready = (
        _clean_artifact(payloads, issues_by_path, EXP5361)
        and _payload_value(exp5361, "speedup_claim") is False
        and _payload_value(exp5361, "authenticated_workload_run") is True
        and _payload_value(exp5361, "polarfire_workload_validated") is True
        and _payload_value(exp5361, "no_host_block_device_evidence") is True
    )

    return [
        {
            "gate": "source_delta",
            "source_artifacts": [str(EXP5350.relative_path)],
            "ready": source_ready,
            "classification": (
                "source_delta_complete_plan_unchanged" if source_ready else "source_delta_not_clean"
            ),
            "claim_boundary": "source delta only; no execution result claim",
            "evidence": {
                "new_actionable_findings_count": _payload_value(
                    exp5350, "new_actionable_findings_count"
                ),
                "executable_plan_change_required": _payload_value(
                    exp5350, "executable_plan_change_required"
                ),
                "retired_scope_reopened": _payload_value(exp5350, "retired_scope_reopened"),
                "references_modified": _payload_value(exp5350, "references_modified"),
            },
        },
        {
            "gate": "structured_protocol",
            "source_artifacts": [str(EXP5351.relative_path)],
            "ready": structured_ready,
            "classification": (
                "structured_protocol_clean"
                if structured_ready
                else "blocked_structured_protocol_clean_false"
            ),
            "claim_boundary": "formatting protocol only; no answer-quality claim",
            "evidence": {
                "structured_protocol_clean": _payload_value(exp5351, "structured_protocol_clean"),
                "parse_success_rate": _payload_value(exp5351, "parse_success_rate"),
                "schema_success_rate": _payload_value(exp5351, "schema_success_rate"),
                "final_json_extraction_rate": _payload_value(exp5351, "final_json_extraction_rate"),
                "methodology_duration_s": _payload_value(exp5351, "methodology_duration_s"),
                "unsafe_false_accepts": _payload_value(exp5351, "unsafe_false_accepts"),
                "no_quality_claim": _payload_value(exp5351, "no_quality_claim"),
            },
        },
        {
            "gate": "constraint_tax_panel",
            "source_artifacts": [str(EXP5352.relative_path), str(EXP5351.relative_path)],
            "ready": constraint_ready,
            "classification": (
                "constraint_tax_panel_ready"
                if constraint_ready
                else "conductor_pre_gate_skipped"
                if _payload_value(exp5352, "blocked_at_layer") == "conductor_pre_gate"
                else "constraint_tax_panel_not_ready"
            ),
            "claim_boundary": "no constraint-tax metrics because the panel did not run",
            "evidence": {
                "blocked_at_layer": _payload_value(exp5352, "blocked_at_layer"),
                "gate_check_summary": _payload_value(exp5352, "gate_check_summary"),
                "gates_evaluated": _payload_value(exp5352, "gates_evaluated", []),
            },
        },
        {
            "gate": "tokenprob_feature_audit",
            "source_artifacts": [str(EXP5353.relative_path)],
            "ready": tokenprob_rows_ready,
            "classification": (
                "feature_rows_present_but_flagged_methodology"
                if tokenprob_rows_ready and _payload_value(exp5353, "flagged_adversarial") is True
                else "tokenprob_feature_rows_ready"
                if tokenprob_rows_ready
                else "tokenprob_feature_rows_not_ready"
            ),
            "claim_boundary": (
                "feature rows only; flagged methodology means no token-energy or quality claim"
            ),
            "evidence": {
                "tokenprob_feature_rows_ready": _payload_value(
                    exp5353, "tokenprob_feature_rows_ready"
                ),
                "tokenprob_feature_row_count": _payload_value(
                    exp5353, "tokenprob_feature_row_count"
                ),
                "per_token_logprob_available": _payload_value(
                    exp5353, "per_token_logprob_available"
                ),
                "topk_alternatives_available": _payload_value(
                    exp5353, "topk_alternatives_available"
                ),
                "flagged_adversarial": _payload_value(exp5353, "flagged_adversarial"),
                "corrigendum_pending": _payload_value(exp5353, "corrigendum_pending", []),
                "external_text_scorer_reopened": _payload_value(
                    exp5353, "external_text_scorer_reopened"
                ),
                "no_quality_claim": _payload_value(exp5353, "no_quality_claim"),
            },
        },
        {
            "gate": "carry_diagnostic",
            "source_artifacts": [str(EXP5354.relative_path), str(EXP5353.relative_path)],
            "ready": carry_ready,
            "classification": (
                "carry_token_energy_signal_ready"
                if carry_ready
                else "blocked_and_flagged_carry_signal_not_ready"
            ),
            "claim_boundary": (
                "bounded addition diagnostic only; no broad internal-energy or hallucination claim"
            ),
            "evidence": {
                "carry_token_energy_signal_ready": _payload_value(
                    exp5354, "carry_token_energy_signal_ready"
                ),
                "feature_complete_rate": _payload_value(exp5354, "feature_complete_rate"),
                "correct_vs_perturbed_margin": _payload_value(
                    exp5354, "correct_vs_perturbed_margin"
                ),
                "missing_feature_names": _payload_value(exp5354, "missing_feature_names", []),
                "flagged_adversarial": _payload_value(exp5354, "flagged_adversarial"),
                "corrigendum_pending": _payload_value(exp5354, "corrigendum_pending", []),
                "no_broad_hallucination_claim": _payload_value(
                    exp5354, "no_broad_hallucination_claim"
                ),
            },
        },
        {
            "gate": "dependency_provenance",
            "source_artifacts": [str(EXP5355.relative_path)],
            "ready": dependency_ready,
            "classification": (
                "dependency_provenance_ready" if dependency_ready else "dependency_not_ready"
            ),
            "claim_boundary": "deterministic context provenance only; no model-weight mutation",
            "evidence": {
                "dependency_edge_precision": _payload_value(exp5355, "dependency_edge_precision"),
                "dependency_edge_recall": _payload_value(exp5355, "dependency_edge_recall"),
                "execution_feedback_attribution_rate": _payload_value(
                    exp5355, "execution_feedback_attribution_rate"
                ),
                "unsafe_false_accepts": _payload_value(exp5355, "unsafe_false_accepts"),
                "no_weight_mutation": _payload_value(exp5355, "no_weight_mutation"),
            },
        },
        {
            "gate": "memory_tool_drift",
            "source_artifacts": [str(EXP5356.relative_path)],
            "ready": memory_ready,
            "classification": "memory_tool_drift_ready" if memory_ready else "drift_not_ready",
            "claim_boundary": "deterministic drift fixture only; no model-weight mutation",
            "evidence": {
                "drift_case_count": _payload_value(exp5356, "drift_case_count"),
                "clean_selection_accuracy": _payload_value(exp5356, "clean_selection_accuracy"),
                "induced_tool_drift_rate": _payload_value(exp5356, "induced_tool_drift_rate"),
                "rollback_recovery_rate": _payload_value(exp5356, "rollback_recovery_rate"),
                "unsafe_false_accepts": _payload_value(exp5356, "unsafe_false_accepts"),
                "no_weight_mutation": _payload_value(exp5356, "no_weight_mutation"),
            },
        },
        {
            "gate": "self_learning_scaleup",
            "source_artifacts": [
                str(EXP5355.relative_path),
                str(EXP5356.relative_path),
                str(EXP5357.relative_path),
            ],
            "ready": scaleup_ready,
            "classification": (
                "dependency_safe_self_learning_scaled"
                if scaleup_ready
                else "self_learning_scaleup_not_clean"
            ),
            "claim_boundary": (
                "deterministic process scale-up only; frozen-model discipline preserved"
            ),
            "evidence": {
                "multi_session_trace_count": _payload_value(exp5357, "multi_session_trace_count"),
                "dependency_attribution_rate": _payload_value(
                    exp5357, "dependency_attribution_rate"
                ),
                "drift_detection_rate": _payload_value(exp5357, "drift_detection_rate"),
                "quality_delta_vs_always_full": _payload_value(
                    exp5357, "quality_delta_vs_always_full"
                ),
                "duplicated_metric_pairs": _payload_value(exp5357, "duplicated_metric_pairs", []),
                "unsafe_false_accepts": _payload_value(exp5357, "unsafe_false_accepts"),
                "no_weight_mutation": _payload_value(exp5357, "no_weight_mutation"),
            },
        },
        {
            "gate": "solver_projection",
            "source_artifacts": [str(EXP5358.relative_path)],
            "ready": solver_ready,
            "classification": (
                "solver_projection_ready" if solver_ready else "solver_projection_not_ready"
            ),
            "claim_boundary": "solver remains authoritative; neural proposals are advisory",
            "evidence": {
                "solver_authoritative": _payload_value(exp5358, "solver_authoritative"),
                "fallback_completeness_rate": _payload_value(exp5358, "fallback_completeness_rate"),
                "projection_success_rate": _payload_value(exp5358, "projection_success_rate"),
                "post_projection_validity_rate": _payload_value(
                    exp5358, "post_projection_validity_rate"
                ),
                "unsafe_false_accepts": _payload_value(exp5358, "unsafe_false_accepts"),
                "readiness_blockers": _payload_value(exp5358, "readiness_blockers", []),
            },
        },
        {
            "gate": "pbit_schedules",
            "source_artifacts": [str(EXP5359.relative_path)],
            "ready": pbit_ready,
            "classification": (
                "cpu_pbit_schedule_signal_ready" if pbit_ready else "pbit_schedule_signal_not_ready"
            ),
            "claim_boundary": "CPU schedule diagnostic only; no hardware execution or speedup claim",
            "evidence": {
                "fixture_count": _payload_value(exp5359, "fixture_count"),
                "schedule_variant_count": _payload_value(exp5359, "schedule_variant_count"),
                "false_accept_count": _payload_value(exp5359, "false_accept_count"),
                "hardware_speedup_claim": _payload_value(exp5359, "hardware_speedup_claim"),
                "claim_limits": _payload_value(exp5359, "claim_limits", []),
            },
        },
        {
            "gate": "arc_level_up",
            "source_artifacts": [str(EXP5360.relative_path)],
            "ready": arc_new_level_banked,
            "classification": (
                "new_arc_level_banked"
                if arc_new_level_banked
                else "honest_null_no_new_level_banked"
            ),
            "claim_boundary": "mandatory live ARC slot; no progress banked without reproduction",
            "evidence": {
                "target_game": _payload_value(exp5360, "target_game"),
                "target_level": _payload_value(exp5360, "target_level"),
                "new_level_banked": _payload_value(exp5360, "new_level_banked"),
                "offline_reproduced": _payload_value(exp5360, "offline_reproduced"),
                "registry_total_before": _payload_value(exp5360, "registry_total_before"),
                "registry_total_after": _payload_value(exp5360, "registry_total_after"),
                "perception_error_classes": _payload_value(exp5360, "perception_error_classes", []),
            },
        },
        {
            "gate": "hardware_continuity",
            "source_artifacts": [str(EXP5361.relative_path)],
            "ready": hardware_ready,
            "classification": (
                "partial_continuity_polarfire_workload_kv260_gatemate_blocked_no_speedup"
                if hardware_ready
                and (
                    kv260.get("ssh_reachable") is not True
                    or str(gatemate.get("status", "")).startswith("blocked_")
                )
                else "hardware_continuity_workload_no_speedup"
                if hardware_ready
                else "hardware_continuity_not_ready"
            ),
            "claim_boundary": "hardware continuity and board-local smoke only; no speedup claim",
            "evidence": {
                "hardware_evidence_level": _payload_value(exp5361, "hardware_evidence_level"),
                "polarfire_workload_validated": _payload_value(
                    exp5361, "polarfire_workload_validated"
                ),
                "authenticated_workload_run": _payload_value(exp5361, "authenticated_workload_run"),
                "speedup_claim": _payload_value(exp5361, "speedup_claim"),
                "kv260_ssh_reachable": kv260.get("ssh_reachable"),
                "kv260_status": kv260.get("status"),
                "polarfire_status": polarfire.get("status"),
                "gatemate_status": gatemate.get("status"),
                "blocked_reason": _payload_value(exp5361, "blocked_reason", {}),
            },
        },
    ]


def gate_value(gates: Sequence[JsonMap], gate: str) -> bool:
    return any(row.get("gate") == gate and row.get("ready") is True for row in gates)


def next_milestone_recommendation(gates: Sequence[JsonMap]) -> JsonDict:
    """Choose the short next branch from the reconciled .488 gate state."""

    structured = next(row for row in gates if row["gate"] == "structured_protocol")
    constraint = next(row for row in gates if row["gate"] == "constraint_tax_panel")
    carry = next(row for row in gates if row["gate"] == "carry_diagnostic")
    arc = next(row for row in gates if row["gate"] == "arc_level_up")
    hardware = next(row for row in gates if row["gate"] == "hardware_continuity")
    return {
        "recommendation": "structured_sota_protocol_repair_then_constraint_tax_panel",
        "why": (
            "Exp5351 left structured_protocol_clean=false, so Exp5352 was "
            "conductor-gate-skipped and constraint-tax metrics are not usable."
        ),
        "secondary_priorities": [
            f"token_energy={carry['classification']}",
            "dependency_safe_self_learning_scaleup_ready=true",
            "solver_projection_and_cpu_pbit_schedules_ready_but_bounded",
            f"arc={arc['classification']}",
            f"hardware={hardware['classification']}",
        ],
        "do_not_claim": [
            "constraint_tax_metrics",
            "headline_sota_quality",
            "token_probability_energy_signal",
            "arc_new_level",
            "hardware_speedup",
        ],
        "blocked_by": [
            f"structured_protocol={structured['classification']}",
            f"constraint_tax_panel={constraint['classification']}",
        ],
    }


def cited_upstream_artifacts(artifacts_read: Sequence[JsonMap]) -> list[JsonDict]:
    """Return sha256 citation rows and imported fields used by the capstone."""

    citations: list[JsonDict] = []
    for row in artifacts_read:
        path = str(row["path"])
        citations.append(
            {
                "experiment_number": row["experiment_number"],
                "task_id": row["task_id"],
                "path": path,
                "sha256": row["sha256"],
                "imported_fields": CITED_FIELDS_BY_PATH.get(path, []),
            }
        )
    return citations


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def default_tests_run() -> list[JsonDict]:
    return [{"command": "validation pending at artifact generation", "outcome": "pending"}]


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    conductor_outcomes = read_conductor_outcomes(root)
    payloads, artifacts_read, missing_or_malformed = read_upstream_artifacts(
        root, conductor_outcomes
    )
    issues = _blocked_flagged_or_skipped_rows(
        root=root,
        payloads=payloads,
        missing_or_malformed=missing_or_malformed,
        conductor_outcomes=conductor_outcomes,
    )
    gates = _gate_table(payloads, issues)
    actual_input_missing_or_malformed = any(
        row["classification"] in {"missing", "malformed"} for row in missing_or_malformed
    )
    status = "blocked_missing_required" if actual_input_missing_or_malformed else "complete"
    verdict_prefix = (
        "blocked_missing_required:" if actual_input_missing_or_malformed else "complete:"
    )
    verdict = (
        f"{verdict_prefix} .488 synthesized with structured_protocol_clean="
        f"{gate_value(gates, 'structured_protocol')}, constraint_tax_panel_ready="
        f"{gate_value(gates, 'constraint_tax_panel')}, tokenprob_feature_rows_ready="
        f"{gate_value(gates, 'tokenprob_feature_audit')}, "
        f"carry_token_energy_signal_ready={gate_value(gates, 'carry_diagnostic')}, "
        f"dependency_safe_self_learning_scaled={gate_value(gates, 'self_learning_scaleup')}, "
        f"solver_projection_ready={gate_value(gates, 'solver_projection')}, "
        f"pbit_schedule_signal_ready={gate_value(gates, 'pbit_schedules')}, "
        f"arc_new_level_banked={gate_value(gates, 'arc_level_up')}, "
        "hardware_speedup_claim=false"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": wrapped("experiment_id", EXPERIMENT_ID),
        "milestone": wrapped("milestone", MILESTONE),
        "status": wrapped("status", status),
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "random_seed": RANDOM_SEED,
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": wrapped("honest_verdict", verdict),
        "inference_substrate": wrapped("inference_substrate", INFERENCE_SUBSTRATE),
        "artifacts_read": wrapped("artifacts_read", artifacts_read),
        "missing_blocked_flagged_or_skipped_artifacts": wrapped(
            "missing_blocked_flagged_or_skipped_artifacts", issues
        ),
        "gate_table": wrapped("gate_table", gates),
        "structured_protocol_clean": gate_value(gates, "structured_protocol"),
        "constraint_tax_panel_ready": gate_value(gates, "constraint_tax_panel"),
        "tokenprob_feature_rows_ready": gate_value(gates, "tokenprob_feature_audit"),
        "carry_token_energy_signal_ready": gate_value(gates, "carry_diagnostic"),
        "dependency_provenance_ready": gate_value(gates, "dependency_provenance"),
        "memory_tool_drift_ready": gate_value(gates, "memory_tool_drift"),
        "self_learning_scaleup_ready": gate_value(gates, "self_learning_scaleup"),
        "solver_projection_ready": gate_value(gates, "solver_projection"),
        "pbit_schedule_signal_ready": gate_value(gates, "pbit_schedules"),
        "arc_new_level_banked": gate_value(gates, "arc_level_up"),
        "hardware_speedup_claim": False,
        "active_roadmap_modified": False,
        "conductor_modified": False,
        "next_milestone_recommendation": wrapped(
            "next_milestone_recommendation", next_milestone_recommendation(gates)
        ),
        "cited_upstream_artifacts": wrapped(
            "cited_upstream_artifacts", cited_upstream_artifacts(artifacts_read)
        ),
        "tests_run": wrapped(
            "tests_run",
            [dict(row) for row in (tests_run if tests_run is not None else default_tests_run())],
        ),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: JsonMap) -> None:
    missing = [field for field in REQUIRED_SCHEMA_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    for field in WRAPPED_FIELDS:
        value = artifact[field]
        if (
            not isinstance(value, Mapping)
            or value.get("principle") != FIELD_PRINCIPLES[field]
            or "value" not in value
        ):
            raise ValueError(f"{field} must be principle-wrapped")
    for field in BARE_BOOLEAN_FIELDS:
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare boolean")
    verdict = artifact["honest_verdict"]["value"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked_")
    if artifact["inference_substrate"]["value"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate drift")
    if artifact["hardware_speedup_claim"] is not False:
        raise ValueError("hardware_speedup_claim must be false")
    for field in ("active_roadmap_modified", "conductor_modified"):
        if artifact[field] is not False:
            raise ValueError(f"{field} must be false")
    gates = artifact["gate_table"]["value"]
    if not isinstance(gates, list) or [row.get("gate") for row in gates] != list(GATE_ORDER):
        raise ValueError("gate_table must preserve GATE_ORDER")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    artifact = build_result_artifact(root=root, tests_run=tests_run)
    validate_artifact(artifact)
    write_json(result_path or Path(root) / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    run(root=args.root, result_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
