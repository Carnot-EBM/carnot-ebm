"""Build the Exp5452 PRD gap and agent-failure table for milestone .495.

Spec refs: REQ-HARNESS-015, SCENARIO-HARNESS-010,
SCENARIO-HARNESS-011.

This helper is an aggregation step, not a new experiment runner. It reads the
landed Exp5441 through Exp5451 result artifacts, keeps every cited field tied
to a file on disk, and classifies bounded or null outcomes without rounding
them into success. That matters for capstone synthesis because the roadmap can
say what a task intended to do, but the PRD gap table must say only what the
artifacts actually support.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
Payloads = Mapping[str, JsonDict]
LaneClassifier = Callable[[Payloads], str]

MILESTONE = "2026.07.495"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_5452_prd_gap_agent_failure_table_v495.json")
SCHEMA = "carnot.prd_gap_agent_failure_table.v495.exp5452"
EXPERIMENT_ID = 5452
CLASSIFICATIONS = ("closed", "partial", "blocked", "honest_null", "missing")
TERMINAL_PREFIXES = ("complete:", "blocked:")
SPEC_REFS = ("REQ-HARNESS-015", "SCENARIO-HARNESS-010", "SCENARIO-HARNESS-011")

PRD_GOALS = (
    "verifiable reasoning",
    "continuous self-learning",
    "hardware acceleration readiness",
    "ARC progress",
    "model locality",
    "safety/traceability",
)

AGENT_FAILURE_PATTERNS = (
    "precondition block",
    "gate block",
    "no-bank",
    "measurement unavailable",
    "tautology risk",
    "unsupported claim",
    "implementation failure",
)

DEFAULT_UPSTREAM_PATHS = (
    "results/experiment_5441_transition_v495.json",
    "results/experiment_5442_source_delta_v495.json",
    "results/experiment_5443_verifier_potential_prefix_fixture_v495.json",
    "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json",
    "results/experiment_5445_static_ast_kb_witness_constraints_v495.json",
    "results/experiment_5446_governed_memory_csl_online_v495.json",
    "results/experiment_5447_gated_csl_memory_failure_stress_v495.json",
    "results/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.json",
    "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json",
    "results/experiment_5450_arc_measurement_access_live_levelup_v495.json",
    "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "milestone",
    "artifacts_expected",
    "artifacts_found",
    "closed_count",
    "partial_count",
    "blocked_count",
    "honest_null_count",
    "missing_count",
    "prd_gap_table",
    "agent_failure_table",
    "unsupported_claims_detected",
    "inference_substrate",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "route key",
    "artifacts_expected": "completeness basis",
    "artifacts_found": "evidence basis",
    "closed_count": "PRD progress",
    "partial_count": "bounded progress",
    "blocked_count": "blocker accounting",
    "honest_null_count": "null-result honesty",
    "missing_count": "no fabricated evidence",
    "prd_gap_table": "traceability",
    "agent_failure_table": "operational learning",
    "unsupported_claims_detected": "claim discipline",
    "inference_substrate": "no hidden live model inference",
    "honest_verdict": "terminal status; start with complete: or blocked:",
}


@dataclass(frozen=True)
class LaneSpec:
    """One PRD lane plus the exact upstream fields needed to classify it."""

    name: str
    artifact_fields: Mapping[str, tuple[str, ...]]
    prd_goals: tuple[str, ...]
    classification_reason: str
    claim_boundary: str
    agent_failure_patterns: tuple[str, ...]
    next_action: str
    classifier: LaneClassifier


def _read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_upstreams(root: Path) -> tuple[dict[str, JsonDict], list[str], list[str]]:
    payloads: dict[str, JsonDict] = {}
    found: list[str] = []
    missing: list[str] = []
    for rel_path in DEFAULT_UPSTREAM_PATHS:
        path = root / rel_path
        if path.exists():
            payloads[rel_path] = _read_json_object(path)
            found.append(rel_path)
        else:
            missing.append(rel_path)
    return payloads, found, missing


def _value(payloads: Payloads, artifact_path: str, field_name: str) -> Any:
    return payloads.get(artifact_path, {}).get(field_name)


def _field_record(payloads: Payloads, artifact_path: str, field_name: str) -> JsonDict:
    payload = payloads.get(artifact_path, {})
    return {
        "artifact_path": artifact_path,
        "field_name": field_name,
        "present": field_name in payload,
        "value": payload.get(field_name),
    }


def _supporting_fields(payloads: Payloads, spec: LaneSpec) -> list[JsonDict]:
    return [
        _field_record(payloads, artifact_path, field_name)
        for artifact_path, field_names in spec.artifact_fields.items()
        if artifact_path in payloads
        for field_name in field_names
    ]


def _missing_supporting_fields(supporting_fields: list[JsonDict]) -> list[str]:
    return [
        f"{field['artifact_path']}.{field['field_name']}"
        for field in supporting_fields
        if field["present"] is not True
    ]


def _is_complete_verdict(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("complete:")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _at_least(value: Any, threshold: float) -> bool:
    return _is_number(value) and float(value) >= threshold


def _equals_number(value: Any, expected: float) -> bool:
    return _is_number(value) and float(value) == expected


def _compact(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True)
    return text if len(text) <= 96 else f"{text[:93]}..."


def _evidence_citation(supporting_fields: list[JsonDict]) -> str:
    present = [field for field in supporting_fields if field["present"] is True]
    cited = present[:3]
    if not cited:
        return "missing supporting artifact fields"
    return "; ".join(
        f"{field['artifact_path']}:{field['field_name']}={_compact(field['value'])}"
        for field in cited
    )


def _upstream_verdicts(payloads: Payloads, artifact_paths: Mapping[str, tuple[str, ...]]) -> list[str]:
    verdicts: list[str] = []
    for artifact_path in artifact_paths:
        value = _value(payloads, artifact_path, "honest_verdict")
        if isinstance(value, str):
            verdicts.append(value)
    return verdicts


def _transition_classifier(payloads: Payloads) -> str:
    complete = _is_complete_verdict(
        _value(payloads, "results/experiment_5441_transition_v495.json", "honest_verdict")
    )
    next_range = _value(payloads, "results/experiment_5441_transition_v495.json", "next_task_range")
    return "closed" if complete and next_range == "exp5441-exp5453" else "blocked"


def _source_delta_classifier(payloads: Payloads) -> str:
    complete = _is_complete_verdict(
        _value(payloads, "results/experiment_5442_source_delta_v495.json", "honest_verdict")
    )
    retired_reopened = (
        _value(payloads, "results/experiment_5442_source_delta_v495.json", "retired_scopes_reopened")
        is True
    )
    count = _value(
        payloads,
        "results/experiment_5442_source_delta_v495.json",
        "new_actionable_findings_count",
    )
    return "closed" if complete and not retired_reopened and _at_least(count, 0.0) else "blocked"


def _verifier_potential_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5443_verifier_potential_prefix_fixture_v495.json"
    ready = _value(payloads, path, "verifier_potential_fixture_ready") is True
    final_authority = _value(payloads, path, "exact_final_authority") is True
    independent = _value(payloads, path, "metric_independence_checks_passed") is True
    disagreements = _value(payloads, path, "prefix_final_disagreement_cases")
    return "closed" if ready and final_authority and independent and _at_least(disagreements, 1) else "partial"


def _local_sota_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json"
    ready = _value(payloads, path, "verifier_guided_decoding_ready") is True
    gpu = _value(payloads, path, "gpu_offload_verified") is True
    flagged = _value(payloads, path, "flagged_adversarial") is True
    pending = bool(_value(payloads, path, "corrigendum_pending"))
    delta_vs_grammar = _value(payloads, path, "guided_validity_delta_vs_grammar_only")
    if ready and gpu and not flagged and not pending and _at_least(delta_vs_grammar, 0.0):
        return "closed"
    if ready and gpu:
        return "partial"
    return "blocked"


def _ast_kb_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5445_static_ast_kb_witness_constraints_v495.json"
    ready = _value(payloads, path, "ast_kb_witness_ready") is True
    parsed = _equals_number(_value(payloads, path, "ast_parse_success_rate"), 1.0)
    valid_accept = _equals_number(_value(payloads, path, "valid_call_accept_rate"), 1.0)
    invalid_reject = _equals_number(_value(payloads, path, "nonexistent_call_reject_rate"), 1.0)
    unsafe_zero = _value(payloads, path, "unsafe_false_accepts") == 0
    return "closed" if ready and parsed and valid_accept and invalid_reject and unsafe_zero else "partial"


def _governed_memory_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5446_governed_memory_csl_online_v495.json"
    ready = _value(payloads, path, "governed_csl_loop_ready") is True
    replay = _equals_number(_value(payloads, path, "replay_success_rate"), 1.0)
    unsafe_zero = _value(payloads, path, "unsafe_false_accepts") == 0
    no_weight = _value(payloads, path, "no_weight_mutation") is True
    rollback = _equals_number(_value(payloads, path, "rollback_recovery_rate"), 1.0)
    return "closed" if ready and replay and unsafe_zero and no_weight and rollback else "partial"


def _memory_stress_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5447_gated_csl_memory_failure_stress_v495.json"
    ready = _value(payloads, path, "csl_memory_stress_ready") is True
    gate_ready = _value(payloads, path, "gated_upstream_ready") is True
    unsafe_zero = _value(payloads, path, "unsafe_false_accepts") == 0
    no_weight = _value(payloads, path, "no_weight_mutation") is True
    deflections = all(
        _equals_number(_value(payloads, path, field_name), 1.0)
        for field_name in (
            "rollback_recovery_rate",
            "stale_memory_deflection_rate",
            "poisoned_memory_deflection_rate",
            "retrieval_collision_deflection_rate",
            "negative_transfer_deflection_rate",
        )
    )
    return "closed" if ready and gate_ready and unsafe_zero and no_weight and deflections else "partial"


def _pbit_bridge_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.json"
    ready = _value(payloads, path, "pbit_assumption_bridge_ready") is True
    solver = _value(payloads, path, "solver_authoritative") is True
    fallback = _equals_number(_value(payloads, path, "fallback_completeness_rate"), 1.0)
    return "partial" if ready and solver and fallback else "blocked"


def _hardware_receipts_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json"
    ready = _value(payloads, path, "hardware_receipts_ready") is True
    gate_ready = _value(payloads, path, "gated_upstream_ready") is True
    hashes = _value(payloads, path, "hashes_match_before_timing_compare") is True
    return "partial" if ready and gate_ready and hashes else "blocked"


def _arc_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5450_arc_measurement_access_live_levelup_v495.json"
    banked = _value(payloads, path, "arc_new_level_banked") is True
    return "closed" if banked else "honest_null"


def _kan_certificate_classifier(payloads: Payloads) -> str:
    path = "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json"
    ready = _value(payloads, path, "kan_certificate_ready") is True
    gated = _value(payloads, path, "gated_upstreams_ready") is True
    true_rate = _equals_number(_value(payloads, path, "true_measured_claim_preservation_rate"), 1.0)
    false_rate = _equals_number(_value(payloads, path, "false_property_rejection_rate"), 1.0)
    unsupported_rate = _equals_number(
        _value(payloads, path, "unsupported_claim_rejection_rate"), 1.0
    )
    broad_claim = _value(payloads, path, "broad_kan_claim_made") is True
    return "partial" if ready and gated and true_rate and false_rate and unsupported_rate and not broad_claim else "blocked"


def _token_internal_classifier(payloads: Payloads) -> str:
    certificate_path = "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json"
    transition_path = "results/experiment_5441_transition_v495.json"
    token_rejected = _value(payloads, certificate_path, "token_internal_claim_rejected") is True
    blocked_lanes = _value(payloads, transition_path, "blocked_lanes")
    transition_blocked = "token_internal" in json.dumps(blocked_lanes, sort_keys=True)
    return "blocked" if token_rejected or transition_blocked else "missing"


def _hardware_speedup_classifier(payloads: Payloads) -> str:
    speedup_claimed = any(
        _value(payloads, path, "hardware_speedup_claim") is True
        for path in (
            "results/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.json",
            "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json",
        )
    )
    return "closed" if speedup_claimed else "honest_null"


LANE_SPECS = (
    LaneSpec(
        name="transition_traceability",
        artifact_fields={
            "results/experiment_5441_transition_v495.json": (
                "honest_verdict",
                "closed_lanes",
                "partial_lanes",
                "blocked_lanes",
                "honest_null_lanes",
                "next_task_range",
            )
        },
        prd_goals=("safety/traceability",),
        classification_reason="v495_transition_receipt_complete",
        claim_boundary="closed for route and prior-truth traceability only",
        agent_failure_patterns=(),
        next_action="Use as route context; do not treat prior bounded lanes as new wins.",
        classifier=_transition_classifier,
    ),
    LaneSpec(
        name="source_delta_traceability",
        artifact_fields={
            "results/experiment_5442_source_delta_v495.json": (
                "honest_verdict",
                "sources_checked",
                "new_actionable_findings_count",
                "new_references_added",
                "retired_scopes_reopened",
                "research_references_updated",
            )
        },
        prd_goals=("safety/traceability", "verifiable reasoning"),
        classification_reason="execution_time_source_delta_closed",
        claim_boundary="closed for literature/source refresh; not implementation success by itself",
        agent_failure_patterns=(),
        next_action="Carry forward the actionable hooks only where downstream artifacts used them.",
        classifier=_source_delta_classifier,
    ),
    LaneSpec(
        name="verifier_potential_reasoning",
        artifact_fields={
            "results/experiment_5443_verifier_potential_prefix_fixture_v495.json": (
                "honest_verdict",
                "verifier_potential_fixture_ready",
                "exact_final_authority",
                "prefix_final_disagreement_cases",
                "metric_independence_checks_passed",
            )
        },
        prd_goals=("verifiable reasoning", "safety/traceability"),
        classification_reason="deterministic_prefix_fixture_with_final_authority",
        claim_boundary="closed for deterministic verifier-potential fixtures, not learned truth",
        agent_failure_patterns=(),
        next_action="Keep exact final verifier authority in later generation pilots.",
        classifier=_verifier_potential_classifier,
    ),
    LaneSpec(
        name="local_sota_decoding_pilot",
        artifact_fields={
            "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json": (
                "honest_verdict",
                "verifier_guided_decoding_ready",
                "flagged_adversarial",
                "corrigendum_pending",
                "guided_validity_delta_vs_grammar_only",
                "guided_validity_delta_vs_unconstrained",
                "gpu_offload_verified",
                "runtime_backend",
                "precondition_details",
                "model_specs",
            )
        },
        prd_goals=("verifiable reasoning", "model locality", "safety/traceability"),
        classification_reason="bounded_local_sota_pilot_with_tautology_risk",
        claim_boundary="partial because the pilot is local/GPU-backed but flagged and worse than grammar-only",
        agent_failure_patterns=("tautology risk",),
        next_action="Repair the tautology flags before using this as structured-decoding progress.",
        classifier=_local_sota_classifier,
    ),
    LaneSpec(
        name="ast_kb_witness_constraints",
        artifact_fields={
            "results/experiment_5445_static_ast_kb_witness_constraints_v495.json": (
                "honest_verdict",
                "ast_kb_witness_ready",
                "ast_parse_success_rate",
                "valid_call_accept_rate",
                "nonexistent_call_reject_rate",
                "unsafe_false_accepts",
            )
        },
        prd_goals=("verifiable reasoning", "safety/traceability"),
        classification_reason="deterministic_code_api_witnesses_closed",
        claim_boundary="closed for deterministic AST/KB witness rows",
        agent_failure_patterns=(),
        next_action="Use as deterministic witness evidence, not as post-training quality evidence.",
        classifier=_ast_kb_classifier,
    ),
    LaneSpec(
        name="governed_online_memory_csl",
        artifact_fields={
            "results/experiment_5446_governed_memory_csl_online_v495.json": (
                "honest_verdict",
                "governed_csl_loop_ready",
                "replay_success_rate",
                "unsafe_false_accepts",
                "no_weight_mutation",
                "rollback_recovery_rate",
                "negative_transfer_deflection_rate",
                "quality_delta_vs_always_full",
                "context_efficiency_delta",
            )
        },
        prd_goals=("continuous self-learning", "safety/traceability"),
        classification_reason="governed_memory_closed_without_weight_mutation",
        claim_boundary="closed for sidecar memory promotion; no model-weight self-training claim",
        agent_failure_patterns=(),
        next_action="Carry forward as governed memory evidence with the no-weight boundary intact.",
        classifier=_governed_memory_classifier,
    ),
    LaneSpec(
        name="csl_memory_failure_stress",
        artifact_fields={
            "results/experiment_5447_gated_csl_memory_failure_stress_v495.json": (
                "honest_verdict",
                "csl_memory_stress_ready",
                "gated_upstream_ready",
                "unsafe_false_accepts",
                "no_weight_mutation",
                "rollback_recovery_rate",
                "stale_memory_deflection_rate",
                "poisoned_memory_deflection_rate",
                "retrieval_collision_deflection_rate",
                "negative_transfer_deflection_rate",
            )
        },
        prd_goals=("continuous self-learning", "safety/traceability"),
        classification_reason="memory_failure_stress_closed",
        claim_boundary="closed for governed failure deflection and rollback",
        agent_failure_patterns=(),
        next_action="Keep failure attribution by operation in the capstone.",
        classifier=_memory_stress_classifier,
    ),
    LaneSpec(
        name="pbit_assumption_bridge",
        artifact_fields={
            "results/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.json": (
                "honest_verdict",
                "pbit_assumption_bridge_ready",
                "solver_authoritative",
                "fallback_completeness_rate",
                "hardware_speedup_claim",
                "claim_limits",
            )
        },
        prd_goals=("hardware acceleration readiness", "verifiable reasoning"),
        classification_reason="advisory_assumptions_with_solver_authority",
        claim_boundary="partial because p-bit assumptions are advisory and no hardware speedup is claimed",
        agent_failure_patterns=(),
        next_action="Treat as solver-authoritative bridge evidence only.",
        classifier=_pbit_bridge_classifier,
    ),
    LaneSpec(
        name="hardware_timing_receipts",
        artifact_fields={
            "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json": (
                "honest_verdict",
                "hardware_receipts_ready",
                "gated_upstream_ready",
                "hashes_match_before_timing_compare",
                "hardware_speedup_claim",
                "board_reachability",
                "timing_repeat_counts",
                "timing_summary",
                "timing_comparison",
            )
        },
        prd_goals=("hardware acceleration readiness", "safety/traceability"),
        classification_reason="hash_matched_receipts_with_unavailable_boards_and_no_speedup",
        claim_boundary="partial because timing receipts exist but KV260/GateMate were unavailable and no speedup is claimed",
        agent_failure_patterns=("measurement unavailable",),
        next_action="Report timing facts only; do not claim acceleration.",
        classifier=_hardware_receipts_classifier,
    ),
    LaneSpec(
        name="arc_live_progress",
        artifact_fields={
            "results/experiment_5450_arc_measurement_access_live_levelup_v495.json": (
                "honest_verdict",
                "status",
                "arc_new_level_banked",
                "new_levels_banked",
                "new_level_reproduced",
                "reproduction_gate",
                "residual_wall",
                "live_attempt_count",
            )
        },
        prd_goals=("ARC progress", "verifiable reasoning"),
        classification_reason="bounded_budget_no_new_level_banked",
        claim_boundary="honest null: live path ran but no reproduction-gated new ARC level was banked",
        agent_failure_patterns=("no-bank",),
        next_action="Keep registry totals unchanged and preserve the residual wall.",
        classifier=_arc_classifier,
    ),
    LaneSpec(
        name="kan_measurement_certificate",
        artifact_fields={
            "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json": (
                "honest_verdict",
                "kan_certificate_ready",
                "gated_upstreams_ready",
                "claim_count",
                "true_measured_claim_preservation_rate",
                "false_property_rejection_rate",
                "unsupported_claim_rejection_rate",
                "hardware_speedup_claim_rejected",
                "token_internal_claim_rejected",
                "broad_kan_claim_made",
                "claim_limits",
                "claim_records",
            )
        },
        prd_goals=("verifiable reasoning", "safety/traceability"),
        classification_reason="bounded_measurement_access_certificate",
        claim_boundary="partial because the certificate is bounded and rejects broad KAN, hardware, and token/internal claims",
        agent_failure_patterns=("unsupported claim",),
        next_action="Carry forward the measurement-access boundary and rejected unsupported claims.",
        classifier=_kan_certificate_classifier,
    ),
    LaneSpec(
        name="token_internal_feature_access",
        artifact_fields={
            "results/experiment_5441_transition_v495.json": (
                "honest_verdict",
                "blocked_lanes",
            ),
            "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json": (
                "honest_verdict",
                "token_internal_claim_rejected",
                "claim_records",
                "claim_limits",
            ),
        },
        prd_goals=("verifiable reasoning", "safety/traceability"),
        classification_reason="authenticated_token_internal_receipts_absent",
        claim_boundary="blocked because token logprob and hidden/internal activation access remains unsupported",
        agent_failure_patterns=("measurement unavailable", "unsupported claim"),
        next_action="Keep token/internal claims blocked until authenticated backend receipts exist.",
        classifier=_token_internal_classifier,
    ),
    LaneSpec(
        name="hardware_speedup_claim",
        artifact_fields={
            "results/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.json": (
                "honest_verdict",
                "hardware_speedup_claim",
            ),
            "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json": (
                "honest_verdict",
                "hardware_speedup_claim",
                "timing_comparison",
            ),
            "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json": (
                "honest_verdict",
                "hardware_speedup_claim_rejected",
            ),
        },
        prd_goals=("hardware acceleration readiness",),
        classification_reason="no_hardware_speedup_claim_supported",
        claim_boundary="honest null: hardware receipts/certificates explicitly do not support speedup",
        agent_failure_patterns=("measurement unavailable", "unsupported claim"),
        next_action="State no-speedup plainly in the capstone.",
        classifier=_hardware_speedup_classifier,
    ),
)


def _lane_entry(
    spec: LaneSpec,
    classification: str,
    payloads: Payloads,
    missing_artifacts: list[str],
    missing_supporting_fields: list[str],
    supporting_fields: list[JsonDict],
) -> JsonDict:
    return {
        "lane": spec.name,
        "classification": classification,
        "classification_reason": spec.classification_reason,
        "prd_goals": list(spec.prd_goals),
        "claim_boundary": spec.claim_boundary,
        "artifact_paths": list(spec.artifact_fields),
        "missing_artifacts": missing_artifacts,
        "missing_supporting_fields": missing_supporting_fields,
        "supporting_fields": supporting_fields,
        "evidence_citation": _evidence_citation(supporting_fields),
        "agent_failure_patterns": list(spec.agent_failure_patterns),
        "next_action": spec.next_action,
        "upstream_honest_verdicts": _upstream_verdicts(payloads, spec.artifact_fields),
    }


def _prd_gap_table(payloads: Payloads) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for spec in LANE_SPECS:
        supporting_fields = _supporting_fields(payloads, spec)
        missing_artifacts = [
            artifact_path for artifact_path in spec.artifact_fields if artifact_path not in payloads
        ]
        missing_supporting_fields = _missing_supporting_fields(supporting_fields)
        classification = (
            "missing"
            if missing_artifacts or missing_supporting_fields
            else spec.classifier(payloads)
        )
        rows.append(
            _lane_entry(
                spec,
                classification,
                payloads,
                missing_artifacts,
                missing_supporting_fields,
                supporting_fields,
            )
        )
    return rows


def _classification_counts(rows: list[JsonDict]) -> dict[str, int]:
    counter: Counter[str] = Counter(str(row["classification"]) for row in rows)
    return {classification: counter.get(classification, 0) for classification in CLASSIFICATIONS}


def _unsupported_claims(payloads: Payloads) -> list[JsonDict]:
    path = "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json"
    records = _value(payloads, path, "claim_records")
    if not isinstance(records, list):
        return []
    unsupported: list[JsonDict] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        claim_kind = record.get("claim_kind")
        classification = record.get("classification")
        if claim_kind not in {"unsupported", "broad_soundness"} and classification != "missing_evidence_unsupported":
            continue
        unsupported.append(
            {
                "source_artifact": path,
                "claim_id": str(record.get("claim_id", "")),
                "claim_kind": str(claim_kind),
                "statement": str(record.get("statement", "")),
                "classification": str(classification or "missing_evidence_unsupported"),
                "rejected": record.get("rejected") is True,
                "missing_evidence": list(record.get("missing_evidence") or []),
            }
        )
    return unsupported


def _board_measurement_unavailable(payloads: Payloads) -> bool:
    path = "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json"
    board_reachability = _value(payloads, path, "board_reachability")
    if not isinstance(board_reachability, Mapping):
        return False
    return any(
        isinstance(receipt, Mapping) and receipt.get("reachable") is False
        for receipt in board_reachability.values()
    )


def _precondition_block_observed(payloads: Payloads) -> bool:
    path = "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json"
    details = _value(payloads, path, "precondition_details")
    if not isinstance(details, Mapping):
        return False
    return details.get("all_passed") is False or bool(details.get("blocked_preconditions"))


def _gate_block_observed(payloads: Payloads) -> bool:
    gate_fields = (
        ("results/experiment_5447_gated_csl_memory_failure_stress_v495.json", "gated_upstream_ready"),
        ("results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json", "gated_upstream_ready"),
        ("results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json", "gated_upstreams_ready"),
    )
    return any(path in payloads and _value(payloads, path, field_name) is False for path, field_name in gate_fields)


def _implementation_failure_observed(payloads: Payloads) -> bool:
    for payload in payloads.values():
        status = payload.get("status")
        verdict = payload.get("honest_verdict")
        if status in {"failed", "implementation_failure"}:
            return True
        if isinstance(verdict, str) and verdict.startswith("failed:"):
            return True
    return False


def _agent_failure_row(
    pattern: str,
    observed: bool,
    classification: str,
    artifact_paths: list[str],
    affected_lanes: list[str],
    evidence_citation: str,
) -> JsonDict:
    return {
        "pattern": pattern,
        "observed": observed,
        "classification": classification,
        "artifact_paths": artifact_paths,
        "affected_lanes": affected_lanes,
        "evidence_citation": evidence_citation,
    }


def _agent_failure_table(payloads: Payloads, prd_rows: list[JsonDict]) -> list[JsonDict]:
    lanes_by_pattern = {
        pattern: [
            str(row["lane"])
            for row in prd_rows
            if pattern in row.get("agent_failure_patterns", [])
        ]
        for pattern in AGENT_FAILURE_PATTERNS
    }
    unsupported = _unsupported_claims(payloads)
    no_bank = _value(
        payloads,
        "results/experiment_5450_arc_measurement_access_live_levelup_v495.json",
        "arc_new_level_banked",
    ) is False
    tautology = _value(
        payloads,
        "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json",
        "flagged_adversarial",
    ) is True or bool(
        _value(
            payloads,
            "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json",
            "corrigendum_pending",
        )
    )
    measurement_unavailable = _board_measurement_unavailable(payloads) or bool(unsupported)
    precondition_block = _precondition_block_observed(payloads)
    gate_block = _gate_block_observed(payloads)
    implementation_failure = _implementation_failure_observed(payloads)

    return [
        _agent_failure_row(
            "precondition block",
            precondition_block,
            "blocked" if precondition_block else "closed",
            ["results/experiment_5444_gated_sota_energy_guided_decoding_v495.json"],
            lanes_by_pattern["precondition block"],
            "Exp5444 precondition_details.blocked_preconditions records whether generation was blocked before run.",
        ),
        _agent_failure_row(
            "gate block",
            gate_block,
            "blocked" if gate_block else "closed",
            [
                "results/experiment_5447_gated_csl_memory_failure_stress_v495.json",
                "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json",
                "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json",
            ],
            lanes_by_pattern["gate block"],
            "Recorded gated-upstream fields are ready when present; no active gate block is inferred from roadmap text.",
        ),
        _agent_failure_row(
            "no-bank",
            no_bank,
            "honest_null" if no_bank else "closed",
            ["results/experiment_5450_arc_measurement_access_live_levelup_v495.json"],
            lanes_by_pattern["no-bank"],
            "Exp5450 arc_new_level_banked=false and residual_wall=bounded_budget_no_levelup.",
        ),
        _agent_failure_row(
            "measurement unavailable",
            measurement_unavailable,
            "partial" if measurement_unavailable else "closed",
            [
                "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json",
                "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json",
            ],
            lanes_by_pattern["measurement unavailable"],
            "Exp5449 has unreachable KV260/GateMate receipts and Exp5451 rejects missing-evidence token/internal claims.",
        ),
        _agent_failure_row(
            "tautology risk",
            tautology,
            "partial" if tautology else "closed",
            ["results/experiment_5444_gated_sota_energy_guided_decoding_v495.json"],
            lanes_by_pattern["tautology risk"],
            "Exp5444 flagged_adversarial/corrigendum_pending records tautology risk.",
        ),
        _agent_failure_row(
            "unsupported claim",
            bool(unsupported),
            "blocked" if unsupported else "closed",
            ["results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json"],
            lanes_by_pattern["unsupported claim"],
            "Exp5451 claim_records reject unsupported hardware, token/internal, and broad KAN claims.",
        ),
        _agent_failure_row(
            "implementation failure",
            implementation_failure,
            "blocked" if implementation_failure else "closed",
            list(DEFAULT_UPSTREAM_PATHS),
            lanes_by_pattern["implementation failure"],
            "No Exp5441-Exp5451 artifact reports status=failed or a failed: honest_verdict.",
        ),
    ]


def _honest_verdict(ready: bool, counts: Mapping[str, int], missing: list[str]) -> str:
    prefix = "complete:" if ready else "blocked:"
    reason = (
        "read actual Exp5441-Exp5451 artifacts"
        if ready
        else f"missing expected artifacts or fields: {missing}"
    )
    return (
        f"{prefix} .495 PRD gap table {reason}; closed={counts['closed']}, "
        f"partial={counts['partial']}, blocked={counts['blocked']}, "
        f"honest_null={counts['honest_null']}, missing={counts['missing']}."
    )


def build_report(root: Path | str) -> JsonDict:
    """Build the Exp5452 artifact from upstream JSON files under ``root``."""

    root_path = Path(root)
    payloads, artifacts_found, artifacts_missing = _load_upstreams(root_path)
    prd_rows = _prd_gap_table(payloads)
    counts = _classification_counts(prd_rows)
    ready = not artifacts_missing and counts["missing"] == 0
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "field_principles": FIELD_PRINCIPLES,
        "milestone": MILESTONE,
        "artifacts_expected": list(DEFAULT_UPSTREAM_PATHS),
        "artifacts_found": artifacts_found,
        "artifacts_missing": artifacts_missing,
        "closed_count": counts["closed"],
        "partial_count": counts["partial"],
        "blocked_count": counts["blocked"],
        "honest_null_count": counts["honest_null"],
        "missing_count": counts["missing"],
        "prd_gap_table": prd_rows,
        "agent_failure_table": _agent_failure_table(payloads, prd_rows),
        "unsupported_claims_detected": _unsupported_claims(payloads),
        "prd_gap_table_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, counts, artifacts_missing),
    }
    validate_artifact(report)
    return report


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp5452 schema fields that downstream capstones consume."""

    missing_fields = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(f"missing required artifact fields: {missing_fields}")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must be 2026.07.495")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    expected = artifact["artifacts_expected"]
    found = artifact["artifacts_found"]
    missing = artifact.get("artifacts_missing", [])
    if expected != list(DEFAULT_UPSTREAM_PATHS):
        raise ValueError("artifacts_expected does not match Exp5441-Exp5451")
    if sorted(found + missing) != sorted(expected):
        raise ValueError("artifacts_found plus artifacts_missing must equal artifacts_expected")

    prd_rows = artifact["prd_gap_table"]
    if not isinstance(prd_rows, list) or not prd_rows:
        raise ValueError("prd_gap_table must be a non-empty list")
    counts = _classification_counts([dict(row) for row in prd_rows])
    for classification in CLASSIFICATIONS:
        field_name = f"{classification}_count"
        if artifact[field_name] != counts[classification]:
            raise ValueError(f"{field_name} does not match prd_gap_table")
    for row in prd_rows:
        if row.get("classification") not in CLASSIFICATIONS:
            raise ValueError("prd_gap_table contains invalid classification")
        if not row.get("evidence_citation"):
            raise ValueError("prd_gap_table row missing evidence_citation")
        if not row.get("prd_goals"):
            raise ValueError("prd_gap_table row missing prd_goals")
        if row.get("classification") != "missing":
            if row.get("missing_artifacts") or row.get("missing_supporting_fields"):
                raise ValueError("non-missing lane has missing evidence")
            for field in row.get("supporting_fields", []):
                if field.get("present") is not True:
                    raise ValueError("non-missing lane cites absent supporting field")

    patterns = artifact["agent_failure_table"]
    if not isinstance(patterns, list):
        raise ValueError("agent_failure_table must be a list")
    observed_patterns = {row.get("pattern") for row in patterns}
    if observed_patterns != set(AGENT_FAILURE_PATTERNS):
        raise ValueError("agent_failure_table must cover every required pattern")

    unsupported_claims = artifact["unsupported_claims_detected"]
    if not isinstance(unsupported_claims, list):
        raise ValueError("unsupported_claims_detected must be a list")
    if any(row.get("rejected") is not True for row in unsupported_claims):
        raise ValueError("unsupported_claims_detected must contain rejected claims only")


def write_artifact(root: Path | str) -> Path:
    """Write the Exp5452 JSON artifact and return the output path."""

    root_path = Path(root)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_report(root_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":  # pragma: no cover - manual artifact writer
    write_artifact(Path.cwd())
