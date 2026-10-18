"""Build the Exp5439 PRD gap and agent-failure taxonomy table.

Spec refs: REQ-HARNESS-015, SCENARIO-HARNESS-010.

This module is deliberately an aggregation step. It does not run ARC, solver,
hardware, ontology, or model work again. It reads the landed .494 artifacts,
records exactly which files and fields were present, and writes a bounded table
for the .494 capstone. That provenance matters because a gap table is easy to
accidentally turn into a wish list; every lane here is anchored to a JSON field
that existed when the helper ran.
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

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_5439_prd_gap_agent_failure_table_v494.json")

DEFAULT_UPSTREAM_PATHS = (
    "results/experiment_5428_transition_v494.json",
    "results/experiment_5429_source_delta_v494.json",
    "results/experiment_5430_structured_tautology_corrigendum_v494.json",
    "results/experiment_5431_structured_constraint_taxonomy_replication_v494.json",
    "results/experiment_5432_ontology_softlogic_constraint_memory_v494.json",
    "results/experiment_5433_active_constraint_diversity_lns_v494.json",
    "results/experiment_5434_pbit_polarfire_timing_variance_v494.json",
    "results/experiment_5435_verified_workflow_memory_csl_v494.json",
    "results/experiment_5436_csl_memory_transfer_stress_v494.json",
    "results/experiment_5437_arc_live_reinduction_levelup_v494.json",
    "results/experiment_5438_kan_ontology_measurement_certificate_v494.json",
)


@dataclass(frozen=True)
class LaneSpec:
    """One PRD lane and the exact upstream fields needed to classify it."""

    name: str
    artifact_fields: Mapping[str, tuple[str, ...]]
    classification_reason: str
    claim_boundary: str
    failure_taxonomy: tuple[str, ...]
    prd_refs: tuple[str, ...]
    research_program_priorities: tuple[str, ...]
    next_action: str
    classifier: LaneClassifier


FIELD_PRINCIPLES = {
    "upstream_artifacts_read": "Provenance: the table can only cite files that existed.",
    "upstream_artifacts_missing": "No fabricated evidence: absent upstreams stay absent.",
    "closed_lanes": "PRD progress: only lanes backed by present upstream fields close.",
    "partial_lanes": "Bounded evidence: useful receipts without broad headline claims.",
    "blocked_lanes": "Honest gaps: unsupported lanes remain blocked instead of inferred.",
    "honest_null_lanes": "Null-result honesty: measured no-gain outcomes are explicit.",
    "missing_lanes": "Absent artifact handling: missing upstreams cannot close a lane.",
    "failure_taxonomy_counts": "Diagnosis: recurring tool, planning, reasoning, measurement, memory, ARC, and hardware gaps are counted.",
    "prd_gap_table_ready": "Capstone input: true only when all named upstream artifacts and required lane fields were present.",
    "inference_substrate": "No hidden live model inference: this helper only aggregates upstream JSON.",
    "honest_verdict": "Terminal status: starts with complete: or blocked: for conductor classification.",
}


def _read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_upstreams(root: Path) -> tuple[dict[str, JsonDict], list[str], list[str]]:
    payloads: dict[str, JsonDict] = {}
    read: list[str] = []
    missing: list[str] = []
    for rel_path in DEFAULT_UPSTREAM_PATHS:
        path = root / rel_path
        target = read if path.exists() else missing
        target.append(rel_path)
        if path.exists():
            payloads[rel_path] = _read_json_object(path)
    return payloads, read, missing


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
        "claim_boundary": spec.claim_boundary,
        "artifact_paths": list(spec.artifact_fields),
        "missing_artifacts": missing_artifacts,
        "missing_supporting_fields": missing_supporting_fields,
        "supporting_fields": supporting_fields,
        "failure_taxonomy": list(spec.failure_taxonomy),
        "prd_refs": list(spec.prd_refs),
        "research_program_priorities": list(spec.research_program_priorities),
        "next_action": spec.next_action,
        "upstream_honest_verdicts": [
            _value(payloads, artifact_path, "honest_verdict")
            for artifact_path in spec.artifact_fields
            if artifact_path in payloads and "honest_verdict" in payloads[artifact_path]
        ],
    }


def _structured_classifier(payloads: Payloads) -> str:
    clean = _value(
        payloads,
        "results/experiment_5430_structured_tautology_corrigendum_v494.json",
        "structured_corrigendum_clean",
    ) is True
    ready = _value(
        payloads,
        "results/experiment_5431_structured_constraint_taxonomy_replication_v494.json",
        "structured_taxonomy_replication_ready",
    ) is True
    independent = _value(
        payloads,
        "results/experiment_5431_structured_constraint_taxonomy_replication_v494.json",
        "metric_independence_checks_passed",
    ) is True
    risk = _value(
        payloads,
        "results/experiment_5431_structured_constraint_taxonomy_replication_v494.json",
        "accepted_risk_bound",
    )
    threshold = _value(
        payloads,
        "results/experiment_5431_structured_constraint_taxonomy_replication_v494.json",
        "accepted_risk_bound_threshold",
    )
    bounded = isinstance(risk, int | float) and isinstance(threshold, int | float) and risk <= threshold
    return "closed" if clean and ready and independent and bounded else "partial"


def _ontology_classifier(payloads: Payloads) -> str:
    ready = _value(
        payloads,
        "results/experiment_5432_ontology_softlogic_constraint_memory_v494.json",
        "ontology_constraint_memory_ready",
    ) is True
    overrode_solver = _value(
        payloads,
        "results/experiment_5432_ontology_softlogic_constraint_memory_v494.json",
        "soft_logic_overrode_solver",
    ) is True
    return "closed" if ready and not overrode_solver else "partial"


def _solver_classifier(payloads: Payloads) -> str:
    ready = _value(
        payloads,
        "results/experiment_5433_active_constraint_diversity_lns_v494.json",
        "active_constraint_diversity_ready",
    ) is True
    solver_preserved = _value(
        payloads,
        "results/experiment_5433_active_constraint_diversity_lns_v494.json",
        "solver_validity_preserved",
    ) is True
    return "partial" if ready and solver_preserved else "blocked"


def _hardware_classifier(payloads: Payloads) -> str:
    ready = _value(
        payloads,
        "results/experiment_5434_pbit_polarfire_timing_variance_v494.json",
        "timing_variance_receipts_ready",
    ) is True
    measured = _value(
        payloads,
        "results/experiment_5434_pbit_polarfire_timing_variance_v494.json",
        "measurement_access_complete",
    ) is True
    return "partial" if ready and measured else "blocked"


def _workflow_memory_classifier(payloads: Payloads) -> str:
    memory_ready = _value(
        payloads,
        "results/experiment_5435_verified_workflow_memory_csl_v494.json",
        "verified_workflow_memory_ready",
    ) is True
    transfer_ready = _value(
        payloads,
        "results/experiment_5436_csl_memory_transfer_stress_v494.json",
        "csl_transfer_stress_ready",
    ) is True
    no_weight_mutation = _value(
        payloads,
        "results/experiment_5435_verified_workflow_memory_csl_v494.json",
        "no_weight_mutation",
    ) is True and _value(
        payloads,
        "results/experiment_5436_csl_memory_transfer_stress_v494.json",
        "no_weight_mutation",
    ) is True
    return "closed" if memory_ready and transfer_ready and no_weight_mutation else "partial"


def _arc_classifier(payloads: Payloads) -> str:
    banked = _value(
        payloads,
        "results/experiment_5437_arc_live_reinduction_levelup_v494.json",
        "arc_new_level_banked",
    ) is True
    return "closed" if banked else "honest_null"


def _certificate_classifier(payloads: Payloads) -> str:
    ready = _value(
        payloads,
        "results/experiment_5438_kan_ontology_measurement_certificate_v494.json",
        "kan_ontology_certificate_ready",
    ) is True
    broad_claim = _value(
        payloads,
        "results/experiment_5438_kan_ontology_measurement_certificate_v494.json",
        "broad_kan_verification_claim",
    ) is True
    return "partial" if ready and not broad_claim else "blocked"


def _token_internal_classifier(payloads: Payloads) -> str:
    missing_evidence = _value(
        payloads,
        "results/experiment_5438_kan_ontology_measurement_certificate_v494.json",
        "missing_evidence_detected",
    ) is True
    transition_blocks = bool(
        _value(payloads, "results/experiment_5428_transition_v494.json", "blocked_lanes")
    )
    return "blocked" if missing_evidence or transition_blocks else "missing"


def _hardware_speedup_classifier(payloads: Payloads) -> str:
    speedup = _value(
        payloads,
        "results/experiment_5434_pbit_polarfire_timing_variance_v494.json",
        "hardware_speedup_claim",
    ) is True
    return "closed" if speedup else "honest_null"


LANE_SPECS = (
    LaneSpec(
        name="structured_verification",
        artifact_fields={
            "results/experiment_5430_structured_tautology_corrigendum_v494.json": (
                "honest_verdict",
                "structured_corrigendum_clean",
                "adversarial_verify_clean",
                "row_count_recomputed",
            ),
            "results/experiment_5431_structured_constraint_taxonomy_replication_v494.json": (
                "honest_verdict",
                "structured_taxonomy_replication_ready",
                "metric_independence_checks_passed",
                "accepted_risk_bound",
                "accepted_risk_bound_threshold",
            ),
        },
        classification_reason="corrigendum_and_taxonomy_replication_closed",
        claim_boundary="closed for structured verification fixtures; not a broad SOTA quality claim",
        failure_taxonomy=("tool-use", "calibration", "structured-corrigendum"),
        prd_refs=("FR-12 Verifiable Reasoning", "NFR-02 Safety"),
        research_program_priorities=("structured verification", "tool-first unsafe-action controls"),
        next_action="Feed only the corrected structured fields into the .494 capstone.",
        classifier=_structured_classifier,
    ),
    LaneSpec(
        name="continuous_self_learning",
        artifact_fields={
            "results/experiment_5435_verified_workflow_memory_csl_v494.json": (
                "honest_verdict",
                "verified_workflow_memory_ready",
                "verify_before_store_pass_rate",
                "retrieval_trap_deflection_rate",
                "rollback_verified",
                "no_weight_mutation",
            ),
            "results/experiment_5436_csl_memory_transfer_stress_v494.json": (
                "honest_verdict",
                "csl_transfer_stress_ready",
                "in_domain_quality_delta",
                "out_of_domain_quality_delta",
                "negative_transfer_deflection_rate",
                "rollback_verified",
                "no_weight_mutation",
            ),
        },
        classification_reason="workflow_memory_and_transfer_controls_closed",
        claim_boundary="closed for controller-level workflow memory; no model or adapter weight mutation",
        failure_taxonomy=("reasoning", "planning", "workflow-memory", "memory-transfer"),
        prd_refs=("FR-11 Autonomous Self-Learning Loop",),
        research_program_priorities=("continuous self-learning", "verified memory transfer with rollback"),
        next_action="Carry forward as bounded CSL evidence with no weight-mutation claim.",
        classifier=_workflow_memory_classifier,
    ),
    LaneSpec(
        name="ontology_memory",
        artifact_fields={
            "results/experiment_5432_ontology_softlogic_constraint_memory_v494.json": (
                "honest_verdict",
                "ontology_constraint_memory_ready",
                "valid_update_preservation_rate",
                "false_triple_rejection_rate",
                "unsupported_update_abstention_rate",
                "soft_logic_overrode_solver",
            ),
        },
        classification_reason="ontology_softlogic_preserved_solver_authority",
        claim_boundary="closed for deterministic ontology-memory fixtures with solver authority preserved",
        failure_taxonomy=("reasoning", "workflow-memory"),
        prd_refs=("FR-11 Autonomous Self-Learning Loop", "FR-12 Verifiable Reasoning"),
        research_program_priorities=("ontology memory", "constraint-backed memory promotion"),
        next_action="Use as ontology-memory evidence, not as a broad graph-soundness proof.",
        classifier=_ontology_classifier,
    ),
    LaneSpec(
        name="solver_guidance",
        artifact_fields={
            "results/experiment_5433_active_constraint_diversity_lns_v494.json": (
                "honest_verdict",
                "active_constraint_diversity_ready",
                "solver_validity_preserved",
                "work_delta",
                "claim_limits",
            ),
        },
        classification_reason="bounded_advisory_solver_guidance",
        claim_boundary="partial because LNS hints are advisory and deterministic solver authority remains final",
        failure_taxonomy=("planning", "reasoning", "solver/hardware"),
        prd_refs=("FR-07 Inference Pipeline", "FR-12 Verifiable Reasoning"),
        research_program_priorities=("solver guidance", "constraint-backed search"),
        next_action="Promote as bounded solver guidance, not autonomous planning competence.",
        classifier=_solver_classifier,
    ),
    LaneSpec(
        name="hardware",
        artifact_fields={
            "results/experiment_5434_pbit_polarfire_timing_variance_v494.json": (
                "honest_verdict",
                "timing_variance_receipts_ready",
                "measurement_access_complete",
                "same_workload_hash_match",
                "same_result_hash_match",
                "hardware_speedup_claim",
            ),
        },
        classification_reason="hash_matched_timing_receipts_without_speedup",
        claim_boundary="partial because CPU and PolarFire timing receipts are comparable evidence only",
        failure_taxonomy=("measurement-access", "hardware", "solver/hardware"),
        prd_refs=("NFR-01 Performance",),
        research_program_priorities=("hardware", "hash-matched board timing receipts"),
        next_action="Use as hardware measurement-access evidence; do not claim acceleration.",
        classifier=_hardware_classifier,
    ),
    LaneSpec(
        name="certificates",
        artifact_fields={
            "results/experiment_5438_kan_ontology_measurement_certificate_v494.json": (
                "honest_verdict",
                "kan_ontology_certificate_ready",
                "false_property_rejection_rate",
                "true_property_preservation_rate",
                "missing_evidence_detected",
                "broad_kan_verification_claim",
                "missing_evidence_controls",
                "claim_limits",
            ),
        },
        classification_reason="bounded_measurement_access_certificate",
        claim_boundary="partial because the KAN certificate is bounded to observable ontology/workflow-memory rows",
        failure_taxonomy=("measurement-access", "reasoning"),
        prd_refs=("FR-10 Spec-Driven Development", "FR-12 Verifiable Reasoning"),
        research_program_priorities=("certificates", "observable-vs-missing evidence separation"),
        next_action="Carry forward certificate limits and missing-evidence controls.",
        classifier=_certificate_classifier,
    ),
    LaneSpec(
        name="token_internal_feature_access",
        artifact_fields={
            "results/experiment_5428_transition_v494.json": (
                "honest_verdict",
                "blocked_lanes",
            ),
            "results/experiment_5438_kan_ontology_measurement_certificate_v494.json": (
                "honest_verdict",
                "missing_evidence_detected",
                "missing_evidence_controls",
                "claim_limits",
            ),
        },
        classification_reason="token_internal_receipts_absent",
        claim_boundary="blocked because token logprob and hidden/internal activation evidence is unsupported",
        failure_taxonomy=("measurement-access", "token/internal"),
        prd_refs=("FR-12 Verifiable Reasoning",),
        research_program_priorities=("certificates", "token/internal gap handling"),
        next_action="Keep token/internal claims blocked until authenticated backend receipts land.",
        classifier=_token_internal_classifier,
    ),
    LaneSpec(
        name="arc_live_progress",
        artifact_fields={
            "results/experiment_5437_arc_live_reinduction_levelup_v494.json": (
                "honest_verdict",
                "status",
                "arc_new_level_banked",
                "offline_reproduced",
                "reproduced_levels",
                "newly_reached_levels",
                "attempt_count",
                "frontier_expansion_count",
                "failure_mode",
            ),
        },
        classification_reason="bounded_budget_no_new_level_banked",
        claim_boundary="honest null: live ARC path ran but no offline-reproduced new level was banked",
        failure_taxonomy=("planning", "live-environment", "ARC"),
        prd_refs=("FR-12 Verifiable Reasoning",),
        research_program_priorities=("ARC live progress", "live hidden-game discovery agent path"),
        next_action="Keep the banked ARC level count unchanged and preserve the frontier evidence.",
        classifier=_arc_classifier,
    ),
    LaneSpec(
        name="hardware_speedup_claim",
        artifact_fields={
            "results/experiment_5434_pbit_polarfire_timing_variance_v494.json": (
                "honest_verdict",
                "timing_variance_receipts_ready",
                "measurement_access_complete",
                "hardware_speedup_claim",
            ),
        },
        classification_reason="no_board_speedup_claim_supported",
        claim_boundary="honest null: measurement receipts exist but hardware_speedup_claim is false",
        failure_taxonomy=("measurement-access", "hardware", "solver/hardware"),
        prd_refs=("NFR-01 Performance",),
        research_program_priorities=("hardware", "no speedup inflation"),
        next_action="Report the no-speedup boundary explicitly in the capstone.",
        classifier=_hardware_speedup_classifier,
    ),
)


def _lane_groups(payloads: Payloads) -> dict[str, list[JsonDict]]:
    groups: dict[str, list[JsonDict]] = {
        "closed_lanes": [],
        "partial_lanes": [],
        "blocked_lanes": [],
        "honest_null_lanes": [],
        "missing_lanes": [],
    }
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
        entry = _lane_entry(
            spec,
            classification,
            payloads,
            missing_artifacts,
            missing_supporting_fields,
            supporting_fields,
        )
        groups[f"{classification}_lanes"].append(entry)
    return groups


def _failure_taxonomy_counts(groups: Mapping[str, list[JsonDict]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for lanes in groups.values():
        for lane in lanes:
            counter.update(str(tag) for tag in lane["failure_taxonomy"])
    return dict(sorted(counter.items()))


def _honest_verdict(ready: bool, groups: Mapping[str, list[JsonDict]], upstream_missing: list[str]) -> str:
    counts = {
        "closed": len(groups["closed_lanes"]),
        "partial": len(groups["partial_lanes"]),
        "blocked": len(groups["blocked_lanes"]),
        "honest_null": len(groups["honest_null_lanes"]),
        "missing": len(groups["missing_lanes"]),
    }
    prefix = "complete:" if ready else "blocked:"
    reason = (
        "read actual .494 upstream artifacts"
        if ready
        else f"missing upstream evidence {upstream_missing}"
    )
    return (
        f"{prefix} .494 PRD gap table {reason}; closed={counts['closed']}, "
        f"partial={counts['partial']}, blocked={counts['blocked']}, "
        f"honest_null={counts['honest_null']}, missing={counts['missing']}."
    )


def build_report(root: Path | str) -> JsonDict:
    """Build the table from files under ``root`` without running experiments.

    The caller supplies a repository-like root path. The function reads only the
    fixed .494 upstream artifact paths, so tests can use a temporary root with
    synthetic JSON and the real run can use the repository root.
    """

    root_path = Path(root)
    payloads, upstream_read, upstream_missing = _load_upstreams(root_path)
    groups = _lane_groups(payloads)
    ready = not upstream_missing and not groups["missing_lanes"]
    return {
        "schema": "carnot.prd_gap_agent_failure_table.v494.exp5439",
        "experiment_id": 5439,
        "spec_refs": ["REQ-HARNESS-015", "SCENARIO-HARNESS-010"],
        "field_principles": FIELD_PRINCIPLES,
        "upstream_artifacts_read": upstream_read,
        "upstream_artifacts_missing": upstream_missing,
        "closed_lanes": groups["closed_lanes"],
        "partial_lanes": groups["partial_lanes"],
        "blocked_lanes": groups["blocked_lanes"],
        "honest_null_lanes": groups["honest_null_lanes"],
        "missing_lanes": groups["missing_lanes"],
        "failure_taxonomy_counts": _failure_taxonomy_counts(groups),
        "prd_gap_table_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, groups, upstream_missing),
    }


def write_artifact(root: Path | str) -> Path:
    """Write the Exp5439 result JSON and return its path."""

    root_path = Path(root)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_report(root_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
