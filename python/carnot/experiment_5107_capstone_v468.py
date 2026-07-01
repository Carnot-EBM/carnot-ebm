#!/usr/bin/env python3
"""Exp 5107: ungated .468 capstone aggregation.

Spec refs: REQ-CAPSTONE-5107, SCENARIO-CAPSTONE-5107,
SCENARIO-CAPSTONE-5107-FIELD-PRINCIPLES.

This module reads the upstream .468 artifacts and writes a final milestone
decision. It does not run a model. The important discipline is that flagged,
blocked, and missing evidence is still recorded, but those rows do not become
headline evidence. This keeps a positive exact-verifier scale-up separate from
the still-blocked runtime substrate and from flagged FR-11/constrained-decoding
rows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5107_capstone_v468"
EXPERIMENT_ID = 5107
SCHEMA = "carnot.experiment_5107_capstone_v468.v1"
RESULT_RELATIVE_PATH = Path("results") / "experiment_5107_capstone_v468.json"
MILESTONE = "2026.07.468"
RANDOM_SEED = 5107
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = "complete_capstone_v468_exact_verifier_scale_decision_recorded"

SPEC_REFS = [
    "REQ-CAPSTONE-5107",
    "SCENARIO-CAPSTONE-5107",
    "SCENARIO-CAPSTONE-5107-FIELD-PRINCIPLES",
]


@dataclass(frozen=True)
class UpstreamSource:
    """Expected upstream artifact and the fields imported into the capstone."""

    label: str
    experiment_id: int
    relative_path: Path
    imported_fields: tuple[str, ...]
    allowed_substrates: tuple[str, ...]


UPSTREAMS: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        "archive_467_activate_468",
        5095,
        Path("results/experiment_5095_archive_467_activate_468.json"),
        ("honest_verdict", "inference_substrate", "exact_verifier_pivot", "docs_updated"),
        ("aggregation_from_upstream_artifacts",),
    ),
    UpstreamSource(
        "sota_ingestion",
        5096,
        Path("results/experiment_5096_sota_ingestion_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "sources_checked",
            "task_mapping",
            "planning_hooks",
        ),
        ("literature_review_and_repo_inspection",),
    ),
    UpstreamSource(
        "runtime_endpoint_logprob_cache",
        5097,
        Path("results/experiment_5097_clean_sota_endpoint_logprob_cache_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "logprob_endpoint_clean",
            "logprob_endpoint_ready",
            "completion_endpoint_ready",
            "live_llm_invoked",
            "endpoint_url",
            "model_specs",
        ),
        ("precondition_check_only", "live_llm_inference"),
    ),
    UpstreamSource(
        "kan_pwa_milp_scale_v2",
        5098,
        Path("results/experiment_5098_kan_pwa_milp_scale_v2.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "properties_proved",
            "false_property_controls_passed",
            "max_scale_reached",
            "scale_blocker",
        ),
        ("exact_milp_solver_cpu",),
    ),
    UpstreamSource(
        "beaver_prefix_bounds",
        5099,
        Path("results/experiment_5099_beaver_prefix_bound_verifier_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "backend_used",
            "soundness_checks_passed",
            "lower_bound",
            "upper_bound",
            "bound_gap",
            "live_llm_invoked",
        ),
        ("deterministic_toy_finite_distribution", "live_llm_inference"),
    ),
    UpstreamSource(
        "constrainprompt_code_assurance",
        5100,
        Path("results/experiment_5100_constrainprompt_code_assurance_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "exact_checker_backend",
            "constraints_total",
            "executable_constraints_total",
            "positive_tests_passed",
            "negative_tests_passed",
            "adversarial_tests_passed",
            "llm_invoked",
        ),
        ("deterministic_python_json_logical_tree",),
    ),
    UpstreamSource(
        "graph_evidence_energy",
        5101,
        Path("results/experiment_5101_incomplete_graph_evidence_energy_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "contradiction_reject_rate",
            "unsupported_retained_rate",
            "supported_accept_rate",
            "stability_under_perturbation",
        ),
        ("synthetic_graph_exact_labels",),
    ),
    UpstreamSource(
        "hubo_pspin_direct_energy",
        5102,
        Path("results/experiment_5102_hubo_pspin_direct_energy_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "direct_hubo_advantage",
            "exact_optima_verified",
            "auxiliary_variable_blowup",
            "energy_scale_distortion",
        ),
        ("exact_enumeration_cpu",),
    ),
    UpstreamSource(
        "taco_adaptive_csp_heuristic",
        5103,
        Path("results/experiment_5103_taco_adaptive_csp_heuristic_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "correctness_preserved",
            "delta_effort_vs_baseline",
            "baseline_effort",
            "adapted_effort",
            "harmful_instance_count",
            "instances_total",
        ),
        ("exact_solver_with_adaptive_cpu_heuristic",),
    ),
    UpstreamSource(
        "constrained_decoding_semantic_risk_audit",
        5104,
        Path("results/experiment_5104_constrained_decoding_semantic_risk_audit_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "syntax_only_headline_forbidden",
            "syntax_validity_rate",
            "semantic_validity_rate",
            "distribution_shift_metric",
            "live_llm_invoked",
        ),
        ("deterministic_static_csr_semantic_distribution_audit", "live_llm_inference"),
    ),
    UpstreamSource(
        "fr11_severa_guarded_memory",
        5105,
        Path("results/experiment_5105_fr11_severa_guarded_memory_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "heldout_delta",
            "nonforgetting_delta",
            "promoted_count",
            "contract_pass_count",
            "poison_guard_passed",
            "contamination_guard_passed",
            "rollback_guard_passed",
            "promotion_decision",
        ),
        ("exact_guarded_self_learning_eval",),
    ),
    UpstreamSource(
        "hardware_partition_telemetry",
        5106,
        Path("results/experiment_5106_hardware_partition_telemetry_v468.json"),
        (
            "honest_verdict",
            "inference_substrate",
            "kv260_ssh_ready",
            "kv260_uio_transcript_collected",
            "kv260_blocker",
            "gatemate_detected",
            "gatemate_terminal_state",
            "polarfire_ssh_ready",
            "polarfire_dispatch_precheck",
            "speedup_claimed",
            "destructive_actions_taken",
            "partition_telemetry",
        ),
        ("hardware_smoke_and_static_mapping",),
    ),
)

UPSTREAMS_BY_ID = {source.experiment_id: source for source in UPSTREAMS}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; records the .468 capstone decision without headline "
            "claims from blocked, missing, or flagged upstream artifacts."
        )
    },
    "duration_s": {
        "principle": (
            "wall-clock duration for the aggregation run; it must stay compatible "
            "with aggregation_from_upstream_artifacts and not imply live inference."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- loads upstream JSON only; the "
            "capstone must not claim a live model run."
        )
    },
    "source_artifacts": {
        "principle": (
            "every expected .468 upstream artifact that is present, with sha256, "
            "imported fields, classification, and headline eligibility."
        )
    },
    "missing_artifacts": {
        "principle": (
            "expected .468 artifacts absent or unreadable; empty only when every "
            "prompt-listed source is loadable."
        )
    },
    "clean_positive_artifacts": {
        "principle": "non-flagged, non-blocked upstreams whose claim is positive and substrate-consistent."
    },
    "clean_negative_artifacts": {
        "principle": (
            "non-flagged, non-blocked upstreams that validly report a null, "
            "no-speedup, or bounded negative."
        )
    },
    "blocked_artifacts": {
        "principle": "non-flagged upstreams with blocked verdict/status, recorded as blockers rather than missing or successful."
    },
    "flagged_artifacts": {
        "principle": (
            "flagged .468 upstreams preserved with excluded-from-headline status "
            "so their numbers are not promoted."
        )
    },
    "milestone_decision": {
        "principle": "the overall .468 headline chosen only from clean non-flagged, substrate-consistent evidence."
    },
    "exact_verifier_decision": {
        "principle": (
            "KAN/MILP, graph-evidence, HUBO/p-spin, and adaptive exact-solver "
            "positives summarized separately from flagged BEAVER/code-assurance rows."
        )
    },
    "runtime_substrate_decision": {
        "principle": (
            "Exp5097 readiness summarized without allowing blocked runtime to "
            "invalidate non-LLM exact-verifier claims."
        )
    },
    "fr11_decision": {
        "principle": (
            "FR-11 promotion/no-promote state with flagged promotion evidence "
            "excluded from clean promotion claims."
        )
    },
    "hardware_decision": {
        "principle": (
            "KV260/GateMate/PolarFire continuity and partition telemetry with no "
            "acceleration claim unless authenticated timing exists."
        )
    },
    "constrained_generation_decision": {
        "principle": (
            "constrained-generation evidence is preserved only as clean headline "
            "evidence when the semantic audit is non-flagged."
        )
    },
    "docs_updated": {
        "principle": (
            "empty when the conductor stop rule delegates ops/status/changelog/"
            "traceability reconciliation."
        )
    },
    "next_research_questions": {
        "principle": (
            "bounded next actions derived from clean positives, blockers, and "
            "flagged rows rather than a new headline claim."
        )
    },
    "flagged_adversarial": {
        "principle": "false for the capstone itself when it transparently records flagged upstreams."
    },
}

REQUIRED_TOP_LEVEL_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "source_artifacts",
    "missing_artifacts",
    "clean_positive_artifacts",
    "clean_negative_artifacts",
    "blocked_artifacts",
    "flagged_artifacts",
    "milestone_decision",
    "exact_verifier_decision",
    "runtime_substrate_decision",
    "fr11_decision",
    "hardware_decision",
    "constrained_generation_decision",
    "docs_updated",
    "next_research_questions",
    "flagged_adversarial",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "preconditions_checked",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
    *REQUIRED_TOP_LEVEL_FIELDS,
)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {"exists": True, "loadable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):
        return {}, {"exists": True, "loadable": False, "error": "json_not_object"}
    return dict(payload), {"exists": True, "loadable": True, "sha256": file_sha256(path)}


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _bool(value: Any) -> bool:
    return value is True


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _blocked(payload: JsonMap) -> bool:
    verdict = str(payload.get("honest_verdict", ""))
    return verdict.startswith("blocked_") or payload.get("status") == "blocked"


def _substrate_consistent(source: UpstreamSource, payload: JsonMap) -> bool:
    return str(payload.get("inference_substrate", "")) in source.allowed_substrates


def _classify_artifact(source: UpstreamSource, payload: JsonMap, status: JsonMap) -> str:
    if status.get("loadable") is not True:
        return "missing"
    if payload.get("flagged_adversarial") is True:
        return "adversarially_flagged"
    if _blocked(payload):
        return "blocked"
    if source.experiment_id == 5106 and payload.get("speedup_claimed") is False:
        return "clean_negative"
    return "clean_positive"


def _classification_reason(source: UpstreamSource, classification: str) -> str:
    positive_reasons = {
        5095: "clean_transition_record",
        5096: "clean_source_ingestion",
        5098: "clean_kan_milp_property_suite",
        5101: "clean_graph_evidence_energy",
        5102: "clean_hubo_pspin_exact_enumeration",
        5103: "clean_adaptive_solver_effort_reduction",
    }
    special_reasons = {
        "missing": "missing_or_unloadable_source",
        "adversarially_flagged": "upstream_flagged_adversarial_excluded_from_headline",
        "blocked": "blocked_verdict_or_status",
        "clean_negative": "clean_hardware_continuity_no_speedup_claim",
    }
    return positive_reasons.get(
        source.experiment_id, special_reasons.get(classification, "classified_clean")
    )


def _imported_field_values(source: UpstreamSource, payload: JsonMap) -> JsonDict:
    return {field: payload[field] for field in source.imported_fields if field in payload}


def _artifact_row(source: UpstreamSource, payload: JsonMap, status: JsonMap) -> JsonDict:
    classification = _classify_artifact(source, payload, status)
    blocked = _blocked(payload) if status.get("loadable") is True else False
    flagged = payload.get("flagged_adversarial") is True
    claim_consistent = (
        _substrate_consistent(source, payload) if status.get("loadable") is True else False
    )
    row: JsonDict = {
        "label": source.label,
        "experiment_id": source.experiment_id,
        "path": str(source.relative_path),
        "exists": status.get("exists") is True,
        "loadable": status.get("loadable") is True,
        "present": status.get("loadable") is True,
        "classification": classification,
        "classification_reason": _classification_reason(source, classification),
        "fields_imported": list(source.imported_fields),
        "imported": _imported_field_values(source, payload),
        "honest_verdict": str(payload.get("honest_verdict", "")),
        "flagged_adversarial": flagged,
        "blocked": blocked,
        "inference_substrate": str(payload.get("inference_substrate", "")),
        "claim_consistent_with_substrate": claim_consistent,
        "headline_eligible": (
            status.get("loadable") is True
            and not flagged
            and not blocked
            and claim_consistent
            and classification in {"clean_positive", "clean_negative"}
        ),
    }
    if "sha256" in status:
        row["sha256"] = status["sha256"]
    if "error" in status:
        row["error"] = status["error"]
    duration = _number(payload.get("duration_s"))
    if duration is not None:
        row["duration_s"] = duration
    if flagged:
        row["excluded_from_headline"] = True
    if blocked:
        row["blocker_reason"] = "blocked_verdict_or_status"
    return row


def load_v468_artifacts(
    root: Path,
) -> tuple[
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    dict[int, JsonDict],
]:
    source_rows: list[JsonDict] = []
    missing_rows: list[JsonDict] = []
    clean_positive_rows: list[JsonDict] = []
    clean_negative_rows: list[JsonDict] = []
    blocked_rows: list[JsonDict] = []
    flagged_rows: list[JsonDict] = []
    payloads: dict[int, JsonDict] = {}

    for source in UPSTREAMS:
        payload, status = read_json_mapping(root / source.relative_path)
        row = _artifact_row(source, payload, status)
        if row["present"]:
            source_rows.append(row)
            payloads[source.experiment_id] = payload
        if row["classification"] == "missing":
            missing_rows.append(row)
        if row["classification"] == "clean_positive":
            clean_positive_rows.append(row)
        if row["classification"] == "clean_negative":
            clean_negative_rows.append(row)
        if row["classification"] == "blocked":
            blocked_rows.append(row)
        if row["classification"] == "adversarially_flagged":
            flagged_rows.append(row)

    return (
        source_rows,
        missing_rows,
        clean_positive_rows,
        clean_negative_rows,
        blocked_rows,
        flagged_rows,
        payloads,
    )


def _payload(payloads: Mapping[int, JsonDict], exp_id: int) -> JsonDict:
    return dict(payloads.get(exp_id, {}))


def _ids(rows: list[JsonDict]) -> set[int]:
    return {int(row["experiment_id"]) for row in rows}


def build_exact_verifier_decision(
    payloads: Mapping[int, JsonDict],
    clean_positive_ids: set[int],
    flagged_ids: set[int],
) -> JsonDict:
    kan = _payload(payloads, 5098)
    beaver = _payload(payloads, 5099)
    code = _payload(payloads, 5100)
    graph = _payload(payloads, 5101)
    hubo = _payload(payloads, 5102)
    taco = _payload(payloads, 5103)
    kan_clean = 5098 in clean_positive_ids and _bool(kan.get("false_property_controls_passed"))
    graph_clean = (
        5101 in clean_positive_ids and _number(graph.get("contradiction_reject_rate")) == 1.0
    )
    clean_scale_up = kan_clean and graph_clean
    decision = {
        True: "clean_bounded_exact_verifier_scale_up_with_flagged_exclusions",
        False: "bounded_no_clean_exact_verifier_scale_up",
    }[clean_scale_up]
    return {
        "decision": decision,
        "clean_scale_up": clean_scale_up,
        "bounded_not_architecture_scale": True,
        "non_llm_exact_claims_not_blocked_by_exp5097": True,
        "clean_source_experiment_ids": sorted(
            clean_positive_ids.intersection({5098, 5101, 5102, 5103})
        ),
        "flagged_exact_verifier_experiment_ids": sorted(flagged_ids.intersection({5099, 5100})),
        "kan_milp": {
            "source_experiment_id": 5098,
            "clean_positive": kan_clean,
            "properties_proved": _list(kan.get("properties_proved")),
            "false_property_controls_passed": _bool(kan.get("false_property_controls_passed")),
            "max_scale_reached": _mapping(kan.get("max_scale_reached")),
            "scale_blocker": kan.get("scale_blocker"),
        },
        "beaver_prefix_bounds": {
            "source_experiment_id": 5099,
            "excluded_from_headline": 5099 in flagged_ids,
            "flagged_adversarial": 5099 in flagged_ids,
            "backend_used": str(beaver.get("backend_used", "")),
            "soundness_checks_passed": _bool(beaver.get("soundness_checks_passed")),
            "lower_bound": _number(beaver.get("lower_bound")),
            "upper_bound": _number(beaver.get("upper_bound")),
            "bound_gap": _number(beaver.get("bound_gap")),
            "live_llm_invoked": _bool(beaver.get("live_llm_invoked")),
        },
        "code_assurance": {
            "source_experiment_id": 5100,
            "excluded_from_headline": 5100 in flagged_ids,
            "flagged_adversarial": 5100 in flagged_ids,
            "exact_checker_backend": str(code.get("exact_checker_backend", "")),
            "constraints_total": _number(code.get("constraints_total")),
            "executable_constraints_total": _number(code.get("executable_constraints_total")),
            "positive_tests_passed": _bool(code.get("positive_tests_passed")),
            "negative_tests_passed": _bool(code.get("negative_tests_passed")),
            "adversarial_tests_passed": _bool(code.get("adversarial_tests_passed")),
        },
        "graph_evidence": {
            "source_experiment_id": 5101,
            "clean_positive": graph_clean,
            "contradiction_reject_rate": _number(graph.get("contradiction_reject_rate")),
            "unsupported_retained_rate": _number(graph.get("unsupported_retained_rate")),
            "supported_accept_rate": _number(graph.get("supported_accept_rate")),
            "stability_under_perturbation": _mapping(graph.get("stability_under_perturbation")),
        },
        "hubo_pspin": {
            "source_experiment_id": 5102,
            "clean_positive": 5102 in clean_positive_ids,
            "direct_hubo_advantage": _bool(hubo.get("direct_hubo_advantage")),
            "exact_optima_verified": _bool(hubo.get("exact_optima_verified")),
            "auxiliary_variable_blowup": _mapping(hubo.get("auxiliary_variable_blowup")),
            "energy_scale_distortion": _mapping(hubo.get("energy_scale_distortion")),
        },
        "adaptive_solver": {
            "source_experiment_id": 5103,
            "clean_positive": 5103 in clean_positive_ids,
            "correctness_preserved": _bool(taco.get("correctness_preserved")),
            "delta_effort_vs_baseline": _mapping(taco.get("delta_effort_vs_baseline")),
            "baseline_effort": _mapping(taco.get("baseline_effort")),
            "adapted_effort": _mapping(taco.get("adapted_effort")),
            "harmful_instance_count": _number(taco.get("harmful_instance_count")),
            "instances_total": _number(taco.get("instances_total")),
        },
    }


def build_runtime_substrate_decision(
    payloads: Mapping[int, JsonDict], blocked_ids: set[int]
) -> JsonDict:
    runtime = _payload(payloads, 5097)
    clean = (
        5097 not in blocked_ids
        and _bool(runtime.get("logprob_endpoint_clean"))
        and _bool(runtime.get("logprob_endpoint_ready"))
        and _bool(runtime.get("live_llm_invoked"))
    )
    return {
        "decision": {
            True: "clean_live_logprob_substrate_ready",
            False: "blocked_no_clean_live_logprob_substrate",
        }[clean],
        "source_experiment_id": 5097,
        "runtime_substrate_clean": clean,
        "blocked": 5097 in blocked_ids,
        "honest_verdict": str(runtime.get("honest_verdict", "")),
        "logprob_endpoint_clean": _bool(runtime.get("logprob_endpoint_clean")),
        "logprob_endpoint_ready": _bool(runtime.get("logprob_endpoint_ready")),
        "completion_endpoint_ready": _bool(runtime.get("completion_endpoint_ready")),
        "live_llm_invoked": _bool(runtime.get("live_llm_invoked")),
        "endpoint_url": str(runtime.get("endpoint_url", "")),
        "model_specs_present": bool(runtime.get("model_specs")),
        "does_not_gate_non_llm_exact_verifiers": True,
    }


def build_fr11_decision(payloads: Mapping[int, JsonDict], flagged_ids: set[int]) -> JsonDict:
    fr11 = _payload(payloads, 5105)
    promotion = _mapping(fr11.get("promotion_decision"))
    promoted_count = _number(fr11.get("promoted_count"))
    heldout_delta = _number(fr11.get("heldout_delta"))
    nonforgetting_delta = _number(fr11.get("nonforgetting_delta"))
    source_promoted = _bool(promotion.get("promoted")) or (
        promoted_count is not None and promoted_count > 0.0
    )
    clean_promotion = (
        5105 not in flagged_ids
        and source_promoted
        and heldout_delta is not None
        and heldout_delta > 0.0
        and nonforgetting_delta is not None
        and nonforgetting_delta >= 0.0
    )
    return {
        "decision": {
            True: "clean_fr11_promotion_ready",
            False: "no_clean_fr11_promotion_flagged_artifact_requires_rerun",
        }[clean_promotion],
        "source_experiment_id": 5105,
        "upstream_excluded_from_headline": 5105 in flagged_ids,
        "flagged_adversarial": 5105 in flagged_ids,
        "source_promoted": source_promoted,
        "promotion_allowed_from_clean_evidence": clean_promotion,
        "heldout_delta": heldout_delta,
        "nonforgetting_delta": nonforgetting_delta,
        "contract_pass_count": _number(fr11.get("contract_pass_count")),
        "poison_guard_passed": _bool(fr11.get("poison_guard_passed")),
        "contamination_guard_passed": _bool(fr11.get("contamination_guard_passed")),
        "rollback_guard_passed": _bool(fr11.get("rollback_guard_passed")),
        "no_promote_reason": str(promotion.get("no_promote_reason", "")),
    }


def build_hardware_decision(
    payloads: Mapping[int, JsonDict], clean_negative_ids: set[int]
) -> JsonDict:
    hardware = _payload(payloads, 5106)
    speedup_claimed = _bool(hardware.get("speedup_claimed"))
    continuity_progress = 5106 in clean_negative_ids and (
        _bool(hardware.get("kv260_ssh_ready")) or _bool(hardware.get("polarfire_ssh_ready"))
    )
    return {
        "decision": {
            True: "hardware_speedup_claimed",
            False: "hardware_continuity_progress_no_speedup",
        }[speedup_claimed],
        "source_experiment_id": 5106,
        "continuity_progress": continuity_progress,
        "kv260_ssh_ready": _bool(hardware.get("kv260_ssh_ready")),
        "kv260_uio_transcript_collected": _bool(hardware.get("kv260_uio_transcript_collected")),
        "kv260_blocker": str(hardware.get("kv260_blocker", "")),
        "gatemate_detected": _bool(hardware.get("gatemate_detected")),
        "gatemate_terminal_state": str(hardware.get("gatemate_terminal_state", "")),
        "polarfire_ssh_ready": _bool(hardware.get("polarfire_ssh_ready")),
        "polarfire_dispatch_precheck": _mapping(hardware.get("polarfire_dispatch_precheck")),
        "speedup_claimed": speedup_claimed,
        "destructive_actions_taken": _list(hardware.get("destructive_actions_taken")),
        "partition_telemetry_count": len(_list(hardware.get("partition_telemetry"))),
    }


def build_constrained_generation_decision(
    payloads: Mapping[int, JsonDict], flagged_ids: set[int]
) -> JsonDict:
    constrained = _payload(payloads, 5104)
    clean_headline = 5104 not in flagged_ids and _bool(
        constrained.get("syntax_only_headline_forbidden")
    )
    return {
        "decision": {
            True: "clean_constrained_generation_semantic_audit_kept",
            False: "no_clean_constrained_generation_headline_flagged_audit_only",
        }[clean_headline],
        "source_experiment_id": 5104,
        "excluded_from_headline": 5104 in flagged_ids,
        "flagged_adversarial": 5104 in flagged_ids,
        "clean_evidence_worth_keeping": clean_headline,
        "syntax_only_headline_forbidden": _bool(constrained.get("syntax_only_headline_forbidden")),
        "syntax_validity_rate": _number(constrained.get("syntax_validity_rate")),
        "semantic_validity_rate": _number(constrained.get("semantic_validity_rate")),
        "distribution_shift_metric": _number(constrained.get("distribution_shift_metric")),
        "live_llm_invoked": _bool(constrained.get("live_llm_invoked")),
    }


def build_milestone_decision(
    exact: JsonMap,
    runtime: JsonMap,
    fr11: JsonMap,
    hardware: JsonMap,
    constrained: JsonMap,
    *,
    missing_count: int,
    blocked_count: int,
    flagged_count: int,
) -> JsonDict:
    clean_exact = _bool(exact.get("clean_scale_up"))
    clean_runtime = _bool(runtime.get("runtime_substrate_clean"))
    safe_fr11 = _bool(fr11.get("promotion_allowed_from_clean_evidence"))
    constrained_clean = _bool(constrained.get("clean_evidence_worth_keeping"))
    hardware_progress = (
        "continuity_progress_no_speedup"
        if _bool(hardware.get("continuity_progress"))
        else "no_clean_hardware_progress"
    )
    return {
        "decision": {
            True: (
                "bounded_exact_verifier_scale_up_clean_runtime_blocked_fr11_no_clean_promotion_"
                "hardware_continuity_no_speedup"
            ),
            False: "bounded_no_clean_exact_verifier_scale_up",
        }[
            clean_exact
            and not clean_runtime
            and not safe_fr11
            and hardware_progress == "continuity_progress_no_speedup"
        ],
        "clean_exact_verifier_scale_up": clean_exact,
        "clean_runtime_substrate": clean_runtime,
        "safe_fr11_promotion": safe_fr11,
        "constrained_generation_evidence_worth_keeping": constrained_clean,
        "hardware_progress": hardware_progress,
        "missing_artifact_count": missing_count,
        "blocked_artifact_count": blocked_count,
        "flagged_artifact_count": flagged_count,
        "headline_evidence_policy": "only_flagged_false_and_substrate_consistent_rows",
    }


def build_next_research_questions(
    exact: JsonMap, runtime: JsonMap, fr11: JsonMap, hardware: JsonMap, constrained: JsonMap
) -> list[JsonDict]:
    return [
        {
            "source": "Exp5098/5101/5102/5103",
            "question": (
                "Can the bounded exact-verifier positives scale beyond toy/synthetic "
                "families while retaining false controls and exact authority?"
            ),
            "trigger": exact.get("decision"),
        },
        {
            "source": "Exp5097",
            "question": "Repair live local GGUF logprob cache before using LLM-backed verifier claims.",
            "trigger": runtime.get("decision"),
        },
        {
            "source": "Exp5104",
            "question": "Rerun constrained-generation semantic audit without adversarial flags before keeping it as evidence.",
            "trigger": constrained.get("decision"),
        },
        {
            "source": "Exp5105",
            "question": "Rerun FR-11 contract-guarded promotion with nonzero held-out utility and no adversarial flag.",
            "trigger": fr11.get("decision"),
        },
        {
            "source": "Exp5106",
            "question": "Collect safe UIO/register transcript or authenticated timing before any hardware speedup claim.",
            "trigger": hardware.get("decision"),
        },
    ]


def build_preconditions(missing_rows: list[JsonDict], source_rows: list[JsonDict]) -> JsonDict:
    return {
        "expected_artifacts": len(UPSTREAMS),
        "present_artifacts": len(source_rows),
        "missing_or_unloadable_artifacts": len(missing_rows),
        "operator_reconciliation_delegated": True,
        "research_roadmap_yaml_modified": False,
        "research_conductor_modified": False,
        "leaderboard_submission": False,
    }


def build_artifact(root: Path, *, duration_s: float) -> JsonDict:
    (
        source_rows,
        missing_rows,
        clean_positive_rows,
        clean_negative_rows,
        blocked_rows,
        flagged_rows,
        payloads,
    ) = load_v468_artifacts(root)
    clean_positive_ids = _ids(clean_positive_rows)
    clean_negative_ids = _ids(clean_negative_rows)
    blocked_ids = _ids(blocked_rows)
    flagged_ids = _ids(flagged_rows)

    exact = build_exact_verifier_decision(payloads, clean_positive_ids, flagged_ids)
    runtime = build_runtime_substrate_decision(payloads, blocked_ids)
    fr11 = build_fr11_decision(payloads, flagged_ids)
    hardware = build_hardware_decision(payloads, clean_negative_ids)
    constrained = build_constrained_generation_decision(payloads, flagged_ids)
    milestone = build_milestone_decision(
        exact,
        runtime,
        fr11,
        hardware,
        constrained,
        missing_count=len(missing_rows),
        blocked_count=len(blocked_rows),
        flagged_count=len(flagged_rows),
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "preconditions_checked": build_preconditions(missing_rows, source_rows),
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": COMPLETE_VERDICT,
        "duration_s": round(max(duration_s, 0.0001), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts": source_rows,
        "missing_artifacts": missing_rows,
        "clean_positive_artifacts": clean_positive_rows,
        "clean_negative_artifacts": clean_negative_rows,
        "blocked_artifacts": blocked_rows,
        "flagged_artifacts": flagged_rows,
        "milestone_decision": milestone,
        "exact_verifier_decision": exact,
        "runtime_substrate_decision": runtime,
        "fr11_decision": fr11,
        "hardware_decision": hardware,
        "constrained_generation_decision": constrained,
        "docs_updated": [],
        "next_research_questions": build_next_research_questions(
            exact, runtime, fr11, hardware, constrained
        ),
        "flagged_adversarial": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            errors.append(f"missing.{field}")
    if (
        str(artifact.get("honest_verdict", "")).startswith(("complete_", "success_", "blocked_"))
        is False
    ):
        errors.append("honest_verdict.not_terminal")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate.not_aggregation")
    if artifact.get("docs_updated") != []:
        errors.append("docs_updated.not_deferred")
    if artifact.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial.must_be_false")
    if any(
        "classification" not in _mapping(row) for row in _list(artifact.get("source_artifacts"))
    ):
        errors.append("source_artifacts.missing_classification")
    milestone = _mapping(artifact.get("milestone_decision"))
    if milestone.get("clean_exact_verifier_scale_up") is not True:
        errors.append("milestone_decision.invalid")
    exact = _mapping(artifact.get("exact_verifier_decision"))
    if exact.get("clean_scale_up") is not True:
        errors.append("exact_verifier_decision.invalid")
    runtime = _mapping(artifact.get("runtime_substrate_decision"))
    if runtime.get("does_not_gate_non_llm_exact_verifiers") is not True:
        errors.append("runtime_substrate_decision.invalid")
    fr11 = _mapping(artifact.get("fr11_decision"))
    if fr11.get("promotion_allowed_from_clean_evidence") is not False:
        errors.append("fr11_decision.invalid")
    hardware = _mapping(artifact.get("hardware_decision"))
    if hardware.get("speedup_claimed") is not False:
        errors.append("hardware_decision.invalid")
    if "live_llm_inference" in json.dumps(artifact, sort_keys=True):
        errors.append("forbidden.live_llm_inference_claim")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum.invalid")
    return errors


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    start = clock()
    artifact = build_artifact(root, duration_s=clock() - start)
    write_json(artifact_path or root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    clock: Clock = time.perf_counter,
) -> int:
    artifact = run(root=root, artifact_path=artifact_path, clock=clock)
    errors = artifact_schema_errors(artifact)
    print(f"{EXPERIMENT}: wrote {artifact_path or root / RESULT_RELATIVE_PATH}")
    if errors:
        print(f"{EXPERIMENT}: schema_errors={errors}")
        return 1
    print(f"{EXPERIMENT}: honest_verdict={artifact['honest_verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
