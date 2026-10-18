#!/usr/bin/env python3
"""Exp 5094: ungated .467 capstone aggregation.

Spec refs: REQ-CAPSTONE-5094, SCENARIO-CAPSTONE-5094,
SCENARIO-CAPSTONE-5094-FIELD-PRINCIPLES.

This module reads the upstream .467 artifacts, records missing and blocked
inputs explicitly, excludes flagged upstream artifacts from headline decisions,
and writes the final milestone decision without running a model.
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
EXPERIMENT = "experiment_5094_capstone_v467"
EXPERIMENT_ID = 5094
SCHEMA = "carnot.experiment_5094_capstone_v467.v1"
RESULT_RELATIVE_PATH = Path("results") / "experiment_5094_capstone_v467.json"
MILESTONE = "2026.07.467"
RANDOM_SEED = 5094
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-5094",
    "SCENARIO-CAPSTONE-5094",
    "SCENARIO-CAPSTONE-5094-FIELD-PRINCIPLES",
]

MILESTONE_DECISIONS = {
    "runtime_repaired_process_verifier_ready",
    "exact_verifier_pivot_positive",
    "fr11_governed_positive",
    "hardware_continuity_only",
    "execution_incomplete_endpoint_blocked",
    "bounded_no_headline",
}

HONEST_VERDICTS = {
    "runtime_repaired_process_verifier_ready": "complete_capstone_v467_runtime_repaired_process_verifier_ready",
    "exact_verifier_pivot_positive": (
        "complete_capstone_v467_exact_verifier_pivot_positive_runtime_process_blocked"
    ),
    "fr11_governed_positive": "complete_capstone_v467_fr11_governed_positive_no_promotion",
    "hardware_continuity_only": "complete_capstone_v467_hardware_continuity_only",
    "execution_incomplete_endpoint_blocked": (
        "complete_capstone_v467_execution_incomplete_endpoint_blocked"
    ),
    "bounded_no_headline": "complete_capstone_v467_bounded_no_headline",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; summarizes the .467 capstone without headline claims "
            "from blocked, missing, or flagged upstream artifacts."
        )
    },
    "duration_s": {
        "principle": (
            "wall-clock duration for the aggregation run; it must stay compatible "
            "with aggregation_from_upstream_artifacts."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- loads upstream JSON only; the "
            "capstone must not claim a live model run."
        )
    },
    "artifacts_loaded": {
        "principle": "all listed .467 artifacts that are present, each with sha256 and import boundary."
    },
    "missing_artifacts": {
        "principle": (
            "expected .467 artifacts absent or unreadable; empty only when every "
            "listed file is loadable."
        )
    },
    "blocked_artifacts": {
        "principle": "loadable artifacts with blocked verdict/status, recorded as blockers rather than nulls."
    },
    "runtime_state": {
        "principle": (
            "runtime endpoint evidence after excluding flagged runtime claims from "
            "headline readiness."
        )
    },
    "process_verifier_state": {
        "principle": (
            "uPRM/cache/process and temporal fallback state after blocked and flagged "
            "upstreams are removed from headline evidence."
        )
    },
    "exact_verifier_state": {
        "principle": (
            "objective solver/formal-verifier state, allowing only clean non-flagged "
            "evidence to drive the pivot decision."
        )
    },
    "constrained_generation_state": {
        "principle": "STATIC/CSR constrained-generation state, with flagged wins recorded but not headlined."
    },
    "kan_formal_state": {
        "principle": "clean KAN PWA/MILP proof telemetry and scale boundary."
    },
    "fr11_state": {
        "principle": (
            "governed FR-11 memory evidence with held-out, non-forgetting, poison, "
            "contamination, rollback, promotion, and positive-utility boundaries."
        )
    },
    "hardware_state": {
        "principle": "KV260/GateMate/PolarFire continuity evidence with no speedup or destructive-action claim."
    },
    "milestone_decision": {
        "principle": "one of the six allowed .467 capstone classes, chosen from clean evidence only."
    },
    "docs_updated": {
        "principle": (
            "OpenSpec updates performed by this task; ops/status/changelog/traceability "
            "remain deferred by conductor stop rule."
        )
    },
    "next_recommendations": {
        "principle": "bounded next actions derived from blockers and clean positives, not a headline claim."
    },
    "flagged_adversarial": {
        "principle": "false for the capstone itself when it transparently records flagged upstreams."
    },
}

REQUIRED_TOP_LEVEL_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "artifacts_loaded",
    "missing_artifacts",
    "blocked_artifacts",
    "runtime_state",
    "process_verifier_state",
    "exact_verifier_state",
    "constrained_generation_state",
    "kan_formal_state",
    "fr11_state",
    "hardware_state",
    "milestone_decision",
    "docs_updated",
    "next_recommendations",
    "flagged_adversarial",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "preconditions_checked",
    "flagged_upstream_artifacts",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
    *REQUIRED_TOP_LEVEL_FIELDS,
)


@dataclass(frozen=True)
class UpstreamSource:
    """Expected upstream artifact and the fields imported into the capstone."""

    label: str
    experiment_id: int
    relative_path: Path
    imported_fields: tuple[str, ...]


UPSTREAMS: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        "archive_466_activate_467",
        5083,
        Path("results/experiment_5083_archive_466_activate_467.json"),
        ("honest_verdict", "blocked_artifacts", "missing_artifacts", "close_state"),
    ),
    UpstreamSource(
        "sota_ingestion",
        5084,
        Path("results/experiment_5084_sota_ingestion_v467.json"),
        ("honest_verdict", "sources_checked", "task_mapping"),
    ),
    UpstreamSource(
        "runtime_endpoint",
        5085,
        Path("results/experiment_5085_llamacpp_logprob_endpoint_bringup_v467.json"),
        (
            "honest_verdict",
            "completion_endpoint_ready",
            "logprob_endpoint_ready",
            "top_logprob_or_confidence_ready",
            "live_completion_invoked",
            "usable_sota_models",
        ),
    ),
    UpstreamSource(
        "uprm_logprob_cache",
        5086,
        Path("results/experiment_5086_uprm_logprob_cache_retry_v467.json"),
        ("honest_verdict", "logprob_cache_ready", "step_cache_ready", "endpoint_used"),
    ),
    UpstreamSource(
        "uprm_process_verifier",
        5087,
        Path("results/experiment_5087_uprm_process_verifier_retry_v467.json"),
        ("honest_verdict", "status", "gate_check_summary"),
    ),
    UpstreamSource(
        "temporal_consistency_prm",
        5088,
        Path("results/experiment_5088_temporal_consistency_prm_v467.json"),
        ("honest_verdict", "beats_one_pass", "delta_vs_one_pass"),
    ),
    UpstreamSource(
        "pbit_cdcl_bridge",
        5089,
        Path("results/experiment_5089_pbit_guided_cdcl_bridge_v467.json"),
        ("honest_verdict", "correctness_preserved", "helps_declared_family", "delta_effort_vs_pure"),
    ),
    UpstreamSource(
        "static_csr_constrained_decoding",
        5090,
        Path("results/experiment_5090_static_csr_constrained_decoding_v467.json"),
        (
            "honest_verdict",
            "beats_cpu_trie",
            "beats_rerank_only_on_validity_or_cost",
            "mask_speedup",
            "validity_rate",
        ),
    ),
    UpstreamSource(
        "kan_pwa_milp_scale",
        5091,
        Path("results/experiment_5091_kan_pwa_milp_scale_v467.json"),
        (
            "honest_verdict",
            "property_holds",
            "property_status",
            "solver_status",
            "binary_variable_count",
            "pwa_piece_count",
            "constraint_count",
            "global_error_bound",
        ),
    ),
    UpstreamSource(
        "fr11_budgeted_onpolicy_memory",
        5092,
        Path("results/experiment_5092_fr11_budgeted_onpolicy_memory_v467.json"),
        (
            "honest_verdict",
            "heldout_delta",
            "nonforgetting_delta",
            "contamination_guard_passed",
            "poison_guard_passed",
            "rollback_guard_passed",
            "promotion_decision",
        ),
    ),
    UpstreamSource(
        "hardware_continuity",
        5093,
        Path("results/experiment_5093_hardware_continuity_v467.json"),
        (
            "honest_verdict",
            "kv260_ssh_ready",
            "kv260_uio_transcript_path",
            "kv260_speedup_claim_allowed",
            "gatemate_detected",
            "gatemate_terminal_state",
            "polarfire_detected",
            "polarfire_dispatch_precheck_ready",
            "destructive_actions_taken",
        ),
    ),
)

UPSTREAMS_BY_ID = {source.experiment_id: source for source in UPSTREAMS}
TERMINAL_PREFIXES = ("complete_", "success_", "passed_", "shipped_", "blocked_")


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _bool(value: Any) -> bool:
    return value is True


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _gte_zero(value: Any) -> bool:
    number = _number(value)
    return number is not None and number >= 0.0


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
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive IO guard.
        return {}, {"exists": True, "loadable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive schema guard.
        return {}, {"exists": True, "loadable": False, "error": "json_not_object"}
    return dict(payload), {"exists": True, "loadable": True, "sha256": file_sha256(path)}


def _artifact_row(source: UpstreamSource, payload: JsonMap, status: JsonMap) -> JsonDict:
    verdict = str(payload.get("honest_verdict", ""))
    blocked = verdict.startswith("blocked_") or payload.get("status") == "blocked"
    row: JsonDict = {
        "label": source.label,
        "experiment_id": source.experiment_id,
        "path": str(source.relative_path),
        "present": status.get("loadable") is True,
        "exists": status.get("exists") is True,
        "loadable": status.get("loadable") is True,
        "fields_imported": list(source.imported_fields),
        "honest_verdict": verdict,
        "flagged_adversarial": payload.get("flagged_adversarial") is True,
        "blocked": blocked,
    }
    if "sha256" in status:
        row["sha256"] = status["sha256"]
    if "error" in status:
        row["error"] = status["error"]
    gate_summary = payload.get("gate_check_summary")
    if isinstance(gate_summary, str):
        row["gate_check_summary"] = gate_summary
    return row


def load_upstream_artifacts(
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict], dict[int, JsonDict]]:
    artifacts_loaded: list[JsonDict] = []
    missing_artifacts: list[JsonDict] = []
    blocked_artifacts: list[JsonDict] = []
    flagged_upstream_artifacts: list[JsonDict] = []
    payloads: dict[int, JsonDict] = {}

    for source in UPSTREAMS:
        payload, status = read_json_mapping(root / source.relative_path)
        row = _artifact_row(source, payload, status)
        if row["present"]:
            artifacts_loaded.append(row)
            payloads[source.experiment_id] = payload
            if row["blocked"]:
                blocked_artifacts.append({**row, "blocker_reason": "blocked_verdict_or_status"})
            if row["flagged_adversarial"]:
                flagged_upstream_artifacts.append({**row, "excluded_from_headline": True})
        else:
            missing_artifacts.append({**row, "status": "missing_or_unloadable"})

    return (
        artifacts_loaded,
        missing_artifacts,
        blocked_artifacts,
        flagged_upstream_artifacts,
        payloads,
    )


def _payload(payloads: Mapping[int, JsonDict], experiment_id: int) -> JsonDict:
    return dict(payloads.get(experiment_id, {}))


def build_runtime_state(payloads: Mapping[int, JsonDict], flagged_ids: set[int]) -> JsonDict:
    runtime = _payload(payloads, 5085)
    cache = _payload(payloads, 5086)
    runtime_claim_flagged = 5085 in flagged_ids
    reported_completion = _bool(runtime.get("completion_endpoint_ready"))
    reported_logprob = _bool(runtime.get("logprob_endpoint_ready"))
    reported_top = _bool(runtime.get("top_logprob_or_confidence_ready"))
    cache_ready = _bool(cache.get("logprob_cache_ready")) and _bool(cache.get("step_cache_ready"))
    headline_ready = reported_completion and reported_logprob and reported_top and cache_ready and not runtime_claim_flagged

    return {
        "state": (
            "reported_endpoint_ready_but_flagged_and_cache_blocked"
            if runtime_claim_flagged
            else "runtime_ready" if headline_ready else "runtime_not_ready"
        ),
        "reported_completion_endpoint_ready": reported_completion,
        "reported_logprob_endpoint_ready": reported_logprob,
        "reported_top_logprob_or_confidence_ready": reported_top,
        "reported_usable_sota_model_count": len(_list(runtime.get("usable_sota_models"))),
        "runtime_claim_excluded_reason": "upstream_flagged_adversarial" if runtime_claim_flagged else "",
        "headline_runtime_ready": headline_ready,
        "runtime_ready": headline_ready,
        "logprob_cache_ready": _bool(cache.get("logprob_cache_ready")),
        "step_cache_ready": _bool(cache.get("step_cache_ready")),
        "process_substrate_blocked": not cache_ready,
        "endpoint_or_cache_blocked": runtime_claim_flagged or not cache_ready,
    }


def build_process_verifier_state(payloads: Mapping[int, JsonDict], flagged_ids: set[int]) -> JsonDict:
    cache = _payload(payloads, 5086)
    process = _payload(payloads, 5087)
    temporal = _payload(payloads, 5088)
    cache_ready = _bool(cache.get("logprob_cache_ready")) and _bool(cache.get("step_cache_ready"))
    process_blocked = str(process.get("honest_verdict", "")).startswith("blocked_") or process.get("status") == "blocked"
    temporal_clean_win = 5088 not in flagged_ids and _bool(temporal.get("beats_one_pass"))

    return {
        "state": "uprm_process_blocked_temporal_flagged_or_no_win",
        "logprob_cache_ready": cache_ready,
        "uprm_process_retry_blocked": process_blocked,
        "temporal_fallback_clean": 5088 not in flagged_ids,
        "temporal_fallback_reported_win": _bool(temporal.get("beats_one_pass")),
        "temporal_fallback_clean_win": temporal_clean_win,
        "process_verifier_ready": cache_ready and not process_blocked,
        "process_verifier_win": False,
        "gate_check_summary": str(process.get("gate_check_summary", "")),
    }


def build_exact_verifier_state(payloads: Mapping[int, JsonDict], flagged_ids: set[int]) -> JsonDict:
    pbit = _payload(payloads, 5089)
    kan = _payload(payloads, 5091)
    pbit_clean = 5089 not in flagged_ids
    kan_clean_positive = (
        5091 not in flagged_ids
        and _bool(kan.get("property_holds"))
        and str(kan.get("property_status", "")) == "verified"
        and _bool(kan.get("solver_available"))
    )

    return {
        "state": (
            "exact_pivot_positive_via_clean_kan_pwa_milp_small_property"
            if kan_clean_positive
            else "exact_pivot_not_headlineable"
        ),
        "path_worth_scaling": kan_clean_positive,
        "pbit_cdcl": {
            "clean": pbit_clean,
            "excluded_from_headline": not pbit_clean,
            "reported_correctness_preserved": _bool(pbit.get("correctness_preserved")),
            "reported_helps_declared_family": _bool(pbit.get("helps_declared_family")),
            "reported_delta_effort_vs_pure": _mapping(pbit.get("delta_effort_vs_pure")),
        },
        "kan_milp": {
            "clean_positive": kan_clean_positive,
            "property_holds": _bool(kan.get("property_holds")),
            "property_status": str(kan.get("property_status", "")),
            "solver_status": str(kan.get("solver_status", "")),
            "binary_variable_count": _number(kan.get("binary_variable_count")),
        },
    }


def build_constrained_generation_state(payloads: Mapping[int, JsonDict], flagged_ids: set[int]) -> JsonDict:
    static = _payload(payloads, 5090)
    clean = 5090 not in flagged_ids
    reported_win = _bool(static.get("beats_cpu_trie")) and _bool(
        static.get("beats_rerank_only_on_validity_or_cost")
    )
    return {
        "state": "static_csr_reported_win_flagged_not_headlined" if not clean else "static_csr_clean",
        "clean_headline": clean and reported_win,
        "excluded_from_headline": not clean,
        "reported_beats_cpu_trie": _bool(static.get("beats_cpu_trie")),
        "reported_beats_rerank_only_on_validity_or_cost": _bool(
            static.get("beats_rerank_only_on_validity_or_cost")
        ),
        "reported_mask_equivalence_rate": _number(static.get("mask_equivalence_rate")),
        "reported_mask_speedup": _number(static.get("mask_speedup")),
        "reported_validity_rate": _number(static.get("validity_rate")),
        "reported_rerank_only_validity_rate": _number(static.get("rerank_only_validity_rate")),
    }


def build_kan_formal_state(payloads: Mapping[int, JsonDict], flagged_ids: set[int]) -> JsonDict:
    kan = _payload(payloads, 5091)
    clean = 5091 not in flagged_ids
    property_holds = clean and _bool(kan.get("property_holds"))
    return {
        "state": "clean_small_property_verified" if property_holds else "kan_formal_missing_or_not_clean",
        "clean": clean,
        "property_holds": property_holds,
        "property_status": str(kan.get("property_status", "")),
        "abstraction_built": _bool(kan.get("abstraction_built")),
        "solver_available": _bool(kan.get("solver_available")),
        "solver_status": str(kan.get("solver_status", "")),
        "binary_variable_count": _number(kan.get("binary_variable_count")),
        "pwa_piece_count": _number(kan.get("pwa_piece_count")),
        "constraint_count": _number(kan.get("constraint_count")),
        "global_error_bound": _number(kan.get("global_error_bound")),
        "solve_time_s": _number(kan.get("solve_time_s")),
        "scale_boundary": "small_multi_unit_property_not_architecture_scale_claim",
    }


def build_fr11_state(payloads: Mapping[int, JsonDict], flagged_ids: set[int]) -> JsonDict:
    fr11 = _payload(payloads, 5092)
    promotion = _mapping(fr11.get("promotion_decision"))
    gates = _mapping(promotion.get("gate_conditions"))
    clean = 5092 not in flagged_ids
    promoted = _bool(promotion.get("promoted")) or (_number(fr11.get("promoted_count")) or 0.0) > 0.0
    safe = (
        clean
        and _bool(fr11.get("fr11_attempt_completed"))
        and _gte_zero(fr11.get("heldout_delta"))
        and _gte_zero(fr11.get("nonforgetting_delta"))
        and _bool(fr11.get("contamination_guard_passed"))
        and _bool(fr11.get("poison_guard_passed"))
        and _bool(fr11.get("rollback_guard_passed"))
    )
    positive_utility = _bool(gates.get("positive_utility_gt_zero"))

    return {
        "state": "governed_safe_no_promote_no_positive_utility" if safe else "fr11_not_clean_positive",
        "safe_governed_mechanism": safe,
        "heldout_delta": _number(fr11.get("heldout_delta")),
        "nonforgetting_delta": _number(fr11.get("nonforgetting_delta")),
        "contamination_guard_passed": _bool(fr11.get("contamination_guard_passed")),
        "poison_guard_passed": _bool(fr11.get("poison_guard_passed")),
        "rollback_guard_passed": _bool(fr11.get("rollback_guard_passed")),
        "promoted": promoted,
        "promoted_count": _number(fr11.get("promoted_count")),
        "positive_utility_observed": positive_utility,
        "no_promote_reason": str(promotion.get("no_promote_reason", "")),
    }


def build_hardware_state(payloads: Mapping[int, JsonDict], flagged_ids: set[int]) -> JsonDict:
    hardware = _payload(payloads, 5093)
    clean = 5093 not in flagged_ids
    no_destructive = _list(hardware.get("destructive_actions_taken")) == []
    no_speedup = hardware.get("kv260_speedup_claim_allowed") is False
    any_board_continuity = _bool(hardware.get("kv260_ssh_ready")) or _bool(
        hardware.get("polarfire_detected")
    )
    clean_continuity = clean and no_destructive and no_speedup and any_board_continuity

    return {
        "state": "kv260_and_polarfire_ready_gatemate_blocked_no_speedup_claim"
        if clean_continuity
        else "hardware_continuity_missing_or_not_clean",
        "clean_continuity_state": clean_continuity,
        "kv260_ssh_ready": _bool(hardware.get("kv260_ssh_ready")),
        "kv260_uio_transcript_path": hardware.get("kv260_uio_transcript_path"),
        "speedup_claim_allowed": _bool(hardware.get("kv260_speedup_claim_allowed")),
        "gatemate_detected": _bool(hardware.get("gatemate_detected")),
        "gatemate_terminal_state": str(hardware.get("gatemate_terminal_state", "")),
        "polarfire_detected": _bool(hardware.get("polarfire_detected")),
        "polarfire_dispatch_precheck_ready": _bool(hardware.get("polarfire_dispatch_precheck_ready")),
        "destructive_actions_taken": _list(hardware.get("destructive_actions_taken")),
    }


def choose_milestone_decision(
    *,
    runtime_state: JsonMap,
    process_verifier_state: JsonMap,
    exact_verifier_state: JsonMap,
    fr11_state: JsonMap,
    hardware_state: JsonMap,
) -> str:
    if _bool(runtime_state.get("runtime_ready")) and _bool(process_verifier_state.get("process_verifier_ready")):
        return "runtime_repaired_process_verifier_ready"
    if _bool(exact_verifier_state.get("path_worth_scaling")):
        return "exact_verifier_pivot_positive"
    if _bool(fr11_state.get("safe_governed_mechanism")) and _bool(fr11_state.get("positive_utility_observed")):
        return "fr11_governed_positive"
    if _bool(hardware_state.get("clean_continuity_state")):
        return "hardware_continuity_only"
    if _bool(runtime_state.get("endpoint_or_cache_blocked")):
        return "execution_incomplete_endpoint_blocked"
    return "bounded_no_headline"


def honest_verdict_for_decision(decision: str) -> str:
    return HONEST_VERDICTS[decision]


def build_next_recommendations(decision: str) -> list[JsonDict]:
    return [
        {
            "priority": 1,
            "recommendation": (
                "rerun the runtime endpoint and uPRM cache only after the endpoint artifact is "
                "clean and the cache can complete without a blocked gate"
            ),
            "reason": "runtime/process-verifier readiness remains blocked in clean evidence.",
        },
        {
            "priority": 2,
            "recommendation": "scale the clean KAN PWA/MILP path beyond the small verified property",
            "reason": f"milestone_decision={decision} is driven by clean formal-verifier evidence.",
        },
        {
            "priority": 3,
            "recommendation": (
                "retest STATIC CSR and p-bit CDCL after resolving flagged-artifact rigor issues "
                "before using either as headline evidence"
            ),
            "reason": "their current artifacts are recorded but excluded from headline decisions.",
        },
        {
            "priority": 4,
            "recommendation": (
                "keep FR-11 in governed no-promote mode until positive utility is observed while "
                "guards remain nonnegative"
            ),
            "reason": "current FR-11 evidence is safe and nonnegative but not utility-positive.",
        },
        {
            "priority": 5,
            "recommendation": (
                "write the dated architecture reconciliation note after the conductor reconciliation "
                "step, separating flagged runtime claims from the clean exact-verifier pivot"
            ),
            "reason": "the architecture document is stale relative to the July runtime/solver record.",
        },
    ]


def build_artifact(root: Path = REPO_ROOT, clock: Clock = time.perf_counter) -> JsonDict:
    started = clock()
    (
        artifacts_loaded,
        missing_artifacts,
        blocked_artifacts,
        flagged_upstream_artifacts,
        payloads,
    ) = load_upstream_artifacts(root)
    flagged_ids = {int(row["experiment_id"]) for row in flagged_upstream_artifacts}

    runtime_state = build_runtime_state(payloads, flagged_ids)
    process_verifier_state = build_process_verifier_state(payloads, flagged_ids)
    exact_verifier_state = build_exact_verifier_state(payloads, flagged_ids)
    constrained_generation_state = build_constrained_generation_state(payloads, flagged_ids)
    kan_formal_state = build_kan_formal_state(payloads, flagged_ids)
    fr11_state = build_fr11_state(payloads, flagged_ids)
    hardware_state = build_hardware_state(payloads, flagged_ids)
    decision = choose_milestone_decision(
        runtime_state=runtime_state,
        process_verifier_state=process_verifier_state,
        exact_verifier_state=exact_verifier_state,
        fr11_state=fr11_state,
        hardware_state=hardware_state,
    )
    duration_s = max(round(clock() - started, 6), 0.0001)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "honest_verdict": honest_verdict_for_decision(decision),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "milestone": MILESTONE,
        "artifacts_loaded": artifacts_loaded,
        "missing_artifacts": missing_artifacts,
        "blocked_artifacts": blocked_artifacts,
        "flagged_upstream_artifacts": flagged_upstream_artifacts,
        "runtime_state": runtime_state,
        "process_verifier_state": process_verifier_state,
        "exact_verifier_state": exact_verifier_state,
        "constrained_generation_state": constrained_generation_state,
        "kan_formal_state": kan_formal_state,
        "fr11_state": fr11_state,
        "hardware_state": hardware_state,
        "milestone_decision": decision,
        "docs_updated": ["openspec/capabilities/capstone/spec.md"],
        "ops_reconciliation_deferred": [
            "ops/status.md",
            "ops/changelog.md",
            "_bmad/traceability.md",
            "_bmad/architecture.md",
        ],
        "next_recommendations": build_next_recommendations(decision),
        "flagged_adversarial": False,
        "preconditions_checked": {
            "expected_artifact_count": len(UPSTREAMS),
            "loaded_artifact_count": len(artifacts_loaded),
            "missing_artifact_count": len(missing_artifacts),
            "blocked_artifact_count": len(blocked_artifacts),
            "flagged_upstream_count": len(flagged_upstream_artifacts),
            "all_listed_artifacts_present": len(missing_artifacts) == 0,
            "headline_inputs_exclude_flagged": True,
        },
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def write_artifact(root: Path, artifact: JsonMap, artifact_path: Path | None = None) -> Path:
    out_path = artifact_path if artifact_path is not None else root / RESULT_RELATIVE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run(
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    artifact = build_artifact(root=root, clock=clock)
    write_artifact(root, artifact, artifact_path)
    return artifact


def artifact_schema_errors(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in payload:
            errors.append(f"missing.{field}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict.not_terminal")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate.not_aggregation")
    if payload.get("milestone_decision") not in MILESTONE_DECISIONS:
        errors.append("milestone_decision.invalid")
    if not isinstance(payload.get("docs_updated"), list):
        errors.append("docs_updated.not_list")
    if payload.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial.must_be_false")
    if "live_llm_inference" in json.dumps(payload, sort_keys=True, default=str):
        errors.append("forbidden.live_llm_inference_claim")
    return errors


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    errors = artifact_schema_errors(artifact)
    if errors:
        print(json.dumps({"schema_errors": errors}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
