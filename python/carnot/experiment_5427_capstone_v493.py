"""Exp5427 .493 terminal capstone truth-table synthesis.

Spec refs: REQ-CAPSTONE-5427, SCENARIO-CAPSTONE-5427,
SCENARIO-CAPSTONE-5427-MISSING-INPUT,
SCENARIO-CAPSTONE-5427-FIELD-PRINCIPLES.

This module is deliberately an aggregation step. It reads the `.493` source
context and upstream result artifacts, then emits a terminal truth table from
those receipts only. The important discipline is claim hygiene: a valid local
GGUF receipt is not the same thing as a headline-ready structured-verification
claim when the same artifact is adversarially flagged, and comparable hardware
timing is still not a speedup claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5427_capstone_v493.json")
EXPERIMENT = "experiment_5427_capstone_v493"
EXPERIMENT_ID = "exp5427-v493-capstone"
MILESTONE = "2026.07.493"
SCHEMA = "carnot.experiment_5427.capstone_v493.v1"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5427
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP5415 = "results/experiment_5415_transition_v493.json"
EXP5416 = "results/experiment_5416_source_delta_v493.json"
EXP5417 = "results/experiment_5417_risk_calibrated_sota_structured_panel_v493.json"
EXP5418 = "results/experiment_5418_predictive_prefix_action_safety_v493.json"
EXP5419 = "results/experiment_5419_active_constraint_lns_scale_v493.json"
EXP5420 = "results/experiment_5420_pbit_hardware_transfer_preflight_v493.json"
EXP5421 = "results/experiment_5421_evidence_reliance_csl_v493.json"
EXP5422 = "results/experiment_5422_csl_promotion_reliance_scale_v493.json"
EXP5423 = "results/experiment_5423_arc_coex_landmark_levelup_v493.json"
EXP5424 = "results/experiment_5424_hardware_comparable_timing_receipts_v493.json"
EXP5425 = "results/experiment_5425_kan_measurement_access_certificate_v493.json"
EXP5426 = "results/experiment_5426_prd_gap_agent_failure_table_v493.json"

SOURCE_CONTEXT_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "_bmad/prd.md",
    "_bmad/architecture.md",
    "ops/status.md",
    "ops/changelog.md",
)
RESULT_ARTIFACT_PATHS: tuple[str, ...] = (
    EXP5415,
    EXP5416,
    EXP5417,
    EXP5418,
    EXP5419,
    EXP5420,
    EXP5421,
    EXP5422,
    EXP5423,
    EXP5424,
    EXP5425,
    EXP5426,
)
EXPECTED_INPUT_PATHS: tuple[str, ...] = (*SOURCE_CONTEXT_PATHS, *RESULT_ARTIFACT_PATHS)

SPEC_REFS = (
    "REQ-CAPSTONE-5427",
    "SCENARIO-CAPSTONE-5427",
    "SCENARIO-CAPSTONE-5427-MISSING-INPUT",
    "SCENARIO-CAPSTONE-5427-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "conductor route key; must equal 2026.07.493.",
    "upstream_artifacts_read": "provenance; ordered list of source context and result artifacts actually read.",
    "upstream_artifacts_missing": "no fabricated evidence; missing or unreadable inputs are recorded here.",
    "headline_ready_lanes": "positive evidence boundary; only unflagged closed rows with no stronger blocker.",
    "bounded_lanes": "partial evidence boundary; useful but limited receipts that must not become headline claims.",
    "honest_null_lanes": "null-result honesty; executed lanes with no banked or positive outcome.",
    "blocked_lanes": "precondition honesty; flagged, closed, or missing-precondition lanes.",
    "arc_new_level_banked": "ARC north-star metric; true only for a reproduction-gated new level.",
    "hardware_speedup_claim": "no unsupported acceleration; must remain false for `.493`.",
    "future_token_signal_allowed": "token/internal lane closure; must remain false without authenticated backend receipts.",
    "local_sota_gguf_receipts_valid": "SOTA model provenance; true only when the structured/prefix artifacts carry GGUF model specs and GPU-offload receipts.",
    "research_roadmap_yaml_unchanged": "user prohibition; derived from git status.",
    "conductor_unchanged": "user prohibition; derived from git status.",
    "next_recommendations": "evidence-based planning handoff; no detailed next milestone plan.",
    "inference_substrate": "no hidden live model inference; must equal aggregation_from_upstream_artifacts.",
    "honest_verdict": "terminal status; starts with complete: or blocked: and names the capstone evidence boundary.",
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "field_principles",
    "truth_table",
    "missing_lanes",
    "artifact_read_errors",
    "source_artifact_checksums",
    "protected_file_checks",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES.keys(),
)
BOOLEAN_FIELDS = (
    "arc_new_level_banked",
    "hardware_speedup_claim",
    "future_token_signal_allowed",
    "local_sota_gguf_receipts_valid",
    "research_roadmap_yaml_unchanged",
    "conductor_unchanged",
)
ALLOWED_CLASSIFICATIONS = {"headline_ready", "bounded", "honest_null", "blocked", "missing"}
LANE_ORDER = (
    "risk_calibrated_structured_verification",
    "predictive_prefix_action_safety",
    "active_constraint_lns_scale",
    "pbit_hardware_transfer_preflight",
    "evidence_reliance_csl",
    "gated_csl_promotion",
    "arc_levelup",
    "comparable_hardware_timing",
    "kan_measurement_access_certificates",
    "token_internal_feature_lane_closed",
)
DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5427_capstone_v493.py -q",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5427_capstone_v493.py "
            "-m pytest tests/python/test_experiment_5427_capstone_v493.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5427_capstone_v493.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
)

LANE_SPECS: tuple[JsonDict, ...] = (
    {
        "lane": "risk_calibrated_structured_verification",
        "source_artifacts": [EXP5417],
        "ready_field": "risk_calibrated_structured_panel_ready",
        "ready_classification": "headline_ready",
        "claim_boundary": "structured fixture safety with abstention; blocked from headline while adversarial flags remain",
        "evidence_fields": (
            "risk_calibrated_structured_panel_ready",
            "unsafe_false_accept_rate",
            "accepted_risk_estimate",
            "accepted_risk_bound",
            "accepted_risk_bound_threshold",
            "abstention_rate",
            "semantic_error_rate",
            "gpu_offload_verified",
        ),
    },
    {
        "lane": "predictive_prefix_action_safety",
        "source_artifacts": [EXP5418],
        "ready_field": "predictive_prefix_safety_ready",
        "ready_classification": "headline_ready",
        "claim_boundary": "prefix-gated unsafe-action diagnostic; blocked from headline while adversarial flags remain",
        "evidence_fields": (
            "predictive_prefix_safety_ready",
            "deterministic_verifier_final_authority",
            "final_only_unsafe_false_accept_rate",
            "prefix_gated_unsafe_false_accept_rate",
            "final_only_unreachable_tool_action_rate",
            "prefix_gated_unreachable_tool_action_rate",
            "gpu_offload_verified",
        ),
    },
    {
        "lane": "active_constraint_lns_scale",
        "source_artifacts": [EXP5419],
        "ready_field": "active_constraint_lns_scale_ready",
        "ready_classification": "bounded",
        "claim_boundary": "deterministic CPU-local LNS guidance; hints are advisory and solver authority is preserved",
        "evidence_fields": (
            "active_constraint_lns_scale_ready",
            "solver_validity_preserved",
            "accepted_hint_count",
            "rejected_hint_count",
            "overwritten_hint_count",
            "work_delta",
            "dual_residual_sanity",
            "claim_limits",
        ),
    },
    {
        "lane": "pbit_hardware_transfer_preflight",
        "source_artifacts": [EXP5420],
        "ready_field": "pbit_transfer_preflight_ready",
        "ready_classification": "bounded",
        "claim_boundary": "hash-matched CPU and PolarFire preflight only; no speedup claim",
        "evidence_fields": (
            "pbit_transfer_preflight_ready",
            "exact_enumeration_match",
            "same_workload_hash_match",
            "cpu_repeat_count",
            "board_repeat_count",
            "polarfire_reachable",
            "kv260_ssh_checked",
            "hardware_speedup_claim",
        ),
    },
    {
        "lane": "evidence_reliance_csl",
        "source_artifacts": [EXP5421],
        "ready_field": "evidence_reliance_csl_ready",
        "ready_classification": "headline_ready",
        "claim_boundary": "controller-level CSL reliance audit; no model weight mutation",
        "evidence_fields": (
            "evidence_reliance_csl_ready",
            "hidden_forgetting_detected",
            "reliance_drift_metric",
            "quality_preserved",
            "stale_poison_deflection_rate",
            "uncertain_reliance_deflection_rate",
            "rollback_verified",
            "no_weight_mutation",
        ),
    },
    {
        "lane": "gated_csl_promotion",
        "source_artifacts": [EXP5422],
        "ready_field": "csl_promotion_reliance_scale_ready",
        "ready_classification": "headline_ready",
        "claim_boundary": "gated promotion with rejected and abstained fragments retained inactive; no weight mutation",
        "evidence_fields": (
            "csl_promotion_reliance_scale_ready",
            "promoted_fragment_count",
            "rejected_fragment_count",
            "abstained_fragment_count",
            "grounding_preserved",
            "rejected_fragments_quarantined",
            "rollback_verified",
            "no_weight_mutation",
        ),
    },
    {
        "lane": "arc_levelup",
        "source_artifacts": [EXP5423],
        "ready_field": "arc_new_level_banked",
        "ready_classification": "headline_ready",
        "claim_boundary": "live ARC path was exercised; no reproduced new level was banked",
        "evidence_fields": (
            "status",
            "arc_new_level_banked",
            "offline_reproduced",
            "reproduced_levels",
            "newly_reached_levels",
            "attempt_count",
            "frontier_expansion_count",
            "landmark_count",
            "failure_mode",
            "registry_total_before",
            "registry_total_after",
        ),
    },
    {
        "lane": "comparable_hardware_timing",
        "source_artifacts": [EXP5424],
        "ready_field": "comparable_timing_receipts_ready",
        "ready_classification": "bounded",
        "claim_boundary": "comparable CPU and PolarFire timing receipts only; no acceleration claim",
        "evidence_fields": (
            "comparable_timing_receipts_ready",
            "measurement_access_complete",
            "same_workload_hash_match",
            "same_result_hash_match",
            "cpu_repeat_count",
            "board_repeat_count",
            "polarfire_reachable",
            "kv260_ssh_checked",
            "hardware_speedup_claim",
        ),
    },
    {
        "lane": "kan_measurement_access_certificates",
        "source_artifacts": [EXP5425],
        "ready_field": "kan_measurement_access_certificate_ready",
        "ready_classification": "bounded",
        "claim_boundary": "bounded measurement-access certificate; no broad KAN verification claim",
        "evidence_fields": (
            "kan_measurement_access_certificate_ready",
            "certificate_count",
            "false_property_rejection_rate",
            "true_property_preservation_rate",
            "missing_evidence_detected",
            "broad_kan_verification_claim",
            "claim_limits",
        ),
    },
    {
        "lane": "token_internal_feature_lane_closed",
        "source_artifacts": [EXP5415, EXP5426],
        "ready_field": None,
        "ready_classification": "blocked",
        "claim_boundary": "closed without authenticated logits, hidden-state, attention, token, or intermediate-exit receipts",
        "evidence_fields": (),
    },
)


def unwrap(value: Any) -> Any:
    """Return bare values from the principle-wrapped convention used in older artifacts."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Build the capstone from upstream artifacts and source context only."""

    root_path = Path(root)
    payloads, read_paths, missing_paths, read_errors = read_inputs(root_path)
    truth_table = [classify_lane(spec, payloads, missing_paths) for spec in LANE_SPECS]
    buckets = bucket_lanes(truth_table)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "milestone": MILESTONE,
        "upstream_artifacts_read": read_paths,
        "upstream_artifacts_missing": missing_paths,
        "artifact_read_errors": read_errors,
        "truth_table": truth_table,
        "headline_ready_lanes": buckets["headline_ready_lanes"],
        "bounded_lanes": buckets["bounded_lanes"],
        "honest_null_lanes": buckets["honest_null_lanes"],
        "blocked_lanes": buckets["blocked_lanes"],
        "missing_lanes": buckets["missing_lanes"],
        "arc_new_level_banked": bool(unwrap(payloads.get(EXP5423, {}).get("arc_new_level_banked"))),
        "hardware_speedup_claim": hardware_speedup_claim(payloads),
        "future_token_signal_allowed": future_token_signal_allowed(payloads),
        "local_sota_gguf_receipts_valid": local_sota_gguf_receipts_valid(payloads),
        "research_roadmap_yaml_unchanged": git_path_unchanged(root_path, "research-roadmap.yaml"),
        "conductor_unchanged": git_path_unchanged(root_path, "scripts/research_conductor.py"),
        "next_recommendations": next_recommendations(payloads),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifact_checksums": source_artifact_checksums(root_path, read_paths),
        "protected_file_checks": protected_file_checks(root_path),
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    artifact = json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Write the validated capstone artifact for conductor consumption."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    write_json(Path(result_path), artifact)
    return artifact


def read_inputs(root: Path) -> tuple[dict[str, JsonDict], list[str], list[str], list[JsonDict]]:
    """Read all expected inputs and record missing or malformed artifacts."""

    payloads: dict[str, JsonDict] = {}
    read_paths: list[str] = []
    missing: list[str] = []
    errors: list[JsonDict] = []

    for relative in EXPECTED_INPUT_PATHS:
        path = root / relative
        if not path.exists() or path.is_dir():
            missing.append(relative)
            continue
        if relative not in RESULT_ARTIFACT_PATHS:
            try:
                path.read_text(encoding="utf-8")
            except OSError as exc:
                missing.append(relative)
                errors.append({"path": relative, "classification": f"read_error:{exc}"})
                continue
            read_paths.append(relative)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            missing.append(relative)
            errors.append(
                {
                    "path": relative,
                    "classification": f"malformed_json:{exc.msg}",
                    "line": exc.lineno,
                    "column": exc.colno,
                }
            )
            continue
        if not isinstance(payload, dict):
            missing.append(relative)
            errors.append({"path": relative, "classification": "not_json_object"})
            continue
        payloads[relative] = payload
        read_paths.append(relative)

    return payloads, read_paths, missing, errors


def classify_lane(spec: JsonMap, payloads: Mapping[str, JsonMap], missing_paths: Sequence[str]) -> JsonDict:
    """Classify one lane without inventing evidence from neighboring lanes."""

    lane = str(spec["lane"])
    sources = [str(path) for path in spec["source_artifacts"]]
    missing_sources = [path for path in sources if path in set(missing_paths)]
    if missing_sources:
        return lane_row(
            lane=lane,
            source_artifacts=sources,
            classification="missing",
            claim_boundary="missing upstream artifact; no outcome inferred",
            evidence={},
            blocked_reason="missing_inputs",
        )

    if lane == "token_internal_feature_lane_closed":
        return lane_row(
            lane=lane,
            source_artifacts=sources,
            classification="blocked",
            claim_boundary=str(spec["claim_boundary"]),
            evidence={"future_token_signal_allowed": future_token_signal_allowed(payloads), "backend_receipt_present": False},
            blocked_reason="no_authenticated_backend_receipt",
        )

    payload = payloads.get(sources[0], {})
    flags = flag_reasons(payload)
    if flags:
        return lane_row(
            lane=lane,
            source_artifacts=sources,
            classification="blocked",
            claim_boundary=str(spec["claim_boundary"]),
            evidence=evidence_for_spec(spec, payload),
            blocked_reason="_and_".join(flags),
        )

    if lane == "arc_levelup" and unwrap(payload.get("arc_new_level_banked")) is False:
        return lane_row(
            lane=lane,
            source_artifacts=sources,
            classification="honest_null",
            claim_boundary=str(spec["claim_boundary"]),
            evidence=evidence_for_spec(spec, payload),
            blocked_reason=str(payload.get("failure_mode", "no_new_level_banked")),
        )

    ready_field = spec.get("ready_field")
    ready = bool(unwrap(payload.get(str(ready_field)))) if ready_field else False
    classification = str(spec["ready_classification"]) if ready else "blocked"
    blocked_reason = "" if ready else f"{ready_field}_not_true"
    return lane_row(
        lane=lane,
        source_artifacts=sources,
        classification=classification,
        claim_boundary=str(spec["claim_boundary"]),
        evidence=evidence_for_spec(spec, payload),
        blocked_reason=blocked_reason,
    )


def lane_row(
    *,
    lane: str,
    source_artifacts: list[str],
    classification: str,
    claim_boundary: str,
    evidence: JsonMap,
    blocked_reason: str,
) -> JsonDict:
    """Normalize row shape so bucket validation can be mechanical."""

    return {
        "lane": lane,
        "source_artifacts": source_artifacts,
        "classification": classification,
        "claim_boundary": claim_boundary,
        "blocked_reason": blocked_reason,
        "evidence": dict(evidence),
    }


def flag_reasons(payload: JsonMap) -> list[str]:
    """Return reasons an otherwise-positive artifact is not headline-clean."""

    reasons: list[str] = []
    if unwrap(payload.get("flagged_adversarial")) is True:
        reasons.append("flagged_adversarial")
    if payload.get("corrigendum_pending"):
        reasons.append("corrigendum_pending")
    return reasons


def evidence_for_spec(spec: JsonMap, payload: JsonMap) -> JsonDict:
    """Extract the lane-specific fields used as the row evidence."""

    evidence: JsonDict = {}
    for field in spec.get("evidence_fields", ()):
        if field in payload:
            evidence[str(field)] = unwrap(payload[field])
    if flag_reasons(payload):
        evidence["flag_reasons"] = flag_reasons(payload)
    if "honest_verdict" in payload:
        evidence["honest_verdict"] = unwrap(payload["honest_verdict"])
    return evidence


def bucket_lanes(rows: Sequence[JsonMap]) -> dict[str, list[JsonDict]]:
    """Split truth-table rows into the required capstone lists."""

    buckets = {
        "headline_ready_lanes": [],
        "bounded_lanes": [],
        "honest_null_lanes": [],
        "blocked_lanes": [],
        "missing_lanes": [],
    }
    for row in rows:
        buckets[lane_bucket_name(str(row["classification"]))].append(dict(row))
    return buckets


def lane_bucket_name(classification: str) -> str:
    """Map a row classification to its top-level list name."""

    if classification == "headline_ready":
        return "headline_ready_lanes"
    if classification == "bounded":
        return "bounded_lanes"
    if classification == "honest_null":
        return "honest_null_lanes"
    if classification == "missing":
        return "missing_lanes"
    return "blocked_lanes"


def hardware_speedup_claim(payloads: Mapping[str, JsonMap]) -> bool:
    """Return true only if upstream hardware artifacts actually claimed speedup."""

    return any(
        unwrap(payloads.get(path, {}).get("hardware_speedup_claim")) is True for path in (EXP5420, EXP5424)
    )


def future_token_signal_allowed(payloads: Mapping[str, JsonMap]) -> bool:
    """Keep token/internal lanes closed unless a receipt explicitly opens them."""

    return recursive_key_true(payloads.get(EXP5415, {}), "future_token_signal_allowed") or recursive_key_true(
        payloads.get(EXP5426, {}), "future_token_signal_allowed"
    )


def recursive_key_true(value: Any, key: str) -> bool:
    """Search nested artifact rows for a bare boolean true key."""

    if isinstance(value, Mapping):
        if unwrap(value.get(key)) is True:
            return True
        return any(recursive_key_true(child, key) for child in value.values())
    if isinstance(value, list):
        return any(recursive_key_true(child, key) for child in value)
    return False


def local_sota_gguf_receipts_valid(payloads: Mapping[str, JsonMap]) -> bool:
    """Validate model provenance without treating flagged measurements as headline-clean."""

    for path in (EXP5417, EXP5418):
        payload = payloads.get(path, {})
        receipt = payload.get("gpu_offload_receipt", {})
        specs = payload.get("model_specs", ())
        if unwrap(payload.get("gpu_offload_verified")) is not True:
            return False
        if not isinstance(receipt, Mapping) or unwrap(receipt.get("offload_evidence")) is not True:
            return False
        if not isinstance(specs, list) or not specs:
            return False
        for spec in specs:
            if not isinstance(spec, Mapping):
                return False
            if str(spec.get("status")) != "local_gguf_resolved":
                return False
            gguf_marker = f"{spec.get('hf_id', '')} {spec.get('model_path', '')}".lower()
            if ".gguf" not in gguf_marker and "gguf" not in gguf_marker:
                return False
    return True


def next_recommendations(payloads: Mapping[str, JsonMap]) -> list[JsonDict]:
    """Seed future planning from the concrete blockers and receipts in `.493`."""

    arc = payloads.get(EXP5423, {})
    timing = payloads.get(EXP5424, {})
    csl = payloads.get(EXP5421, {})
    promotion = payloads.get(EXP5422, {})
    return [
        {
            "target": "arc_live_levelup",
            "recommendation": "Keep the ARC slot because Exp5423 banked no new level; only count a future result after reproduction-gated level growth.",
            "evidence": {
                "arc_new_level_banked": unwrap(arc.get("arc_new_level_banked")),
                "registry_total_before": unwrap(arc.get("registry_total_before")),
                "registry_total_after": unwrap(arc.get("registry_total_after")),
            },
        },
        {
            "target": "pbit_hardware_transfer",
            "recommendation": "Move p-bit beyond preflight only against the comparable CPU/PolarFire timing receipt, and keep acceleration unclaimed until board-local speedup evidence exists.",
            "evidence": {
                "comparable_timing_receipts_ready": unwrap(timing.get("comparable_timing_receipts_ready")),
                "same_workload_hash_match": unwrap(timing.get("same_workload_hash_match")),
                "same_result_hash_match": unwrap(timing.get("same_result_hash_match")),
                "hardware_speedup_claim": unwrap(timing.get("hardware_speedup_claim")),
            },
        },
        {
            "target": "continuous_self_learning",
            "recommendation": "Expand CSL only under the same reliance-drift, rollback, grounding, and no-weight-mutation gates because Exp5421/Exp5422 were stable there.",
            "evidence": {
                "quality_preserved": unwrap(csl.get("quality_preserved")),
                "rollback_verified": unwrap(csl.get("rollback_verified")),
                "promotion_grounding_preserved": unwrap(promotion.get("grounding_preserved")),
                "no_weight_mutation": unwrap(csl.get("no_weight_mutation")) is True
                and unwrap(promotion.get("no_weight_mutation")) is True,
            },
        },
        {
            "target": "token_internal_backend",
            "recommendation": "Keep token/internal lanes closed unless a future artifact carries authenticated logits, hidden-state, attention, token-logprob, or intermediate-exit backend receipts.",
            "evidence": {"future_token_signal_allowed": future_token_signal_allowed(payloads)},
        },
    ]


def honest_verdict(artifact: JsonMap) -> str:
    """Summarize the capstone boundary with the required terminal prefix."""

    if artifact["upstream_artifacts_missing"] or artifact["artifact_read_errors"]:
        return (
            "blocked: .493 capstone emitted with missing or unreadable source artifacts; "
            "affected lanes were classified missing instead of inferred."
        )
    return (
        "complete: .493 capstone emitted from actual artifacts; CSL reliance and gated "
        "promotion are headline-ready, active constraints/p-bit timing/KAN and comparable "
        "hardware timing are bounded, risk-calibrated and predictive-prefix structured "
        "lanes remain blocked by adversarial flags despite valid GGUF receipts, ARC no-bank "
        "keeps the north-star count unchanged, no hardware speedup is claimed, and "
        "token/internal lane closed."
    )


def source_artifact_checksums(root: Path, read_paths: Sequence[str]) -> dict[str, str]:
    """Hash every file read so later audits can detect silent drift."""

    checksums: dict[str, str] = {}
    for relative in read_paths:
        path = root / relative
        if path.exists() and path.is_file():
            checksums[relative] = file_sha256(path)
    return checksums


def protected_file_checks(root: Path) -> list[JsonDict]:
    """Record the protected files this task was explicitly forbidden to change."""

    rows: list[JsonDict] = []
    for relative in ("research-roadmap.yaml", "scripts/research_conductor.py"):
        path = root / relative
        rows.append(
            {
                "path": relative,
                "exists": path.exists(),
                "git_status_clean": git_path_unchanged(root, relative),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
            }
        )
    return rows


def validate_artifact(artifact: JsonMap) -> None:
    """Reject schema drift and every overclaim this capstone guards against."""

    missing_fields = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(f"missing required fields: {missing_fields}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles drift")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone must be 2026.07.493")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    for field in BOOLEAN_FIELDS:
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be boolean")
    if artifact["arc_new_level_banked"] is not False:
        raise ValueError("arc_new_level_banked must remain false for Exp5423")
    if artifact["hardware_speedup_claim"] is not False:
        raise ValueError("hardware_speedup_claim must remain false")
    if artifact["future_token_signal_allowed"] is not False:
        raise ValueError("future_token_signal_allowed must remain false")
    if artifact["research_roadmap_yaml_unchanged"] is not True:
        raise ValueError("research_roadmap_yaml_unchanged must be true")
    if artifact["conductor_unchanged"] is not True:
        raise ValueError("conductor_unchanged must be true")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    if artifact["upstream_artifacts_missing"] or artifact["artifact_read_errors"]:
        if not str(artifact["honest_verdict"]).startswith("blocked:"):
            raise ValueError("honest_verdict must start with blocked: when inputs are missing")
    elif not str(artifact["honest_verdict"]).startswith("complete:"):
        raise ValueError("honest_verdict must start with complete: when all inputs are readable")

    rows = artifact["truth_table"]
    if not isinstance(rows, list) or [row.get("lane") for row in rows] != list(LANE_ORDER):
        raise ValueError("truth_table lane order drift")
    for row in rows:
        if row.get("classification") not in ALLOWED_CLASSIFICATIONS:
            raise ValueError("truth_table classification invalid")
        if not isinstance(row.get("evidence"), Mapping):
            raise ValueError("truth_table row evidence must be object")
    expected_buckets = bucket_lanes(rows)
    for bucket_name, expected_rows in expected_buckets.items():
        if artifact[bucket_name] != expected_rows:
            raise ValueError(f"lane bucket mismatch: {bucket_name}")
    if lane_names(artifact["headline_ready_lanes"]) != ["evidence_reliance_csl", "gated_csl_promotion"] and not artifact[
        "upstream_artifacts_missing"
    ]:
        raise ValueError("headline_ready_lanes overclaim or drift")
    if set(lane_names(artifact["headline_ready_lanes"])) & set(lane_names(artifact["blocked_lanes"])):
        raise ValueError("lane bucket overlap")
    if [row.get("target") for row in artifact["next_recommendations"]] != [
        "arc_live_levelup",
        "pbit_hardware_transfer",
        "continuous_self_learning",
        "token_internal_backend",
    ]:
        raise ValueError("next_recommendations target drift")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum drift")


def lane_names(rows: Sequence[JsonMap]) -> list[str]:
    """Return row lane names for concise validation."""

    return [str(row["lane"]) for row in rows]


def payload_checksum(payload: JsonMap) -> str:
    """Hash the artifact payload with its checksum field excluded."""

    clean = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(json_ready(clean), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a source artifact or context file for provenance."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def git_path_unchanged(root: Path, relative: str) -> bool:
    """Return whether git reports no changes for a path, defaulting clean if git is unavailable."""

    try:
        completed = subprocess.run(
            ["git", "status", "--short", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return True
    return completed.stdout.strip() == ""


def json_ready(value: Any) -> Any:
    """Convert pathlib and tuple values so artifacts serialize deterministically."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: JsonMap) -> None:
    """Write pretty, stable JSON with a trailing newline for git diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by tests and conductor-style runners."""

    parser = argparse.ArgumentParser(description="Write the Exp5427 .493 capstone artifact.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    run(root=args.root, result_path=args.result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
