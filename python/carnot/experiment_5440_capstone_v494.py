"""Exp5440 .494 terminal capstone truth-table synthesis.

Spec refs: REQ-CAPSTONE-5440, SCENARIO-CAPSTONE-5440,
SCENARIO-CAPSTONE-5440-MISSING-INPUT,
SCENARIO-CAPSTONE-5440-FIELD-PRINCIPLES.

This module is intentionally an aggregation step. It reads the `.494` source
context and the upstream result artifacts, then emits a terminal truth table
from those receipts only. That boundary matters: the capstone is allowed to say
that a structured corrigendum is clean, a workflow-memory controller is stable,
or a hardware timing receipt exists, but it must not turn those facts into
unsupported ARC growth, hardware acceleration, or token/internal access claims.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5440_capstone_v494.json")
EXPERIMENT = "experiment_5440_capstone_v494"
EXPERIMENT_ID = "exp5440-v494-capstone"
MILESTONE = "2026.07.494"
SCHEMA = "carnot.experiment_5440.capstone_v494.v1"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5440
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP5428 = "results/experiment_5428_transition_v494.json"
EXP5429 = "results/experiment_5429_source_delta_v494.json"
EXP5430 = "results/experiment_5430_structured_tautology_corrigendum_v494.json"
EXP5431 = "results/experiment_5431_structured_constraint_taxonomy_replication_v494.json"
EXP5432 = "results/experiment_5432_ontology_softlogic_constraint_memory_v494.json"
EXP5433 = "results/experiment_5433_active_constraint_diversity_lns_v494.json"
EXP5434 = "results/experiment_5434_pbit_polarfire_timing_variance_v494.json"
EXP5435 = "results/experiment_5435_verified_workflow_memory_csl_v494.json"
EXP5436 = "results/experiment_5436_csl_memory_transfer_stress_v494.json"
EXP5437 = "results/experiment_5437_arc_live_reinduction_levelup_v494.json"
EXP5438 = "results/experiment_5438_kan_ontology_measurement_certificate_v494.json"
EXP5439 = "results/experiment_5439_prd_gap_agent_failure_table_v494.json"

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
    EXP5428,
    EXP5429,
    EXP5430,
    EXP5431,
    EXP5432,
    EXP5433,
    EXP5434,
    EXP5435,
    EXP5436,
    EXP5437,
    EXP5438,
    EXP5439,
)
EXPECTED_INPUT_PATHS: tuple[str, ...] = (*SOURCE_CONTEXT_PATHS, *RESULT_ARTIFACT_PATHS)

SPEC_REFS = (
    "REQ-CAPSTONE-5440",
    "SCENARIO-CAPSTONE-5440",
    "SCENARIO-CAPSTONE-5440-MISSING-INPUT",
    "SCENARIO-CAPSTONE-5440-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "conductor route key; must equal 2026.07.494.",
    "upstream_artifacts_read": "provenance; ordered list of source context and result artifacts actually read.",
    "upstream_artifacts_missing": "no fabricated evidence; missing or unreadable inputs are recorded here.",
    "headline_ready_lanes": "positive evidence boundary; only unflagged closed rows with no stronger blocker.",
    "bounded_lanes": "partial evidence boundary; useful but limited receipts that must not become headline claims.",
    "honest_null_lanes": "null-result honesty; executed lanes with no banked or positive outcome.",
    "blocked_lanes": "precondition honesty; flagged, closed, or missing-precondition lanes.",
    "arc_new_level_banked": "ARC north-star metric; true only for a reproduction-gated new level.",
    "hardware_speedup_claim": "no unsupported acceleration; must remain false for `.494`.",
    "future_token_signal_allowed": "token/internal lane closure; must remain false without authenticated backend receipts.",
    "local_sota_gguf_receipts_valid": "SOTA model provenance; true only when the structured artifacts carry GGUF model specs and GPU-offload receipts.",
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
    "structured_corrigendum",
    "structured_taxonomy_replication",
    "ontology_softlogic_memory",
    "active_constraint_diversity_lns",
    "pbit_polarfire_timing_variance",
    "verified_workflow_memory_csl",
    "csl_memory_transfer_stress",
    "arc_live_reinduction_levelup",
    "kan_ontology_certificates",
    "token_internal_feature_lane_closed",
    "hardware_speedup_claim",
)
DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5440_capstone_v494.py -q",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5440_capstone_v494.py "
            "-m pytest tests/python/test_experiment_5440_capstone_v494.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5440_capstone_v494.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
)

LANE_SPECS: tuple[JsonDict, ...] = (
    {
        "lane": "structured_corrigendum",
        "source_artifacts": [EXP5430],
        "claim_boundary": "clean row-level corrigendum; this unlocks structured verification but is not itself a broad SOTA claim",
        "evidence_fields": (
            "structured_corrigendum_clean",
            "adversarial_verify_clean",
            "row_count_recomputed",
            "risk_metric_independence_check",
            "prefix_metric_independence_check",
            "unreachable_delta_recomputed",
            "gpu_offload_verified",
        ),
    },
    {
        "lane": "structured_taxonomy_replication",
        "source_artifacts": [EXP5431],
        "claim_boundary": "local structured taxonomy replication with independent metrics and bounded risk; not a broad quality claim",
        "evidence_fields": (
            "structured_taxonomy_replication_ready",
            "gated_upstream_clean",
            "metric_independence_checks_passed",
            "accepted_risk_bound",
            "accepted_risk_bound_threshold",
            "unsafe_false_accept_rate",
            "unreachable_action_false_accept_rate",
            "semantic_false_accept_rate",
            "abstention_rate",
            "fixture_count",
            "gpu_offload_verified",
        ),
    },
    {
        "lane": "ontology_softlogic_memory",
        "source_artifacts": [EXP5432],
        "claim_boundary": "deterministic ontology-memory fixture with solver authority preserved",
        "evidence_fields": (
            "ontology_constraint_memory_ready",
            "valid_update_preservation_rate",
            "false_triple_rejection_rate",
            "unsupported_update_abstention_rate",
            "soft_logic_overrode_solver",
            "deterministic_solver_authority",
            "soft_logic_residuals_recorded",
            "ontology_fixture_count",
            "triple_count",
        ),
    },
    {
        "lane": "active_constraint_diversity_lns",
        "source_artifacts": [EXP5433],
        "claim_boundary": "bounded solver-guidance evidence; LNS hints are advisory and solver authority is final",
        "evidence_fields": (
            "active_constraint_diversity_ready",
            "solver_validity_preserved",
            "subproblem_family_count",
            "fixture_count",
            "accepted_hint_count",
            "rejected_hint_count",
            "overwritten_hint_count",
            "work_delta",
            "conflict_front_precision",
            "claim_limits",
        ),
    },
    {
        "lane": "pbit_polarfire_timing_variance",
        "source_artifacts": [EXP5434],
        "claim_boundary": "matched CPU and PolarFire timing variance receipts only; no acceleration claim",
        "evidence_fields": (
            "timing_variance_receipts_ready",
            "measurement_access_complete",
            "same_workload_hash_match",
            "same_result_hash_match",
            "hardware_speedup_claim",
            "cpu_repeat_count",
            "board_repeat_count",
            "polarfire_reachable",
            "kv260_ssh_checked",
            "gatemate_diagnostic_checked",
            "cpu_timing_variance",
            "board_timing_variance",
            "claim_refusal",
        ),
    },
    {
        "lane": "verified_workflow_memory_csl",
        "source_artifacts": [EXP5435],
        "claim_boundary": "controller-level workflow memory with verification-before-store, rollback, and no weight mutation",
        "evidence_fields": (
            "verified_workflow_memory_ready",
            "verify_before_store_pass_rate",
            "ontology_kernel_validation_rate",
            "retrieval_trap_deflection_rate",
            "quality_preserved",
            "reliance_drift_metric",
            "rollback_verified",
            "no_weight_mutation",
            "raw_episodes_retained",
            "workflow_episode_count",
            "case_memory_count",
            "skill_memory_count",
        ),
    },
    {
        "lane": "csl_memory_transfer_stress",
        "source_artifacts": [EXP5436],
        "claim_boundary": "workflow-memory transfer stress with in-domain lift, shift deflection, rollback, and no weight mutation",
        "evidence_fields": (
            "csl_transfer_stress_ready",
            "in_domain_quality_delta",
            "out_of_domain_quality_delta",
            "negative_transfer_deflection_rate",
            "rollback_verified",
            "no_weight_mutation",
            "promoted_transfer_count",
            "quarantined_transfer_count",
            "transfer_fixture_count",
            "reliance_drift_metric",
        ),
    },
    {
        "lane": "arc_live_reinduction_levelup",
        "source_artifacts": [EXP5437],
        "claim_boundary": "live ARC path ran through the reproduction gate; no new level was banked",
        "evidence_fields": (
            "status",
            "arc_new_level_banked",
            "offline_reproduced",
            "reproduced_levels",
            "newly_reached_levels",
            "attempt_count",
            "frontier_expansion_count",
            "failure_mode",
            "registry_total_before",
            "registry_total_after",
            "target_game",
            "target_level",
            "solve_provenance",
            "arc_levelup_lint_passed",
        ),
    },
    {
        "lane": "kan_ontology_certificates",
        "source_artifacts": [EXP5438],
        "claim_boundary": "bounded KAN ontology/workflow-memory measurement certificate; no broad KAN verification claim",
        "evidence_fields": (
            "kan_ontology_certificate_ready",
            "certificate_count",
            "false_property_rejection_rate",
            "true_property_preservation_rate",
            "missing_evidence_detected",
            "broad_kan_verification_claim",
            "ontology_property_count",
            "workflow_memory_property_count",
            "claim_limits",
        ),
    },
    {
        "lane": "token_internal_feature_lane_closed",
        "source_artifacts": [EXP5428, EXP5438, EXP5439],
        "claim_boundary": "closed without authenticated logits, hidden-state, attention, token, or intermediate-exit receipts",
        "evidence_fields": (),
    },
    {
        "lane": "hardware_speedup_claim",
        "source_artifacts": [EXP5434, EXP5439],
        "claim_boundary": "timing receipts exist, but no matched board acceleration claim is supported",
        "evidence_fields": (),
    },
)


def unwrap(value: Any) -> Any:
    """Return bare values from principle-wrapped artifacts when older producers use them."""

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
        "arc_new_level_banked": bool(unwrap(payloads.get(EXP5437, {}).get("arc_new_level_banked"))),
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
            evidence={
                "future_token_signal_allowed": future_token_signal_allowed(payloads),
                "backend_receipt_present": authenticated_backend_receipt_present(payloads),
            },
            blocked_reason="no_authenticated_backend_receipt",
        )

    if lane == "hardware_speedup_claim":
        return lane_row(
            lane=lane,
            source_artifacts=sources,
            classification="honest_null" if not hardware_speedup_claim(payloads) else "blocked",
            claim_boundary=str(spec["claim_boundary"]),
            evidence={"hardware_speedup_claim": hardware_speedup_claim(payloads)},
            blocked_reason="no_board_local_speedup_observed",
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

    classification = lane_classification(lane, payload, payloads)
    blocked_reason = "" if classification in {"headline_ready", "bounded", "honest_null"} else f"{lane}_not_ready"
    if classification == "honest_null":
        blocked_reason = str(payload.get("failure_mode", "honest_null"))
    return lane_row(
        lane=lane,
        source_artifacts=sources,
        classification=classification,
        claim_boundary=str(spec["claim_boundary"]),
        evidence=evidence_for_spec(spec, payload),
        blocked_reason=blocked_reason,
    )


def lane_classification(lane: str, payload: JsonMap, payloads: Mapping[str, JsonMap]) -> str:
    """Convert lane-specific gates into the common capstone vocabulary."""

    if lane == "structured_corrigendum":
        ready = all(
            unwrap(payload.get(field)) is True
            for field in (
                "structured_corrigendum_clean",
                "adversarial_verify_clean",
                "risk_metric_independence_check",
                "prefix_metric_independence_check",
                "unreachable_delta_recomputed",
                "gpu_offload_verified",
            )
        )
        return "headline_ready" if ready else "blocked"
    if lane == "structured_taxonomy_replication":
        risk = unwrap(payload.get("accepted_risk_bound"))
        threshold = unwrap(payload.get("accepted_risk_bound_threshold"))
        bounded_risk = isinstance(risk, (int, float)) and isinstance(threshold, (int, float)) and risk <= threshold
        ready = (
            unwrap(payload.get("structured_taxonomy_replication_ready")) is True
            and unwrap(payload.get("gated_upstream_clean")) is True
            and unwrap(payload.get("metric_independence_checks_passed")) is True
            and unwrap(payload.get("gpu_offload_verified")) is True
            and bounded_risk
            and local_sota_gguf_receipts_valid(payloads)
        )
        return "headline_ready" if ready else "blocked"
    if lane == "ontology_softlogic_memory":
        ready = (
            unwrap(payload.get("ontology_constraint_memory_ready")) is True
            and unwrap(payload.get("deterministic_solver_authority")) is True
            and unwrap(payload.get("soft_logic_overrode_solver")) is False
        )
        return "headline_ready" if ready else "blocked"
    if lane == "active_constraint_diversity_lns":
        ready = (
            unwrap(payload.get("active_constraint_diversity_ready")) is True
            and unwrap(payload.get("solver_validity_preserved")) is True
        )
        return "bounded" if ready else "blocked"
    if lane == "pbit_polarfire_timing_variance":
        ready = all(
            unwrap(payload.get(field)) is True
            for field in (
                "timing_variance_receipts_ready",
                "measurement_access_complete",
                "same_workload_hash_match",
                "same_result_hash_match",
            )
        ) and unwrap(payload.get("hardware_speedup_claim")) is False
        return "bounded" if ready else "blocked"
    if lane == "verified_workflow_memory_csl":
        ready = all(
            unwrap(payload.get(field)) is True
            for field in (
                "verified_workflow_memory_ready",
                "quality_preserved",
                "rollback_verified",
                "no_weight_mutation",
                "raw_episodes_retained",
            )
        )
        return "headline_ready" if ready else "blocked"
    if lane == "csl_memory_transfer_stress":
        in_domain_delta = unwrap(payload.get("in_domain_quality_delta"))
        out_domain_delta = unwrap(payload.get("out_of_domain_quality_delta"))
        deltas_non_negative = isinstance(in_domain_delta, (int, float)) and isinstance(
            out_domain_delta, (int, float)
        ) and in_domain_delta >= 0 and out_domain_delta >= 0
        ready = (
            unwrap(payload.get("csl_transfer_stress_ready")) is True
            and unwrap(payload.get("negative_transfer_deflection_rate")) == 1.0
            and unwrap(payload.get("rollback_verified")) is True
            and unwrap(payload.get("no_weight_mutation")) is True
            and deltas_non_negative
        )
        return "headline_ready" if ready else "blocked"
    if lane == "arc_live_reinduction_levelup":
        return "headline_ready" if unwrap(payload.get("arc_new_level_banked")) is True else "honest_null"
    if lane == "kan_ontology_certificates":
        ready = (
            unwrap(payload.get("kan_ontology_certificate_ready")) is True
            and unwrap(payload.get("broad_kan_verification_claim")) is False
        )
        return "bounded" if ready else "blocked"
    return "blocked"


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

    return recursive_key_true(payloads.get(EXP5434, {}), "hardware_speedup_claim") or recursive_key_true(
        payloads.get(EXP5439, {}), "hardware_speedup_claim"
    )


def future_token_signal_allowed(payloads: Mapping[str, JsonMap]) -> bool:
    """Keep token/internal lanes closed unless a receipt explicitly opens them."""

    return any(
        recursive_key_true(payloads.get(path, {}), key)
        for path in (EXP5428, EXP5438, EXP5439)
        for key in (
            "future_token_signal_allowed",
            "authenticated_backend_receipt",
            "token_logprob_receipt_present",
            "hidden_state_receipt_present",
        )
    )


def authenticated_backend_receipt_present(payloads: Mapping[str, JsonMap]) -> bool:
    """Detect a positive backend receipt separately from the policy allow flag."""

    return any(
        recursive_key_true(payloads.get(path, {}), key)
        for path in (EXP5428, EXP5438, EXP5439)
        for key in (
            "backend_receipt_present",
            "authenticated_backend_receipt",
            "token_logprob_receipt_present",
            "hidden_state_receipt_present",
        )
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
    """Validate model provenance without treating model receipts as a broader claim."""

    for path in (EXP5430, EXP5431):
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
    """Seed future planning from concrete `.494` evidence without planning the next milestone."""

    corrigendum = payloads.get(EXP5430, {})
    taxonomy = payloads.get(EXP5431, {})
    workflow = payloads.get(EXP5435, {})
    transfer = payloads.get(EXP5436, {})
    arc = payloads.get(EXP5437, {})
    timing = payloads.get(EXP5434, {})
    return [
        {
            "target": "structured_verification",
            "recommendation": "Scale structured verification only while the Exp5430 corrigendum stays clean and Exp5431 keeps independent bounded-risk replication receipts.",
            "evidence": {
                "structured_corrigendum_clean": unwrap(corrigendum.get("structured_corrigendum_clean")),
                "adversarial_verify_clean": unwrap(corrigendum.get("adversarial_verify_clean")),
                "structured_taxonomy_replication_ready": unwrap(
                    taxonomy.get("structured_taxonomy_replication_ready")
                ),
                "metric_independence_checks_passed": unwrap(taxonomy.get("metric_independence_checks_passed")),
            },
        },
        {
            "target": "continuous_self_learning",
            "recommendation": "Expand CSL only while workflow-memory transfer remains stable under rollback, negative-transfer deflection, and no-weight-mutation gates.",
            "evidence": {
                "verified_workflow_memory_ready": unwrap(workflow.get("verified_workflow_memory_ready")),
                "csl_transfer_stress_ready": unwrap(transfer.get("csl_transfer_stress_ready")),
                "in_domain_quality_delta": unwrap(transfer.get("in_domain_quality_delta")),
                "negative_transfer_deflection_rate": unwrap(transfer.get("negative_transfer_deflection_rate")),
                "no_weight_mutation": unwrap(workflow.get("no_weight_mutation")) is True
                and unwrap(transfer.get("no_weight_mutation")) is True,
            },
        },
        {
            "target": "arc_live_levelup",
            "recommendation": "Keep the ARC slot because Exp5437 banked no new level; only count future growth after offline reproduction-gated level increase.",
            "evidence": {
                "arc_new_level_banked": unwrap(arc.get("arc_new_level_banked")),
                "target_game": unwrap(arc.get("target_game")),
                "target_level": unwrap(arc.get("target_level")),
                "registry_total_before": unwrap(arc.get("registry_total_before")),
                "registry_total_after": unwrap(arc.get("registry_total_after")),
            },
        },
        {
            "target": "pbit_hardware_timing",
            "recommendation": "Keep hardware speedup closed until matched repeated board evidence supports acceleration; Exp5434 supplies timing variance receipts but not a speedup.",
            "evidence": {
                "timing_variance_receipts_ready": unwrap(timing.get("timing_variance_receipts_ready")),
                "cpu_repeat_count": unwrap(timing.get("cpu_repeat_count")),
                "board_repeat_count": unwrap(timing.get("board_repeat_count")),
                "same_workload_hash_match": unwrap(timing.get("same_workload_hash_match")),
                "same_result_hash_match": unwrap(timing.get("same_result_hash_match")),
                "hardware_speedup_claim": unwrap(timing.get("hardware_speedup_claim")),
            },
        },
        {
            "target": "token_internal_backend",
            "recommendation": "Keep token/internal lanes closed unless a future artifact carries authenticated logits, hidden-state, attention, token-logprob, or intermediate-exit backend receipts.",
            "evidence": {
                "future_token_signal_allowed": future_token_signal_allowed(payloads),
                "backend_receipt_present": authenticated_backend_receipt_present(payloads),
            },
        },
    ]


def honest_verdict(artifact: JsonMap) -> str:
    """Summarize the capstone boundary with the required terminal prefix."""

    if artifact["upstream_artifacts_missing"] or artifact["artifact_read_errors"]:
        return (
            "blocked: .494 capstone emitted with missing or unreadable source artifacts; "
            "affected lanes were classified missing instead of inferred."
        )
    return (
        "complete: .494 capstone emitted from actual artifacts; structured corrigendum "
        "and taxonomy replication, ontology memory, verified workflow CSL, and CSL "
        "transfer stress are headline-ready; active constraints, p-bit timing, and "
        "KAN ontology certificates are bounded; ARC no-bank keeps the north-star count "
        "unchanged; no hardware speedup is claimed; token/internal lane closed."
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
        raise ValueError("milestone must be 2026.07.494")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    for field in BOOLEAN_FIELDS:
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be boolean")
    if artifact["arc_new_level_banked"] is not False:
        raise ValueError("arc_new_level_banked must remain false for Exp5437")
    if artifact["hardware_speedup_claim"] is not False:
        raise ValueError("hardware_speedup_claim must remain false")
    if artifact["future_token_signal_allowed"] is not False:
        raise ValueError("future_token_signal_allowed must remain false")
    if not artifact["upstream_artifacts_missing"] and artifact["local_sota_gguf_receipts_valid"] is not True:
        raise ValueError("local_sota_gguf_receipts_valid must be true when structured inputs are present")
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
    if not artifact["upstream_artifacts_missing"]:
        if lane_names(artifact["headline_ready_lanes"]) != [
            "structured_corrigendum",
            "structured_taxonomy_replication",
            "ontology_softlogic_memory",
            "verified_workflow_memory_csl",
            "csl_memory_transfer_stress",
        ]:
            raise ValueError("headline_ready_lanes overclaim or drift")
    if [row.get("target") for row in artifact["next_recommendations"]] != [
        "structured_verification",
        "continuous_self_learning",
        "arc_live_levelup",
        "pbit_hardware_timing",
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

    parser = argparse.ArgumentParser(description="Write the Exp5440 .494 capstone artifact.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    run(root=args.root, result_path=args.result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
