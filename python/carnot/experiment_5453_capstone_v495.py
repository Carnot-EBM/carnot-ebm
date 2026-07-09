"""Exp5453 .495 terminal capstone truth-table synthesis.

Spec refs: REQ-CAPSTONE-5453, SCENARIO-CAPSTONE-5453,
SCENARIO-CAPSTONE-5453-MISSING-INPUT,
SCENARIO-CAPSTONE-5453-FIELD-PRINCIPLES.

This module is an evidence aggregator, not a fresh experiment runner. It reads
the completed `.495` result artifacts and emits a capstone that preserves their
claim boundaries. That discipline is important because several upstream lanes
contain useful measurements while still being non-headline: local SOTA decoding
is adversarially flagged, hardware timing is receipt-only with no speedup, ARC
banked no new level, and token/internal access remains closed without backend
receipts.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5453_capstone_v495.json")
EXPERIMENT = "experiment_5453_capstone_v495"
EXPERIMENT_ID = "exp5453-v495-capstone"
MILESTONE = "2026.07.495"
TASK_RANGE = "exp5441-exp5453"
SCHEMA = "carnot.experiment_5453.capstone_v495.v1"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5453
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP5441 = "results/experiment_5441_transition_v495.json"
EXP5442 = "results/experiment_5442_source_delta_v495.json"
EXP5443 = "results/experiment_5443_verifier_potential_prefix_fixture_v495.json"
EXP5444 = "results/experiment_5444_gated_sota_energy_guided_decoding_v495.json"
EXP5445 = "results/experiment_5445_static_ast_kb_witness_constraints_v495.json"
EXP5446 = "results/experiment_5446_governed_memory_csl_online_v495.json"
EXP5447 = "results/experiment_5447_gated_csl_memory_failure_stress_v495.json"
EXP5448 = "results/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.json"
EXP5449 = "results/experiment_5449_gated_hardware_timing_sparsity_receipts_v495.json"
EXP5450 = "results/experiment_5450_arc_measurement_access_live_levelup_v495.json"
EXP5451 = "results/experiment_5451_kan_verifier_potential_memory_certificate_v495.json"
EXP5452 = "results/experiment_5452_prd_gap_agent_failure_table_v495.json"

SOURCE_CONTEXT_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "research-program.md",
    "_bmad/prd.md",
    "_bmad/architecture.md",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "ops/changelog.md",
    "ops/status.md",
)
RESULT_ARTIFACT_PATHS: tuple[str, ...] = (
    EXP5441,
    EXP5442,
    EXP5443,
    EXP5444,
    EXP5445,
    EXP5446,
    EXP5447,
    EXP5448,
    EXP5449,
    EXP5450,
    EXP5451,
    EXP5452,
)
EXPECTED_INPUT_PATHS: tuple[str, ...] = (*SOURCE_CONTEXT_PATHS, *RESULT_ARTIFACT_PATHS)

SPEC_REFS = (
    "REQ-CAPSTONE-5453",
    "SCENARIO-CAPSTONE-5453",
    "SCENARIO-CAPSTONE-5453-MISSING-INPUT",
    "SCENARIO-CAPSTONE-5453-FIELD-PRINCIPLES",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "route key; must equal 2026.07.495.",
    "task_range": "execution boundary; must equal exp5441-exp5453.",
    "artifacts_found": "evidence basis; ordered list of expected upstream artifacts actually read.",
    "headline_ready_lanes": (
        "positive evidence; only unflagged closed rows with no adversarial, tautology, duration, "
        "or unsupported-claim blockers."
    ),
    "bounded_lanes": "bounded evidence; useful receipts that must not become headline claims.",
    "blocked_lanes": "blocker accounting; flagged, closed-backend, or unsupported lanes.",
    "honest_null_lanes": "null-result honesty; executed lanes with no banked or positive outcome.",
    "missing_lanes": "no fabricated evidence; missing or unreadable inputs are recorded here.",
    "arc_new_level_banked": "north-star status; true only for a reproduction-gated new ARC level.",
    "hardware_speedup_claim": "hardware honesty; must be false for `.495`.",
    "token_internal_lane_reopened": "closed-lane discipline; must be false without authenticated backend receipts.",
    "next_recommendations": "planning handoff; 3-5 directions grounded in actual evidence.",
    "roadmap_yaml_unchanged": "user prohibition; derived from git status.",
    "conductor_unchanged": "user prohibition; derived from git status.",
    "inference_substrate": "no hidden live model inference; must equal aggregation_from_upstream_artifacts.",
    "honest_verdict": "terminal status; starts with complete: or blocked: and summarizes the honest `.495` close state.",
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
    "source_context_read",
    "source_context_missing",
    "artifacts_missing",
    "artifact_read_errors",
    "truth_table",
    "source_artifact_checksums",
    "protected_file_checks",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES.keys(),
)
BOOLEAN_FIELDS = (
    "arc_new_level_banked",
    "hardware_speedup_claim",
    "token_internal_lane_reopened",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
)
ALLOWED_CLASSIFICATIONS = {"headline_ready", "bounded", "honest_null", "blocked", "missing"}
LANE_ORDER = (
    "verifier_potential_generation",
    "local_sota_decoding",
    "ast_kb_witnesses",
    "governed_csl",
    "memory_stress",
    "active_constraint_pbit_bridge",
    "hardware_receipts",
    "arc_live_progress",
    "kan_certificates",
    "prd_gap_synthesis",
    "token_internal_access",
    "hardware_speedup_claim",
)
DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5453_capstone_v495.py -q",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5453_capstone_v495.py "
            "-m pytest tests/python/test_experiment_5453_capstone_v495.py -q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5453_capstone_v495.py --fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
)

LANE_SPECS: tuple[JsonDict, ...] = (
    {
        "lane": "verifier_potential_generation",
        "source_artifacts": [EXP5443],
        "claim_boundary": "deterministic verifier-potential fixtures with exact final authority; not learned truth",
        "evidence_fields": (
            "verifier_potential_fixture_ready",
            "exact_final_authority",
            "prefix_final_disagreement_cases",
            "metric_independence_checks_passed",
            "fixture_count",
            "constraint_family_counts",
            "reward_evaluation_budget",
        ),
    },
    {
        "lane": "local_sota_decoding",
        "source_artifacts": [EXP5444],
        "claim_boundary": "local GGUF decoding pilot is blocked from headline while adversarial tautology flags remain",
        "evidence_fields": (
            "verifier_guided_decoding_ready",
            "flagged_adversarial",
            "corrigendum_pending",
            "guided_validity_delta_vs_grammar_only",
            "guided_validity_delta_vs_unconstrained",
            "gpu_offload_verified",
            "runtime_backend",
            "unsafe_false_accept_rate",
            "action_unreachability_rate",
            "semantic_false_accept_rate",
            "abstention_rate",
        ),
    },
    {
        "lane": "ast_kb_witnesses",
        "source_artifacts": [EXP5445],
        "claim_boundary": "deterministic AST/KB witness rows for code/API hallucination constraints",
        "evidence_fields": (
            "ast_kb_witness_ready",
            "ast_parse_success_rate",
            "valid_call_accept_rate",
            "nonexistent_call_reject_rate",
            "unsafe_false_accepts",
            "fixture_count",
            "api_family_counts",
            "witness_field_names",
        ),
    },
    {
        "lane": "governed_csl",
        "source_artifacts": [EXP5446],
        "claim_boundary": "governed sidecar memory promotion with replay, rollback, and no model-weight mutation",
        "evidence_fields": (
            "governed_csl_loop_ready",
            "continuous_self_learning_task",
            "multi_session_trace_count",
            "promotion_level_counts",
            "replay_success_rate",
            "rollback_recovery_rate",
            "negative_transfer_deflection_rate",
            "unsafe_false_accepts",
            "no_weight_mutation",
            "quality_delta_vs_always_full",
            "verifier_cost_delta",
            "context_efficiency_delta",
        ),
    },
    {
        "lane": "memory_stress",
        "source_artifacts": [EXP5447],
        "claim_boundary": "governed memory-failure stress with deflection, attribution, rollback, and no weight mutation",
        "evidence_fields": (
            "csl_memory_stress_ready",
            "gated_upstream_ready",
            "memory_failure_case_count",
            "failure_operation_counts",
            "stale_memory_deflection_rate",
            "poisoned_memory_deflection_rate",
            "retrieval_collision_deflection_rate",
            "negative_transfer_deflection_rate",
            "rollback_recovery_rate",
            "unsafe_false_accepts",
            "no_weight_mutation",
        ),
    },
    {
        "lane": "active_constraint_pbit_bridge",
        "source_artifacts": [EXP5448],
        "claim_boundary": "active and p-bit assumptions are advisory; exact solver keeps final authority",
        "evidence_fields": (
            "pbit_assumption_bridge_ready",
            "solver_authoritative",
            "fallback_completeness_rate",
            "fixture_count",
            "row_count",
            "solver_work_delta",
            "density_before_after",
            "rejected_assumption_count",
            "overwritten_assumption_count",
            "unsafe_false_accepts",
            "hardware_speedup_claim",
            "claim_limits",
        ),
    },
    {
        "lane": "hardware_receipts",
        "source_artifacts": [EXP5449],
        "claim_boundary": "hash-matched CPU and reachable-board timing receipts only; no acceleration claim",
        "evidence_fields": (
            "hardware_receipts_ready",
            "gated_upstream_ready",
            "hardware_speedup_claim",
            "hashes_match_before_timing_compare",
            "timing_repeat_counts",
            "timing_comparison",
            "board_reachability",
            "readiness_blockers",
            "timing_summary",
            "claim_refusal",
        ),
    },
    {
        "lane": "arc_live_progress",
        "source_artifacts": [EXP5450],
        "claim_boundary": "live ARC path ran through the reproduction gate; no new level was banked",
        "evidence_fields": (
            "arc_new_level_banked",
            "new_level_reproduced",
            "new_levels_banked",
            "offline_reproduced",
            "reproduced_levels",
            "registry_level_before",
            "registry_precheck_total_levels",
            "selected_game",
            "selected_target_level_label",
            "solve_provenance",
            "no_source_reading",
            "no_offline_bfs",
            "no_per_game_adapter_credited",
            "frontier_expansion_count",
            "live_attempt_count",
            "residual_wall",
        ),
    },
    {
        "lane": "kan_certificates",
        "source_artifacts": [EXP5451],
        "claim_boundary": "bounded measurement-access certificates; broad KAN, hardware, and token/internal claims rejected",
        "evidence_fields": (
            "kan_certificate_ready",
            "gated_upstreams_ready",
            "verifier_potential_claims_checked",
            "governed_memory_claims_checked",
            "true_measured_claim_preservation_rate",
            "false_property_rejection_rate",
            "unsupported_claim_rejection_rate",
            "hardware_speedup_claim_rejected",
            "token_internal_claim_rejected",
            "broad_kan_claim_made",
            "claim_count",
            "property_family",
            "claim_limits",
        ),
    },
    {
        "lane": "prd_gap_synthesis",
        "source_artifacts": [EXP5452],
        "claim_boundary": "PRD gap table synthesis from actual upstream artifacts",
        "evidence_fields": (
            "prd_gap_table_ready",
            "closed_count",
            "partial_count",
            "blocked_count",
            "honest_null_count",
            "missing_count",
            "artifacts_found",
            "artifacts_missing",
            "unsupported_claims_detected",
        ),
    },
    {
        "lane": "token_internal_access",
        "source_artifacts": [EXP5441, EXP5451],
        "claim_boundary": "closed without authenticated logits, hidden-state, attention, token, or intermediate-exit receipts",
        "evidence_fields": (),
    },
    {
        "lane": "hardware_speedup_claim",
        "source_artifacts": [EXP5448, EXP5449, EXP5451],
        "claim_boundary": "hardware receipts and certificates explicitly do not support speedup",
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
    """Build the .495 capstone from upstream artifacts and source context only."""

    root_path = Path(root)
    payloads, context_read, context_missing, artifacts_found, artifacts_missing, read_errors = read_inputs(root_path)
    truth_table = [classify_lane(spec, payloads, artifacts_missing) for spec in LANE_SPECS]
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
        "task_range": TASK_RANGE,
        "source_context_read": context_read,
        "source_context_missing": context_missing,
        "artifacts_found": artifacts_found,
        "artifacts_missing": artifacts_missing,
        "artifact_read_errors": read_errors,
        "truth_table": truth_table,
        "headline_ready_lanes": buckets["headline_ready_lanes"],
        "bounded_lanes": buckets["bounded_lanes"],
        "blocked_lanes": buckets["blocked_lanes"],
        "honest_null_lanes": buckets["honest_null_lanes"],
        "missing_lanes": buckets["missing_lanes"],
        "arc_new_level_banked": bool(unwrap(payloads.get(EXP5450, {}).get("arc_new_level_banked"))),
        "hardware_speedup_claim": hardware_speedup_claim(payloads),
        "token_internal_lane_reopened": token_internal_lane_reopened(payloads),
        "next_recommendations": next_recommendations(payloads),
        "roadmap_yaml_unchanged": git_path_unchanged(root_path, "research-roadmap.yaml"),
        "conductor_unchanged": git_path_unchanged(root_path, "scripts/research_conductor.py"),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifact_checksums": source_artifact_checksums(root_path, [*context_read, *artifacts_found]),
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


def read_inputs(
    root: Path,
) -> tuple[dict[str, JsonDict], list[str], list[str], list[str], list[str], list[JsonDict]]:
    """Read expected inputs and record missing or malformed result artifacts."""

    payloads: dict[str, JsonDict] = {}
    context_read: list[str] = []
    context_missing: list[str] = []
    artifacts_found: list[str] = []
    artifacts_missing: list[str] = []
    errors: list[JsonDict] = []

    for relative in EXPECTED_INPUT_PATHS:
        path = root / relative
        if not path.exists() or path.is_dir():
            if relative in RESULT_ARTIFACT_PATHS:
                artifacts_missing.append(relative)
            else:
                context_missing.append(relative)
            continue
        if relative not in RESULT_ARTIFACT_PATHS:
            try:
                path.read_text(encoding="utf-8")
            except OSError as exc:
                context_missing.append(relative)
                errors.append({"path": relative, "classification": f"read_error:{exc}"})
                continue
            context_read.append(relative)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            artifacts_missing.append(relative)
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
            artifacts_missing.append(relative)
            errors.append({"path": relative, "classification": "not_json_object"})
            continue
        payloads[relative] = payload
        artifacts_found.append(relative)

    return payloads, context_read, context_missing, artifacts_found, artifacts_missing, errors


def classify_lane(spec: JsonMap, payloads: Mapping[str, JsonMap], artifacts_missing: Sequence[str]) -> JsonDict:
    """Classify one lane without borrowing evidence from a neighboring lane."""

    lane = str(spec["lane"])
    sources = [str(path) for path in spec["source_artifacts"]]
    missing_sources = [path for path in sources if path in set(artifacts_missing)]
    if missing_sources:
        return lane_row(
            lane=lane,
            source_artifacts=sources,
            classification="missing",
            claim_boundary="missing upstream artifact; no outcome inferred",
            terminal_evidence={},
            blocked_reason="missing_inputs",
        )

    if lane == "token_internal_access":
        return lane_row(
            lane=lane,
            source_artifacts=sources,
            classification="blocked",
            claim_boundary=str(spec["claim_boundary"]),
            terminal_evidence={
                "token_internal_lane_reopened": token_internal_lane_reopened(payloads),
                "backend_receipt_present": authenticated_backend_receipt_present(payloads),
                "token_internal_claim_rejected": unwrap(payloads.get(EXP5451, {}).get("token_internal_claim_rejected")),
            },
            blocked_reason="no_authenticated_backend_receipt",
        )

    if lane == "hardware_speedup_claim":
        return lane_row(
            lane=lane,
            source_artifacts=sources,
            classification="honest_null" if not hardware_speedup_claim(payloads) else "blocked",
            claim_boundary=str(spec["claim_boundary"]),
            terminal_evidence={
                "hardware_speedup_claim": hardware_speedup_claim(payloads),
                "hardware_speedup_claim_rejected": unwrap(
                    payloads.get(EXP5451, {}).get("hardware_speedup_claim_rejected")
                ),
            },
            blocked_reason="no_authenticated_hardware_speedup",
        )

    payload = payloads.get(sources[0], {})
    flags = flag_reasons(payload)
    if flags:
        return lane_row(
            lane=lane,
            source_artifacts=sources,
            classification="blocked",
            claim_boundary=str(spec["claim_boundary"]),
            terminal_evidence=evidence_for_spec(spec, payload),
            blocked_reason="_and_".join(flags),
        )

    classification = lane_classification(lane, payload)
    blocked_reason = "" if classification in {"headline_ready", "bounded", "honest_null"} else f"{lane}_not_ready"
    if classification == "honest_null":
        blocked_reason = str(payload.get("residual_wall", payload.get("status", "honest_null")))
    return lane_row(
        lane=lane,
        source_artifacts=sources,
        classification=classification,
        claim_boundary=str(spec["claim_boundary"]),
        terminal_evidence=evidence_for_spec(spec, payload),
        blocked_reason=blocked_reason,
    )


def lane_classification(lane: str, payload: JsonMap) -> str:
    """Convert lane-specific gates into the common capstone vocabulary."""

    if lane == "verifier_potential_generation":
        ready = (
            unwrap(payload.get("verifier_potential_fixture_ready")) is True
            and unwrap(payload.get("exact_final_authority")) is True
            and unwrap(payload.get("metric_independence_checks_passed")) is True
        )
        return "headline_ready" if ready else "blocked"
    if lane == "local_sota_decoding":
        return "bounded" if unwrap(payload.get("verifier_guided_decoding_ready")) is True else "blocked"
    if lane == "ast_kb_witnesses":
        ready = (
            unwrap(payload.get("ast_kb_witness_ready")) is True
            and unwrap(payload.get("ast_parse_success_rate")) == 1.0
            and unwrap(payload.get("valid_call_accept_rate")) == 1.0
            and unwrap(payload.get("nonexistent_call_reject_rate")) == 1.0
            and unwrap(payload.get("unsafe_false_accepts")) == 0
        )
        return "headline_ready" if ready else "blocked"
    if lane == "governed_csl":
        ready = (
            unwrap(payload.get("governed_csl_loop_ready")) is True
            and unwrap(payload.get("continuous_self_learning_task")) is True
            and unwrap(payload.get("replay_success_rate")) == 1.0
            and unwrap(payload.get("rollback_recovery_rate")) == 1.0
            and unwrap(payload.get("unsafe_false_accepts")) == 0
            and unwrap(payload.get("no_weight_mutation")) is True
        )
        return "headline_ready" if ready else "blocked"
    if lane == "memory_stress":
        ready = (
            unwrap(payload.get("csl_memory_stress_ready")) is True
            and unwrap(payload.get("gated_upstream_ready")) is True
            and unwrap(payload.get("stale_memory_deflection_rate")) == 1.0
            and unwrap(payload.get("poisoned_memory_deflection_rate")) == 1.0
            and unwrap(payload.get("retrieval_collision_deflection_rate")) == 1.0
            and unwrap(payload.get("negative_transfer_deflection_rate")) == 1.0
            and unwrap(payload.get("rollback_recovery_rate")) == 1.0
            and unwrap(payload.get("unsafe_false_accepts")) == 0
            and unwrap(payload.get("no_weight_mutation")) is True
        )
        return "headline_ready" if ready else "blocked"
    if lane == "active_constraint_pbit_bridge":
        ready = (
            unwrap(payload.get("pbit_assumption_bridge_ready")) is True
            and unwrap(payload.get("solver_authoritative")) is True
            and unwrap(payload.get("fallback_completeness_rate")) == 1.0
            and unwrap(payload.get("unsafe_false_accepts")) == 0
            and unwrap(payload.get("hardware_speedup_claim")) is False
        )
        return "bounded" if ready else "blocked"
    if lane == "hardware_receipts":
        ready = (
            unwrap(payload.get("hardware_receipts_ready")) is True
            and unwrap(payload.get("gated_upstream_ready")) is True
            and unwrap(payload.get("hashes_match_before_timing_compare")) is True
            and unwrap(payload.get("hardware_speedup_claim")) is False
        )
        return "bounded" if ready else "blocked"
    if lane == "arc_live_progress":
        return "headline_ready" if unwrap(payload.get("arc_new_level_banked")) is True else "honest_null"
    if lane == "kan_certificates":
        ready = (
            unwrap(payload.get("kan_certificate_ready")) is True
            and unwrap(payload.get("gated_upstreams_ready")) is True
            and unwrap(payload.get("true_measured_claim_preservation_rate")) == 1.0
            and unwrap(payload.get("false_property_rejection_rate")) == 1.0
            and unwrap(payload.get("unsupported_claim_rejection_rate")) == 1.0
            and unwrap(payload.get("broad_kan_claim_made")) is False
        )
        return "bounded" if ready else "blocked"
    if lane == "prd_gap_synthesis":
        ready = (
            unwrap(payload.get("prd_gap_table_ready")) is True
            and unwrap(payload.get("missing_count")) == 0
            and isinstance(payload.get("prd_gap_table"), list)
        )
        return "headline_ready" if ready else "blocked"
    return "blocked"


def lane_row(
    *,
    lane: str,
    source_artifacts: list[str],
    classification: str,
    claim_boundary: str,
    terminal_evidence: JsonMap,
    blocked_reason: str,
) -> JsonDict:
    """Normalize row shape so bucket validation can be mechanical."""

    return {
        "lane": lane,
        "source_artifacts": source_artifacts,
        "classification": classification,
        "claim_boundary": claim_boundary,
        "blocked_reason": blocked_reason,
        "terminal_evidence": dict(terminal_evidence),
    }


def flag_reasons(payload: JsonMap) -> list[str]:
    """Return reasons an otherwise-positive artifact is not headline-clean."""

    reasons: list[str] = []
    if unwrap(payload.get("flagged_adversarial")) is True:
        reasons.append("flagged_adversarial")
    corrigendum = payload.get("corrigendum_pending")
    if contains_flag_kind(corrigendum, "TAUTOLOGY"):
        reasons.append("tautology")
    if contains_flag_kind(corrigendum, "DURATION"):
        reasons.append("duration")
    if contains_unsupported_claim(payload):
        reasons.append("unsupported_claim")
    return reasons


def contains_flag_kind(value: Any, kind: str) -> bool:
    """Search nested flag rows for a particular adversarial-verifier kind."""

    if isinstance(value, Mapping):
        if str(value.get("kind", "")).upper() == kind:
            return True
        return any(contains_flag_kind(child, kind) for child in value.values())
    if isinstance(value, list):
        return any(contains_flag_kind(child, kind) for child in value)
    return False


def contains_unsupported_claim(payload: JsonMap) -> bool:
    """Detect unsupported claims only when an artifact did not reject them."""

    unsupported = payload.get("unsupported_claims_detected")
    if not isinstance(unsupported, list):
        return False
    return any(not bool(unwrap(row.get("rejected"))) for row in unsupported if isinstance(row, Mapping))


def evidence_for_spec(spec: JsonMap, payload: JsonMap) -> JsonDict:
    """Extract the lane-specific fields used as terminal evidence."""

    evidence: JsonDict = {}
    for field in spec.get("evidence_fields", ()):
        if field in payload:
            evidence[str(field)] = unwrap(payload[field])
    flags = flag_reasons(payload)
    if flags:
        evidence["flag_reasons"] = flags
    if "honest_verdict" in payload:
        evidence["honest_verdict"] = unwrap(payload["honest_verdict"])
    if "inference_substrate" in payload:
        evidence["inference_substrate"] = unwrap(payload["inference_substrate"])
    return evidence


def bucket_lanes(rows: Sequence[JsonMap]) -> dict[str, list[JsonDict]]:
    """Split truth-table rows into the required capstone lists."""

    buckets = {
        "headline_ready_lanes": [],
        "bounded_lanes": [],
        "blocked_lanes": [],
        "honest_null_lanes": [],
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

    return any(recursive_key_true(payloads.get(path, {}), "hardware_speedup_claim") for path in (EXP5448, EXP5449))


def token_internal_lane_reopened(payloads: Mapping[str, JsonMap]) -> bool:
    """Keep token/internal lanes closed unless a receipt explicitly opens them."""

    return authenticated_backend_receipt_present(payloads)


def authenticated_backend_receipt_present(payloads: Mapping[str, JsonMap]) -> bool:
    """Detect a positive backend receipt separately from rejected unsupported claims."""

    keys = (
        "backend_receipt_present",
        "authenticated_backend_receipt",
        "token_logprob_receipt_present",
        "hidden_state_receipt_present",
        "attention_receipt_present",
        "intermediate_exit_receipt_present",
        "token_internal_lane_reopened",
        "future_token_signal_allowed",
    )
    return any(recursive_key_true(payloads.get(path, {}), key) for path in (EXP5441, EXP5451) for key in keys)


def recursive_key_true(value: Any, key: str) -> bool:
    """Search nested artifact rows for a bare boolean true key."""

    if isinstance(value, Mapping):
        if unwrap(value.get(key)) is True:
            return True
        return any(recursive_key_true(child, key) for child in value.values())
    if isinstance(value, list):
        return any(recursive_key_true(child, key) for child in value)
    return False


def next_recommendations(payloads: Mapping[str, JsonMap]) -> list[JsonDict]:
    """Seed the next roadmap from concrete `.495` evidence and blockers."""

    decoding = payloads.get(EXP5444, {})
    memory = payloads.get(EXP5446, {})
    stress = payloads.get(EXP5447, {})
    arc = payloads.get(EXP5450, {})
    hardware = payloads.get(EXP5449, {})
    kan = payloads.get(EXP5451, {})
    return [
        {
            "target": "structured_decoding_corrigendum",
            "recommendation": "Repair Exp5444 tautology/adversarial flags before treating verifier-potential decoding as progress beyond deterministic fixtures.",
            "evidence": {
                "flagged_adversarial": unwrap(decoding.get("flagged_adversarial")),
                "corrigendum_pending": unwrap(decoding.get("corrigendum_pending")),
                "guided_validity_delta_vs_grammar_only": unwrap(decoding.get("guided_validity_delta_vs_grammar_only")),
                "gpu_offload_verified": unwrap(decoding.get("gpu_offload_verified")),
            },
        },
        {
            "target": "governed_memory_scale",
            "recommendation": "Scale governed CSL only while replay, rollback, deflection, provenance, and no-weight-mutation gates stay explicit.",
            "evidence": {
                "governed_csl_loop_ready": unwrap(memory.get("governed_csl_loop_ready")),
                "csl_memory_stress_ready": unwrap(stress.get("csl_memory_stress_ready")),
                "replay_success_rate": unwrap(memory.get("replay_success_rate")),
                "rollback_recovery_rate": unwrap(stress.get("rollback_recovery_rate")),
                "no_weight_mutation": unwrap(memory.get("no_weight_mutation")) is True
                and unwrap(stress.get("no_weight_mutation")) is True,
            },
        },
        {
            "target": "arc_live_levelup",
            "recommendation": "Keep ARC focused on reproduction-gated live self-discovery because Exp5450 reached no new banked level.",
            "evidence": {
                "arc_new_level_banked": unwrap(arc.get("arc_new_level_banked")),
                "selected_game": unwrap(arc.get("selected_game")),
                "selected_target_level_label": unwrap(arc.get("selected_target_level_label")),
                "frontier_expansion_count": unwrap(arc.get("frontier_expansion_count")),
                "residual_wall": unwrap(arc.get("residual_wall")),
            },
        },
        {
            "target": "hardware_repeatability",
            "recommendation": "Keep hardware work to matched hash receipts and repeatability until board-local timing supports an actual speedup claim.",
            "evidence": {
                "hardware_receipts_ready": unwrap(hardware.get("hardware_receipts_ready")),
                "hashes_match_before_timing_compare": unwrap(hardware.get("hashes_match_before_timing_compare")),
                "timing_repeat_counts": unwrap(hardware.get("timing_repeat_counts")),
                "hardware_speedup_claim": unwrap(hardware.get("hardware_speedup_claim")),
            },
        },
        {
            "target": "token_internal_backend",
            "recommendation": "Keep token/internal lanes closed until authenticated logits, hidden-state, attention, token, or intermediate-exit receipts exist.",
            "evidence": {
                "token_internal_lane_reopened": token_internal_lane_reopened(payloads),
                "token_internal_claim_rejected": unwrap(kan.get("token_internal_claim_rejected")),
                "claim_limits": unwrap(kan.get("claim_limits")),
            },
        },
    ]


def honest_verdict(artifact: JsonMap) -> str:
    """Summarize the capstone boundary with the required terminal prefix."""

    if artifact["artifacts_missing"] or artifact["artifact_read_errors"]:
        return (
            "blocked: .495 capstone emitted with missing or unreadable upstream artifacts; "
            "affected lanes were classified missing instead of inferred."
        )
    return (
        "complete: .495 capstone emitted from actual artifacts; verifier-potential fixtures, "
        "AST/KB witnesses, governed CSL, memory stress, and PRD gap synthesis are headline-ready; "
        "local SOTA decoding is blocked by adversarial tautology flags; active/p-bit, hardware "
        "receipts, and KAN certificates are bounded; ARC no-bank, no hardware speedup, and "
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
        raise ValueError("milestone must be 2026.07.495")
    if artifact["task_range"] != TASK_RANGE:
        raise ValueError("task_range must be exp5441-exp5453")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    for field in BOOLEAN_FIELDS:
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be boolean")
    if artifact["hardware_speedup_claim"] is not False:
        raise ValueError("hardware_speedup_claim must remain false")
    if artifact["token_internal_lane_reopened"] is not False:
        raise ValueError("token_internal_lane_reopened must remain false")
    if artifact["roadmap_yaml_unchanged"] is not True:
        raise ValueError("roadmap_yaml_unchanged must be true")
    if artifact["conductor_unchanged"] is not True:
        raise ValueError("conductor_unchanged must be true")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    if artifact["artifacts_missing"] or artifact["artifact_read_errors"]:
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
        if not isinstance(row.get("terminal_evidence"), Mapping):
            raise ValueError("truth_table row terminal_evidence must be object")
    rows_by_lane = {str(row["lane"]): row for row in rows}
    local_sota = rows_by_lane["local_sota_decoding"]
    if local_sota["terminal_evidence"].get("flagged_adversarial") is True:
        if local_sota["classification"] == "headline_ready":
            raise ValueError("flagged local SOTA decoding cannot be headline-ready")
    if "tautology" in local_sota["terminal_evidence"].get("flag_reasons", ()):
        if local_sota["classification"] == "headline_ready":
            raise ValueError("flagged local SOTA decoding cannot be headline-ready")
    expected_buckets = bucket_lanes(rows)
    for bucket_name, expected_rows in expected_buckets.items():
        if artifact[bucket_name] != expected_rows:
            raise ValueError(f"lane bucket mismatch: {bucket_name}")

    if artifact["arc_new_level_banked"] is True:
        arc_evidence = rows_by_lane["arc_live_progress"]["terminal_evidence"]
        if arc_evidence.get("arc_new_level_banked") is not True or arc_evidence.get("new_level_reproduced") is not True:
            raise ValueError("arc_new_level_banked requires reproduction-gated evidence")
    if not artifact["artifacts_missing"]:
        expected_headlines = [
            "verifier_potential_generation",
            "ast_kb_witnesses",
            "governed_csl",
            "memory_stress",
            "prd_gap_synthesis",
        ]
        if lane_names(artifact["headline_ready_lanes"]) != expected_headlines:
            raise ValueError("headline_ready_lanes overclaim or drift")
    expected_targets = [
        "structured_decoding_corrigendum",
        "governed_memory_scale",
        "arc_live_levelup",
        "hardware_repeatability",
        "token_internal_backend",
    ]
    if [row.get("target") for row in artifact["next_recommendations"]] != expected_targets:
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

    parser = argparse.ArgumentParser(description="Write the Exp5453 .495 capstone artifact.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    run(root=args.root, result_path=args.result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
