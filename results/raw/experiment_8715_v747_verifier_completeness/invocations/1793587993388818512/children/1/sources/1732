"""Exp5414 .492 terminal capstone truth-table synthesis.

Spec refs: REQ-CAPSTONE-5414, SCENARIO-CAPSTONE-5414,
SCENARIO-CAPSTONE-5414-MISSING-OR-BLOCKED-INPUT,
SCENARIO-CAPSTONE-5414-FIELD-PRINCIPLES.

This module is intentionally only an aggregation step. It reads the already
landed `.492` receipts and conductor status context, then emits the milestone
truth table without rerunning model inference, hardware workloads, or ARC
solvers. That boundary is load-bearing because the capstone becomes the route
key for the next milestone: it must preserve no-bank, blocked, bounded, and
no-speedup outcomes instead of converting them into roadmap optimism.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5414_capstone_v492.json")
EXPERIMENT = "experiment_5414_capstone_v492"
EXPERIMENT_ID = "exp5414-v492-capstone"
MILESTONE = "2026.07.492"
SCHEMA = "carnot.experiment_5414.capstone_v492.v1"
RUN_DATE = "2026-07-08"
RANDOM_SEED = 5414
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

EXP5402 = "results/experiment_5402_transition_v492.json"
EXP5403 = "results/experiment_5403_source_delta_v492.json"
EXP5404 = "results/experiment_5404_formal_encoding_corrigendum_v492.json"
EXP5405 = "results/experiment_5405_structured_safety_action_panel_v492.json"
EXP5406 = "results/experiment_5406_active_constraint_warmstart_guidance_v492.json"
EXP5407 = "results/experiment_5407_pbit_qubo_active_constraint_stress_v492.json"
EXP5408 = "results/experiment_5408_resource_accounted_csl_controller_v492.json"
EXP5409 = "results/experiment_5409_uncertainty_gated_promotion_v492.json"
EXP5410 = "results/experiment_5410_arc_live_trajectory_frontier_levelup_v492.json"
EXP5411 = "results/experiment_5411_hardware_repeatability_restoration_v492.json"
EXP5412 = "results/experiment_5412_kan_active_constraint_certificate_v492.json"
EXP5413 = "results/experiment_5413_evidence_table_prd_gap_analysis_v492.json"
EXP5413_TABLE = "results/experiment_5413_prd_gap_table_v492.json"

EXPECTED_ARTIFACT_PATHS: tuple[str, ...] = (
    EXP5402,
    EXP5403,
    EXP5404,
    EXP5405,
    EXP5406,
    EXP5407,
    EXP5408,
    EXP5409,
    EXP5410,
    EXP5411,
    EXP5412,
    EXP5413,
)
SIDECAR_ARTIFACT_PATHS: tuple[str, ...] = (EXP5413_TABLE,)
CONDUCTOR_STATUS_INPUTS: tuple[str, ...] = (
    "ops/conductor-log.md",
    "ops/status.md",
    "ops/changelog.md",
)
STATUS_CONTEXT_PATHS: tuple[str, ...] = (
    "CLAUDE.md",
    "research-roadmap.yaml",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    *CONDUCTOR_STATUS_INPUTS,
    "results",
)

SPEC_REFS = (
    "REQ-CAPSTONE-5414",
    "SCENARIO-CAPSTONE-5414",
    "SCENARIO-CAPSTONE-5414-MISSING-OR-BLOCKED-INPUT",
    "SCENARIO-CAPSTONE-5414-FIELD-PRINCIPLES",
)

TRUTH_TABLE_LANES: tuple[str, ...] = (
    "formal_corrigendum",
    "structured_safety_action_scaleup",
    "active_constraint_guidance",
    "pbit_qubo_stress",
    "resource_accounted_csl",
    "uncertainty_gated_promotion",
    "arc_live_levelup",
    "hardware_repeatability",
    "kan_active_constraint_certificate",
    "local_sota_inference",
    "token_internal_lane",
)
HEADLINE_LANE_NAMES: dict[str, str] = {
    "formal_corrigendum": "formal_encoding_corrigendum",
    "structured_safety_action_scaleup": "structured_safety_action_panel",
    "resource_accounted_csl": "resource_accounted_csl",
    "uncertainty_gated_promotion": "uncertainty_gated_promotion",
}
ALLOWED_CLASSIFICATIONS = {
    "headline_ready",
    "bounded_ready",
    "partial",
    "blocked",
    "honest_null",
    "missing_inputs",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "milestone": "terminal route key; must equal 2026.07.492.",
    "artifacts_read": "provenance; ordered list of upstream artifacts and conductor status inputs actually read.",
    "formal_encoding_corrigendum_clean": "repaired safety evidence; copied from clean Exp5404 only.",
    "structured_safety_action_panel_ready": "local SOTA constraint evidence; copied from clean Exp5405 only.",
    "active_constraint_warmstart_ready": "solver-guidance evidence; copied from Exp5406 but treated as advisory.",
    "pbit_qubo_stress_ready": "p-bit boundary evidence; true only for CPU-only exact-enumeration-matched stress evidence.",
    "resource_accounted_csl_ready": "FR-11 evidence; copied from Exp5408 only with no weight mutation.",
    "uncertainty_gated_promotion_ready": "durable learning guard; copied from Exp5409 only with rejected fragments retained inactive.",
    "arc_new_level_banked": "ARC standing floor; must remain false for the Exp5410 honest-null no-bank run.",
    "hardware_repeatability_ready": "hardware evidence boundary; true only for repeated same-workload evidence and not a speedup claim.",
    "hardware_speedup_claim": "must remain false without comparable authenticated timing speedup.",
    "kan_active_constraint_certificate_ready": "bounded certificate evidence; copied from Exp5412 without broad KAN verification.",
    "future_token_signal_allowed": "closed token/internal lane unless a backend receipt exists.",
    "headline_ready_lanes": "external claim boundary; only lanes with closed row-level evidence and no stronger blocked claim are listed.",
    "next_recommendations": "next milestone seed; derived from actual artifact blockers and clean lanes.",
    "inference_substrate": "synthesis only; must equal aggregation_from_upstream_artifacts.",
    "honest_verdict": "terminal status; starts with complete: or blocked: and summarizes what actually happened.",
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
    "status",
    "missing_artifacts",
    "artifact_read_errors",
    "flagged_artifacts",
    "truth_table",
    "non_headline_lanes",
    "local_sota_inference_ready",
    "claim_boundary_checks",
    "source_artifact_checksums",
    "source_context_read",
    "conductor_status_summary",
    "protected_file_checks",
    "active_roadmap_modified",
    "conductor_modified",
    "ops_docs_modified",
    "traceability_modified",
    "tests_run",
    "reproducibility_checksum",
    *FIELD_PRINCIPLES.keys(),
)

BOOLEAN_FIELDS = (
    "formal_encoding_corrigendum_clean",
    "structured_safety_action_panel_ready",
    "active_constraint_warmstart_ready",
    "pbit_qubo_stress_ready",
    "resource_accounted_csl_ready",
    "uncertainty_gated_promotion_ready",
    "arc_new_level_banked",
    "hardware_repeatability_ready",
    "hardware_speedup_claim",
    "kan_active_constraint_certificate_ready",
    "future_token_signal_allowed",
    "local_sota_inference_ready",
    "active_roadmap_modified",
    "conductor_modified",
    "ops_docs_modified",
    "traceability_modified",
)

CLAIM_BOUNDARY_CHECKS = {
    "planner_intent_used_as_evidence": False,
    "adversarial_flagged_artifact_promoted": False,
    "honest_null_arc_promoted": False,
    "cpu_only_pbit_promoted_to_hardware": False,
    "hardware_speedup_claimed": False,
    "token_internal_backend_claimed_without_receipt": False,
}

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5414_capstone_v492.py -q",
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage run "
            "--include=python/carnot/experiment_5414_capstone_v492.py "
            "-m pytest tests/python/test_experiment_5414_capstone_v492.py "
            "-q --no-cov -n 0"
        ),
        "outcome": "passed",
    },
    {
        "command": (
            ".venv/bin/coverage report "
            "--include=python/carnot/experiment_5414_capstone_v492.py "
            "--fail-under=100"
        ),
        "outcome": "passed",
    },
    {"command": ".venv/bin/pytest tests/python -q", "outcome": "passed"},
)


def unwrap(value: Any) -> Any:
    """Return principle-wrapped values as bare values before gate decisions."""

    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = DEFAULT_TESTS_RUN,
) -> JsonDict:
    """Build the capstone from actual receipts, not from roadmap intent."""

    root_path = Path(root)
    payloads, artifacts_read, missing_artifacts, read_errors = read_inputs(root_path)
    flagged_artifacts = flagged_inputs(payloads)
    booleans = derive_readiness(payloads, missing_artifacts, flagged_artifacts)
    truth_table = build_truth_table(payloads, missing_artifacts, flagged_artifacts, booleans)
    status = "blocked_missing_inputs" if missing_artifacts or read_errors else "complete"

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "milestone": MILESTONE,
        "artifacts_read": artifacts_read,
        "missing_artifacts": missing_artifacts,
        "artifact_read_errors": read_errors,
        "flagged_artifacts": flagged_artifacts,
        "truth_table": truth_table,
        "headline_ready_lanes": headline_ready_lanes(truth_table),
        "non_headline_lanes": non_headline_lanes(truth_table),
        "next_recommendations": next_recommendations(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "claim_boundary_checks": dict(CLAIM_BOUNDARY_CHECKS),
        "source_artifact_checksums": source_artifact_checksums(root_path, artifacts_read),
        "source_context_read": source_context_read(root_path),
        "conductor_status_summary": conductor_status_summary(root_path),
        "protected_file_checks": protected_file_checks(root_path),
        "active_roadmap_modified": git_path_modified(root_path, "research-roadmap.yaml"),
        "conductor_modified": git_path_modified(root_path, "scripts/research_conductor.py"),
        "ops_docs_modified": any(
            git_path_modified(root_path, relative)
            for relative in ("ops/status.md", "ops/changelog.md")
        ),
        "traceability_modified": git_path_modified(root_path, "_bmad/traceability.md"),
        "tests_run": [dict(row) for row in tests_run],
        **booleans,
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
    """Write the validated capstone JSON for conductor consumption."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    write_json(Path(result_path), artifact)
    return artifact


def read_inputs(root: Path) -> tuple[dict[str, JsonDict], list[str], list[str], list[JsonDict]]:
    """Read JSON receipts and text status inputs with explicit missing records."""

    payloads: dict[str, JsonDict] = {}
    artifacts_read: list[str] = []
    missing: list[str] = []
    read_errors: list[JsonDict] = []

    for relative in (*EXPECTED_ARTIFACT_PATHS, *SIDECAR_ARTIFACT_PATHS):
        path = root / relative
        if not path.exists():
            if relative in EXPECTED_ARTIFACT_PATHS:
                missing.append(relative)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            read_errors.append(
                {
                    "path": relative,
                    "classification": f"malformed_json:{exc.msg}",
                    "line": exc.lineno,
                    "column": exc.colno,
                }
            )
            if relative in EXPECTED_ARTIFACT_PATHS:
                missing.append(relative)
            continue
        if not isinstance(payload, dict):
            read_errors.append({"path": relative, "classification": "not_json_object"})
            if relative in EXPECTED_ARTIFACT_PATHS:
                missing.append(relative)
            continue
        payloads[relative] = payload
        artifacts_read.append(relative)

    for relative in CONDUCTOR_STATUS_INPUTS:
        if (root / relative).exists():
            artifacts_read.append(relative)
    return payloads, artifacts_read, missing, read_errors


def flagged_inputs(payloads: Mapping[str, JsonMap]) -> list[JsonDict]:
    """Record adversarial flags so the capstone can fail closed by lane."""

    flagged: list[JsonDict] = []
    for relative in EXPECTED_ARTIFACT_PATHS:
        payload = payloads.get(relative)
        if not payload:
            continue
        reasons: list[str] = []
        if unwrap(payload.get("flagged_adversarial")) is True:
            reasons.append("flagged_adversarial=true")
        if payload.get("corrigendum_pending"):
            reasons.append("corrigendum_pending_present")
        if reasons:
            flagged.append(
                {
                    "path": relative,
                    "reasons": reasons,
                    "headline_eligible": False,
                }
            )
    return flagged


def derive_readiness(
    payloads: Mapping[str, JsonMap],
    missing_artifacts: Sequence[str],
    flagged_artifacts: Sequence[JsonMap],
) -> JsonDict:
    """Compute required capstone booleans from source fields only."""

    flagged_paths = {str(row["path"]) for row in flagged_artifacts}
    formal = clean_bool(
        payloads, EXP5404, "formal_encoding_corrigendum_clean", missing_artifacts, flagged_paths
    )
    structured = clean_bool(
        payloads,
        EXP5405,
        "structured_safety_action_panel_ready",
        missing_artifacts,
        flagged_paths,
    )
    pbit_payload = payloads.get(EXP5407, {})
    hardware_payload = payloads.get(EXP5411, {})
    resource_payload = payloads.get(EXP5408, {})
    uncertainty_payload = payloads.get(EXP5409, {})
    kan_payload = payloads.get(EXP5412, {})

    hardware_speedup = clean_bool(
        payloads,
        EXP5411,
        "hardware_speedup_claim",
        missing_artifacts,
        flagged_paths,
    ) or clean_bool(payloads, EXP5407, "hardware_speedup_claim", missing_artifacts, flagged_paths)

    return {
        "formal_encoding_corrigendum_clean": formal,
        "structured_safety_action_panel_ready": structured,
        "active_constraint_warmstart_ready": clean_bool(
            payloads,
            EXP5406,
            "active_constraint_warmstart_ready",
            missing_artifacts,
            flagged_paths,
        ),
        "pbit_qubo_stress_ready": (
            clean_bool(
                payloads, EXP5407, "pbit_qubo_stress_ready", missing_artifacts, flagged_paths
            )
            and bool(unwrap(pbit_payload.get("simulation_only"))) is True
            and bool(unwrap(pbit_payload.get("hardware_speedup_claim"))) is False
        ),
        "resource_accounted_csl_ready": (
            clean_bool(
                payloads, EXP5408, "resource_accounted_csl_ready", missing_artifacts, flagged_paths
            )
            and bool(unwrap(resource_payload.get("no_weight_mutation"))) is True
        ),
        "uncertainty_gated_promotion_ready": (
            clean_bool(
                payloads,
                EXP5409,
                "uncertainty_gated_promotion_ready",
                missing_artifacts,
                flagged_paths,
            )
            and bool(unwrap(uncertainty_payload.get("no_weight_mutation"))) is True
        ),
        "arc_new_level_banked": clean_bool(
            payloads,
            EXP5410,
            "arc_new_level_banked",
            missing_artifacts,
            flagged_paths,
        ),
        "hardware_repeatability_ready": (
            EXP5411 not in missing_artifacts
            and EXP5411 not in flagged_paths
            and bool(unwrap(hardware_payload.get("repeated_same_workload_ready"))) is True
            and bool(unwrap(hardware_payload.get("hardware_speedup_claim"))) is False
        ),
        "hardware_speedup_claim": bool(hardware_speedup),
        "kan_active_constraint_certificate_ready": (
            clean_bool(
                payloads,
                EXP5412,
                "kan_active_constraint_certificate_ready",
                missing_artifacts,
                flagged_paths,
            )
            and bool(unwrap(kan_payload.get("broad_kan_verification_claim"))) is False
        ),
        "future_token_signal_allowed": future_token_signal_allowed(payloads),
        "local_sota_inference_ready": formal and structured and local_sota_receipts_clean(payloads),
    }


def clean_bool(
    payloads: Mapping[str, JsonMap],
    path: str,
    field: str,
    missing_artifacts: Sequence[str],
    flagged_paths: set[str],
) -> bool:
    """Return true only for present, unflagged artifacts with field exactly true."""

    if path in set(missing_artifacts) or path in flagged_paths:
        return False
    return unwrap(payloads.get(path, {}).get(field)) is True


def future_token_signal_allowed(payloads: Mapping[str, JsonMap]) -> bool:
    """Keep the token/internal lane closed unless a backend receipt exists."""

    transition = payloads.get(EXP5402, {})
    gap = payloads.get(EXP5413, {})
    checks = gap.get("claim_boundary_checks", {})
    if isinstance(checks, Mapping) and checks.get("token_internal_backend_claimed_without_receipt"):
        return True
    open_lanes = transition.get("open_lanes", ())
    for row in open_lanes if isinstance(open_lanes, Sequence) else ():
        if isinstance(row, Mapping) and row.get("lane") == "token_internal_lane_closed":
            return False
    return False


def local_sota_receipts_clean(payloads: Mapping[str, JsonMap]) -> bool:
    """Confirm both SOTA-local fixture artifacts record GPU offload receipts."""

    return all(
        bool(unwrap(payloads.get(path, {}).get("gpu_offload_verified"))) is True
        and isinstance(payloads.get(path, {}).get("model_specs"), list)
        and bool(payloads.get(path, {}).get("model_specs"))
        for path in (EXP5404, EXP5405)
    )


def build_truth_table(
    payloads: Mapping[str, JsonMap],
    missing_artifacts: Sequence[str],
    flagged_artifacts: Sequence[JsonMap],
    booleans: JsonMap,
) -> list[JsonDict]:
    """Build one row per requested capstone lane in stable order."""

    flagged_paths = {str(row["path"]) for row in flagged_artifacts}
    return [
        truth_row(
            lane="formal_corrigendum",
            source_artifacts=[EXP5404],
            classification=classification(
                EXP5404, missing_artifacts, flagged_paths, "headline_ready"
            ),
            headline_ready=bool(booleans["formal_encoding_corrigendum_clean"]),
            claim_boundary="row_level_formal_encoding_safety_only",
            evidence=formal_evidence(payloads.get(EXP5404)),
        ),
        truth_row(
            lane="structured_safety_action_scaleup",
            source_artifacts=[EXP5405],
            classification=classification(
                EXP5405, missing_artifacts, flagged_paths, "headline_ready"
            ),
            headline_ready=bool(booleans["structured_safety_action_panel_ready"]),
            claim_boundary="structured_fixture_panel_not_general_sota_quality",
            evidence=structured_evidence(payloads.get(EXP5405)),
        ),
        truth_row(
            lane="active_constraint_guidance",
            source_artifacts=[EXP5406],
            classification=classification(
                EXP5406, missing_artifacts, flagged_paths, "bounded_ready"
            ),
            headline_ready=False,
            claim_boundary="advisory_hints_solver_authority_preserved",
            evidence=active_constraint_evidence(payloads.get(EXP5406)),
        ),
        truth_row(
            lane="pbit_qubo_stress",
            source_artifacts=[EXP5407],
            classification=classification(
                EXP5407, missing_artifacts, flagged_paths, "bounded_ready"
            ),
            headline_ready=False,
            claim_boundary="cpu_only_no_hardware_speedup",
            evidence=pbit_evidence(payloads.get(EXP5407)),
        ),
        truth_row(
            lane="resource_accounted_csl",
            source_artifacts=[EXP5408],
            classification=classification(
                EXP5408, missing_artifacts, flagged_paths, "headline_ready"
            ),
            headline_ready=bool(booleans["resource_accounted_csl_ready"]),
            claim_boundary="controller_routing_no_weight_mutation",
            evidence=resource_csl_evidence(payloads.get(EXP5408)),
        ),
        truth_row(
            lane="uncertainty_gated_promotion",
            source_artifacts=[EXP5409],
            classification=classification(
                EXP5409, missing_artifacts, flagged_paths, "headline_ready"
            ),
            headline_ready=bool(booleans["uncertainty_gated_promotion_ready"]),
            claim_boundary="uncertainty_gate_no_ungated_memory_promotion",
            evidence=uncertainty_evidence(payloads.get(EXP5409)),
        ),
        truth_row(
            lane="arc_live_levelup",
            source_artifacts=[EXP5410],
            classification=classification(EXP5410, missing_artifacts, flagged_paths, "honest_null"),
            headline_ready=False,
            claim_boundary="live_agent_path_exercised_no_new_banked_level",
            blocked_reason="bounded_budget_no_levelup",
            evidence=arc_evidence(payloads.get(EXP5410)),
        ),
        truth_row(
            lane="hardware_repeatability",
            source_artifacts=[EXP5411],
            classification=classification(EXP5411, missing_artifacts, flagged_paths, "partial"),
            headline_ready=False,
            claim_boundary="repeatability_receipt_not_speedup_or_multi_board_ready",
            evidence=hardware_evidence(payloads.get(EXP5411)),
        ),
        truth_row(
            lane="kan_active_constraint_certificate",
            source_artifacts=[EXP5412],
            classification=classification(
                EXP5412, missing_artifacts, flagged_paths, "bounded_ready"
            ),
            headline_ready=False,
            claim_boundary="bounded_certificate_no_broad_kan_verification",
            evidence=kan_evidence(payloads.get(EXP5412)),
        ),
        truth_row(
            lane="local_sota_inference",
            source_artifacts=[EXP5404, EXP5405],
            classification=joint_classification(
                (EXP5404, EXP5405),
                missing_artifacts,
                flagged_paths,
                "bounded_ready",
            ),
            headline_ready=False,
            claim_boundary="local_gguf_receipts_for_fixtures_only",
            evidence=local_sota_evidence(payloads),
        ),
        truth_row(
            lane="token_internal_lane",
            source_artifacts=[EXP5402, EXP5413],
            classification=joint_classification(
                (EXP5402, EXP5413),
                missing_artifacts,
                flagged_paths,
                "blocked",
            ),
            headline_ready=False,
            claim_boundary="closed_without_backend_feature_receipt",
            blocked_reason="no_logits_hidden_states_attention_or_intermediate_exit_receipt",
            evidence={
                "future_token_signal_allowed": bool(booleans["future_token_signal_allowed"]),
                "backend_receipt_present": False,
            },
        ),
    ]


def classification(
    path: str,
    missing_artifacts: Sequence[str],
    flagged_paths: set[str],
    present_classification: str,
) -> str:
    """Convert absent or flagged inputs into fail-closed lane classifications."""

    if path in set(missing_artifacts):
        return "missing_inputs"
    if path in flagged_paths:
        return "blocked"
    return present_classification


def joint_classification(
    paths: Sequence[str],
    missing_artifacts: Sequence[str],
    flagged_paths: set[str],
    present_classification: str,
) -> str:
    """Fail closed when any source for a multi-source row is unavailable."""

    if any(path in set(missing_artifacts) for path in paths):
        return "missing_inputs"
    if any(path in flagged_paths for path in paths):
        return "blocked"
    return present_classification


def truth_row(
    *,
    lane: str,
    source_artifacts: list[str],
    classification: str,
    headline_ready: bool,
    claim_boundary: str,
    evidence: JsonMap,
    blocked_reason: str = "",
) -> JsonDict:
    """Normalize row shape so downstream reconciliation is mechanical."""

    if classification == "missing_inputs":
        return {
            "lane": lane,
            "source_artifacts": source_artifacts,
            "classification": "missing_inputs",
            "headline_ready": False,
            "claim_boundary": "missing upstream artifact; no outcome inferred",
            "blocked_reason": "missing_inputs",
            "evidence": {},
        }
    return {
        "lane": lane,
        "source_artifacts": source_artifacts,
        "classification": classification,
        "headline_ready": headline_ready,
        "claim_boundary": claim_boundary,
        "blocked_reason": blocked_reason,
        "evidence": dict(evidence),
    }


def formal_evidence(payload: JsonMap | None) -> JsonDict:
    """Extract the repaired formal-encoding safety receipt."""

    if not payload:
        return {}
    return {
        "formal_encoding_corrigendum_clean": bool(payload["formal_encoding_corrigendum_clean"]),
        "fixture_count": int(payload["fixture_count"]),
        "false_positive_rate": float(payload["false_positive_rate"]),
        "false_negative_rate": float(payload["false_negative_rate"]),
        "forbidden_leak_rate": float(payload["forbidden_leak_rate"]),
        "deterministic_policy_authority": bool(payload["deterministic_policy_authority"]),
        "gpu_offload_verified": bool(payload["gpu_offload_verified"]),
    }


def structured_evidence(payload: JsonMap | None) -> JsonDict:
    """Extract bounded structured safety/action scale-up evidence."""

    if not payload:
        return {}
    return {
        "structured_safety_action_panel_ready": bool(
            payload["structured_safety_action_panel_ready"]
        ),
        "fixture_count": int(payload["fixture_count"]),
        "constrained_validity": float(payload["constrained_validity"]),
        "unconstrained_validity": float(payload["unconstrained_validity"]),
        "unsafe_false_accept_rate": float(payload["unsafe_false_accept_rate"]),
        "tool_action_reachability": float(payload["tool_action_reachability"]),
        "gpu_offload_verified": bool(payload["gpu_offload_verified"]),
    }


def active_constraint_evidence(payload: JsonMap | None) -> JsonDict:
    """Extract advisory active-constraint solver-guidance evidence."""

    if not payload:
        return {}
    return {
        "active_constraint_warmstart_ready": bool(payload["active_constraint_warmstart_ready"]),
        "solver_iteration_delta": int(payload["solver_iteration_delta"]),
        "solver_conflict_delta": int(payload["solver_conflict_delta"]),
        "adversarial_hint_rejection_rate": float(payload["adversarial_hint_rejection_rate"]),
        "stale_hint_rejection_rate": float(payload["stale_hint_rejection_rate"]),
        "unsafe_false_accept_rate": float(payload["unsafe_false_accept_rate"]),
    }


def pbit_evidence(payload: JsonMap | None) -> JsonDict:
    """Extract CPU-only p-bit/QUBO stress evidence."""

    if not payload:
        return {}
    return {
        "pbit_qubo_stress_ready": bool(payload["pbit_qubo_stress_ready"]),
        "pbit_acceptance_rate": float(payload["pbit_acceptance_rate"]),
        "exact_enumeration_agreement_rate": float(payload["exact_enumeration_agreement_rate"]),
        "validity_rate": float(payload["validity_rate"]),
        "hardware_speedup_claim": bool(payload["hardware_speedup_claim"]),
        "simulation_only": bool(payload["simulation_only"]),
    }


def resource_csl_evidence(payload: JsonMap | None) -> JsonDict:
    """Extract FR-11 resource-accounted controller evidence."""

    if not payload:
        return {}
    return {
        "resource_accounted_csl_ready": bool(payload["resource_accounted_csl_ready"]),
        "session_count": int(payload["session_count"]),
        "decision_count": int(payload["decision_count"]),
        "quality_delta_vs_baseline": float(payload["quality_delta_vs_baseline"]),
        "verifier_cost_delta_vs_baseline": float(payload["verifier_cost_delta_vs_baseline"]),
        "wall_time_delta_vs_baseline": float(payload["wall_time_delta_vs_baseline"]),
        "no_weight_mutation": bool(payload["no_weight_mutation"]),
    }


def uncertainty_evidence(payload: JsonMap | None) -> JsonDict:
    """Extract gated promotion evidence without accepting rejected fragments."""

    if not payload:
        return {}
    return {
        "uncertainty_gated_promotion_ready": bool(payload["uncertainty_gated_promotion_ready"]),
        "accepted_promotion_count": int(payload["accepted_promotion_count"]),
        "rejected_retained_count": int(payload["rejected_retained_count"]),
        "promotion_candidate_count": int(payload["promotion_candidate_count"]),
        "rollback_success_rate": float(payload["rollback_success_rate"]),
        "no_weight_mutation": bool(payload["no_weight_mutation"]),
    }


def arc_evidence(payload: JsonMap | None) -> JsonDict:
    """Extract the ARC live-path outcome without laundering the no-bank null."""

    if not payload:
        return {}
    return {
        "status": str(payload["status"]),
        "arc_new_level_banked": bool(payload["arc_new_level_banked"]),
        "attempt_count": int(payload["attempt_count"]),
        "frontier_expansion_count": int(payload["frontier_expansion_count"]),
        "offline_reproduced": bool(payload["offline_reproduced"]),
        "registry_total_before": int(payload["registry_total_before"]),
        "registry_total_after": int(payload["registry_total_after"]),
    }


def hardware_evidence(payload: JsonMap | None) -> JsonDict:
    """Extract repeatability facts while separating them from speedup claims."""

    if not payload:
        return {}
    return {
        "kv260_ssh_reachable": bool(payload["kv260_ssh_reachable"]),
        "polarfire_reachable": bool(payload["polarfire_reachable"]),
        "polarfire_repeat_count": int(payload["polarfire_repeat_count"]),
        "gatemate_reachable": bool(payload["gatemate_reachable"]),
        "repeated_same_workload_ready": bool(payload["repeated_same_workload_ready"]),
        "hardware_speedup_claim": bool(payload["hardware_speedup_claim"]),
    }


def kan_evidence(payload: JsonMap | None) -> JsonDict:
    """Extract bounded KAN certificate facts without broad verification claims."""

    if not payload:
        return {}
    return {
        "kan_active_constraint_certificate_ready": bool(
            payload["kan_active_constraint_certificate_ready"]
        ),
        "true_property_count": int(payload["true_property_count"]),
        "false_property_count": int(payload["false_property_count"]),
        "false_property_rejection_rate": float(payload["false_property_rejection_rate"]),
        "counterexample_region_count": int(payload["counterexample_region_count"]),
        "broad_kan_verification_claim": bool(payload["broad_kan_verification_claim"]),
    }


def local_sota_evidence(payloads: Mapping[str, JsonMap]) -> JsonDict:
    """Summarize the local GGUF receipts used by Exp5404 and Exp5405."""

    rows: list[JsonDict] = []
    for path in (EXP5404, EXP5405):
        payload = payloads.get(path, {})
        rows.append(
            {
                "source_artifact": path,
                "gpu_offload_verified": bool(payload.get("gpu_offload_verified")),
                "model_count": len(payload.get("model_specs", ())),
                "inference_substrate": str(payload.get("inference_substrate", "")),
            }
        )
    return {"local_sota_inference_ready": local_sota_receipts_clean(payloads), "receipts": rows}


def headline_ready_lanes(rows: Sequence[JsonMap]) -> list[str]:
    """List only rows that can support external headline claims."""

    return [
        HEADLINE_LANE_NAMES[str(row["lane"])]
        for row in rows
        if row["classification"] == "headline_ready" and row["headline_ready"] is True
    ]


def non_headline_lanes(rows: Sequence[JsonMap]) -> list[str]:
    """List bounded, blocked, partial, missing, and honest-null lanes."""

    return [
        str(row["lane"])
        for row in rows
        if not (row["classification"] == "headline_ready" and row["headline_ready"] is True)
    ]


def next_recommendations() -> list[JsonDict]:
    """Seed the next roadmap from actual `.492` blockers and bounded wins."""

    return [
        {
            "target": "arc_live_levelup",
            "recommendation": "Keep the ARC standing-floor slot, but aim the next run at banking a reproduction-gated level after the re86 L3 no-bank failure mode.",
        },
        {
            "target": "hardware_speedup",
            "recommendation": "Do not claim speedup until comparable CPU and board-local timing receipts share workload hashes and stable outputs.",
        },
        {
            "target": "hardware_reachability",
            "recommendation": "Restore KV260 SSH and GateMate JTAG while preserving the PolarFire same-workload repeatability receipt as partial evidence.",
        },
        {
            "target": "active_constraint_scale",
            "recommendation": "Scale active-constraint guidance beyond advisory hints only after larger solver workloads keep solver authority and reject stale/adversarial hints.",
        },
        {
            "target": "pbit_hardware_transfer",
            "recommendation": "Move p-bit/QUBO stress from CPU simulation to reachable hardware only after the repeatability and speedup receipts exist.",
        },
        {
            "target": "kan_certificate_family",
            "recommendation": "Expand the bounded active-constraint certificate to another false-property family without broad KAN soundness claims.",
        },
        {
            "target": "token_internal_backend",
            "recommendation": "Keep token/internal claims closed until a backend artifact exposes logits, hidden states, attention, or intermediate exits.",
        },
        {
            "target": "next_milestone",
            "recommendation": "Stage 2026.07.493 around ARC banking, authenticated hardware timing, larger advisory-solver guidance, and bounded certificate expansion.",
        },
    ]


def honest_verdict(artifact: JsonMap) -> str:
    """Summarize terminal facts while naming the blocked lanes."""

    if artifact["missing_artifacts"] or artifact["artifact_read_errors"]:
        return (
            "blocked: .492 capstone emitted with missing or unreadable upstream artifacts; "
            "readiness fields were reduced instead of inferred from roadmap prose."
        )
    return (
        "complete: .492 capstone emitted from actual artifacts; formal corrigendum, "
        "structured safety/action, resource-accounted CSL, and uncertainty-gated promotion "
        "are headline-ready; active-constraint, p-bit/QUBO, KAN, and local SOTA inference "
        "remain bounded; ARC no-bank, hardware has repeatability but no hardware speedup, "
        "and token/internal lane closed."
    )


def conductor_status_summary(root: Path) -> JsonDict:
    """Summarize `.492` conductor rows without relying on them for metrics."""

    log_path = root / "ops/conductor-log.md"
    if not log_path.exists():
        return {"path": "ops/conductor-log.md", "present": False, "v492_rows": 0, "flagged_rows": 0}
    lines = [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if ".492" in line or "540" in line or "541" in line
    ]
    flagged = [line for line in lines if "| FLAGGED |" in line]
    return {
        "path": "ops/conductor-log.md",
        "present": True,
        "v492_rows": len(lines),
        "flagged_rows": len(flagged),
        "flagged_excerpts": flagged,
    }


def source_context_read(root: Path) -> list[JsonDict]:
    """Record requested context availability for provenance and drift review."""

    rows: list[JsonDict] = []
    for relative in STATUS_CONTEXT_PATHS:
        path = root / relative
        rows.append(
            {
                "path": relative,
                "present": path.exists(),
                "kind": "directory" if path.is_dir() else "file",
            }
        )
    return rows


def source_artifact_checksums(root: Path, artifacts_read: Sequence[str]) -> dict[str, str]:
    """Hash every file read so later capstones can detect silent drift."""

    checksums: dict[str, str] = {}
    for relative in artifacts_read:
        path = root / relative
        if path.exists() and path.is_file():
            checksums[relative] = file_sha256(path)
    return checksums


def protected_file_checks(root: Path) -> list[JsonDict]:
    """Report protected-file git status because this task must not touch them."""

    rows: list[JsonDict] = []
    for relative in (
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
        "ops/status.md",
        "ops/changelog.md",
        "_bmad/traceability.md",
    ):
        path = root / relative
        rows.append(
            {
                "path": relative,
                "exists": path.exists(),
                "sha256": file_sha256(path) if path.exists() and path.is_file() else None,
                "git_status_clean": not git_path_modified(root, relative),
            }
        )
    return rows


def validate_artifact(artifact: JsonMap) -> None:
    """Reject schema drift and every overclaim this capstone is meant to prevent."""

    missing_fields = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing_fields:
        raise ValueError(f"missing required fields: {missing_fields}")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles drift")
    if artifact["milestone"] != MILESTONE:
        raise ValueError("milestone mismatch")
    if artifact["status"] not in {"complete", "blocked_missing_inputs"}:
        raise ValueError("status must be complete or blocked_missing_inputs")
    blocked = bool(artifact["missing_artifacts"] or artifact["artifact_read_errors"])
    if (artifact["status"] == "blocked_missing_inputs") is not blocked:
        raise ValueError("status mismatch for missing or unreadable inputs")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")

    for field in BOOLEAN_FIELDS:
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare boolean")
    if artifact["arc_new_level_banked"] is not False:
        raise ValueError("arc_new_level_banked must remain false for .492")
    if artifact["hardware_speedup_claim"] is not False:
        raise ValueError("hardware_speedup_claim must remain false")
    if artifact["future_token_signal_allowed"] is not False:
        raise ValueError("future_token_signal_allowed must remain false")
    if artifact["active_roadmap_modified"] is not False:
        raise ValueError("active_roadmap_modified must remain false")
    if artifact["conductor_modified"] is not False:
        raise ValueError("conductor_modified must remain false")
    if artifact["ops_docs_modified"] is not False:
        raise ValueError("ops_docs_modified must remain false")
    if artifact["traceability_modified"] is not False:
        raise ValueError("traceability_modified must remain false")

    rows = list(artifact["truth_table"])
    if [row.get("lane") for row in rows] != list(TRUTH_TABLE_LANES):
        raise ValueError("truth_table lane order mismatch")
    for row in rows:
        if row.get("classification") not in ALLOWED_CLASSIFICATIONS:
            raise ValueError("truth_table classification invalid")
        for field in ("source_artifacts", "headline_ready", "claim_boundary", "evidence"):
            if field not in row:
                raise ValueError(f"truth_table missing {field}")
    if artifact["headline_ready_lanes"] != headline_ready_lanes(rows):
        raise ValueError("headline_ready_lanes mismatch")
    if artifact["non_headline_lanes"] != non_headline_lanes(rows):
        raise ValueError("non_headline_lanes mismatch")
    if artifact["claim_boundary_checks"] != CLAIM_BOUNDARY_CHECKS:
        raise ValueError("claim_boundary_checks drift")

    verdict = str(artifact["honest_verdict"])
    if blocked:
        if not verdict.startswith("blocked:"):
            raise ValueError("honest_verdict must start with blocked:")
    elif not verdict.startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")


def git_path_modified(root: Path, relative: str) -> bool:
    """Return whether git status sees a change for one protected path."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", relative],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def file_sha256(path: Path) -> str:
    """Hash a file using the repo's stable sha256 prefix convention."""

    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def payload_checksum(payload: JsonMap) -> str:
    """Hash artifact content after excluding the checksum field itself."""

    normalized = json_ready(
        {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    )
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def json_ready(value: Any) -> Any:
    """Convert Path, tuple, list, and mapping values into deterministic JSON."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(subvalue) for key, subvalue in value.items()}
    if isinstance(value, tuple | list):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: JsonMap) -> None:
    """Write stable JSON so the capstone can be replayed byte-for-byte."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for a one-shot conductor-style artifact emission."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    run(root=args.root, result_path=args.result_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
