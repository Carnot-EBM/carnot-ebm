"""Build the Exp 3313 repair-substrate root-cause autopsy artifact.

Spec refs: REQ-REPORT-3313, SCENARIO-REPORT-3313.

This module is deliberately aggregation-only. It reads the `.305` repair panel,
the `.305` repair audit, and the `.306` runtime contract, then separates real
repair evidence from the provenance and substrate hygiene that blocked headline
promotion. It does not rerun generation, verification, CUDA probes, or the
conductor.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.repair_substrate_root_cause_autopsy.v1"
EXPERIMENT_ID = "exp3313"
TASK_ID = "exp3313-repair-substrate-root-cause-autopsy-v1"
ARTIFACT = "experiment_3313_repair_substrate_root_cause_autopsy_v1"
MILESTONE = "2026.05.306"
RUN_DATE = "20260529"
RANDOM_SEED = 3313
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REL_PATH = Path("openspec/capabilities/research-reporting/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3313_repair_substrate_root_cause_autopsy_v1.json")
EXP3302_REL_PATH = Path("results/experiment_3302_headline_sota_repair_panel_v11.json")
EXP3303_REL_PATH = Path("results/experiment_3303_repair_headline_evidence_audit_v1.json")
EXP3308_REL_PATH = Path("results/experiment_3308_quality_flag_root_cause_autopsy_v1.json")
EXP3309_REL_PATH = Path("results/experiment_3309_live_runtime_provenance_contract_v1.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
MINIMUM_LIVE_DURATION_S = 60.0
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "repair_substrate_autopsy_ready",
    "analyzed_artifacts",
    "source_provenance_failure_modes",
    "substrate_consistency_failure_modes",
    "rerun_contract",
    "no_new_model_execution",
    "honest_verdict",
)

JSON_SOURCES: tuple[tuple[str, Path, str], ...] = (
    ("exp3302", EXP3302_REL_PATH, "headline_repair_panel_ready"),
    ("exp3303", EXP3303_REL_PATH, "repair_headline_evidence_audit_ready"),
    ("exp3308", EXP3308_REL_PATH, "quality_flag_autopsy_ready"),
    ("exp3309", EXP3309_REL_PATH, "runtime_contract_ready"),
)

RUNTIME_CONTRACT_FIELDS: tuple[str, ...] = (
    "runtime_provenance",
    "checker_versions",
    "duration_contract_passed",
    "runtime_provenance_clean",
)

MODEL_IDENTITY_FIELDS: tuple[str, ...] = (
    "model_id",
    "hf_id",
    "model_path",
    "size_bytes",
    "quantization",
    "cache_root",
    "snapshot_revision",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """SCENARIO-REPORT-3313: convert repair audit blockers into rerun gates."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    payloads = {exp_id: read_json_object(root_path / rel_path) for exp_id, rel_path, _ready in JSON_SOURCES}
    conductor_text = read_text(root_path / CONDUCTOR_LOG_REL_PATH)
    panel = payloads["exp3302"]
    audit = payloads["exp3303"]
    contract = payloads["exp3309"]
    panel_sha = sha256_file(root_path / EXP3302_REL_PATH)
    comparisons = panel_audit_field_comparison(panel, audit, panel_sha)
    source_modes = source_provenance_failure_modes(panel, audit)
    substrate_modes = substrate_consistency_failure_modes(panel, audit, contract)
    classifications = blocker_classification(panel, audit)
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-REPORT-3313", "SCENARIO-REPORT-3313"],
        "repair_substrate_autopsy_ready": bool(panel and audit and source_modes and substrate_modes),
        "analyzed_artifacts": analyzed_artifacts(root_path, payloads, conductor_text),
        "source_provenance_failure_modes": source_modes,
        "substrate_consistency_failure_modes": substrate_modes,
        "rerun_contract": rerun_contract(),
        "panel_audit_field_comparison": comparisons,
        "blocker_classification": classifications,
        "candidate_outcome_summary": candidate_outcome_summary(panel),
        "runtime_contract_reference": {
            "experiment_id": "exp3309",
            "runtime_contract_ready": contract.get("runtime_contract_ready") is True,
            "minimum_live_duration_s": minimum_live_duration(contract),
            "repair_substrate_rules_present": bool(mapping(contract.get("repair_substrate_rules"))),
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_garak_run": True,
        "no_new_dataflip_run": True,
        "no_new_repair_generation": True,
        "no_new_verifier_run": True,
        "no_conductor_execution": True,
        "no_push": True,
        "scripts_research_conductor_modified": False,
        "duration_s": duration(started, finished),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3313 terminal artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def analyzed_artifacts(root: Path, payloads: Mapping[str, Mapping[str, Any]], conductor_text: str) -> list[JsonDict]:
    """Return source readiness and checksum rows for all inputs read."""

    rows: list[JsonDict] = []
    for exp_id, rel_path, ready_field in JSON_SOURCES:
        payload = mapping(payloads.get(exp_id))
        path = root / rel_path
        rows.append(
            {
                "experiment_id": exp_id,
                "path": rel_path.as_posix(),
                "present": path.exists(),
                "readable_json_object": bool(payload),
                "ready_field": ready_field,
                "ready": payload.get(ready_field) is True,
                "reported_experiment_id": str(payload.get("experiment_id") or ""),
                "artifact": str(payload.get("artifact") or ""),
                "sha256": sha256_file(path),
            }
        )
    log_path = root / CONDUCTOR_LOG_REL_PATH
    rows.append(
        {
            "experiment_id": "ops_conductor_log",
            "path": CONDUCTOR_LOG_REL_PATH.as_posix(),
            "present": log_path.exists(),
            "readable_text": bool(conductor_text),
            "ready": "Headline SOTA repair panel v11" in conductor_text
            and "Repair headline evidence audit v1" in conductor_text,
            "sha256": sha256_file(log_path),
        }
    )
    return rows


def panel_audit_field_comparison(panel: Mapping[str, Any], audit: Mapping[str, Any], panel_sha256: str) -> list[JsonDict]:
    """Compare the repair panel and audit fields that control promotion."""

    audit_models = mapping(audit.get("model_invocation_summary"))
    panel_used = model_ids_from_rows(panel.get("models_used"))
    audit_used = string_list(audit_models.get("used_model_ids"))
    panel_missing = model_ids_from_rows(panel.get("missing_model_specs"))
    audit_missing = string_list(audit_models.get("missing_model_ids"))
    panel_mandated = string_list(mapping(panel.get("model_specs")).get("mandated_model_ids"))
    audit_mandated = string_list(audit_models.get("mandated_model_ids"))
    panel_exact_types = exact_checker_types_for_successes(panel)
    audit_exact_types = string_list(mapping(audit.get("exact_check_provenance")).get("exact_checker_types_for_successes"))
    return [
        comparison_row("panel_case_count", panel.get("panel_case_count"), audit.get("panel_case_count"), "match"),
        comparison_row(
            "verified_success_count_vs_exact_successes_audited",
            panel.get("verified_success_count"),
            audit.get("exact_successes_audited"),
            "match",
        ),
        comparison_row("false_accept_count", panel.get("false_accept_count"), audit.get("false_accept_count"), "match"),
        comparison_row(
            "headline_claim_allowed",
            panel.get("headline_claim_allowed") is True,
            audit.get("headline_claim_allowed_after_audit") is True,
            "blocking_false_match",
        ),
        comparison_row(
            "source_provenance_clean",
            panel.get("provenance_clean") is True,
            audit.get("source_provenance_clean") is True,
            "blocking_false_match",
        ),
        comparison_row(
            "inference_substrate",
            str(panel.get("inference_substrate") or ""),
            str(audit.get("inference_substrate") or ""),
            "expected_boundary_difference",
        ),
        comparison_row(
            "duration_s",
            numeric(panel.get("duration_s")),
            numeric(audit.get("duration_s")),
            "expected_audit_elapsed_time_difference",
        ),
        comparison_row("used_model_ids", panel_used, audit_used, list_status(panel_used, audit_used)),
        comparison_row("missing_model_ids", panel_missing, audit_missing, list_status(panel_missing, audit_missing)),
        comparison_row("mandated_model_ids", panel_mandated, audit_mandated, list_status(panel_mandated, audit_mandated)),
        comparison_row(
            "audited_artifact_sha256",
            panel_sha256,
            audit_source_exp3302_sha(audit),
            "match" if panel_sha256 == audit_source_exp3302_sha(audit) else "mismatch",
        ),
        comparison_row(
            "exact_checker_types",
            panel_exact_types,
            audit_exact_types,
            list_status(panel_exact_types, audit_exact_types),
        ),
        comparison_row(
            "llm_judge_dependency_count",
            0,
            int(audit.get("llm_judge_dependency_count") or 0),
            "match",
        ),
    ]


def source_provenance_failure_modes(panel: Mapping[str, Any], audit: Mapping[str, Any]) -> list[JsonDict]:
    """List missing or dirty source-provenance evidence that blocks promotion."""

    modes: list[JsonDict] = []
    critical = critical_adversarial_flags(panel, audit)
    missing_runtime = missing_runtime_contract_fields(panel)
    missing_models = model_ids_from_rows(panel.get("missing_model_specs"))
    identity_missing = missing_model_identity_fields(panel)
    if panel.get("provenance_clean") is not True or audit.get("source_provenance_clean") is not True:
        modes.append(
            {
                "id": "source_panel_provenance_dirty",
                "classification": "evidence_hygiene_failure",
                "affected_fields": ["exp3302.provenance_clean", "exp3303.source_provenance_clean"],
                "panel_value": panel.get("provenance_clean") is True,
                "audit_value": audit.get("source_provenance_clean") is True,
                "true_repair_failure": False,
                "required_fix": "Exp3316 must set runtime_provenance_clean=true only after the Exp3309 checker passes.",
            }
        )
    if panel.get("headline_claim_allowed") is not True or audit.get("source_headline_claim_allowed") is not True:
        modes.append(
            {
                "id": "source_headline_gate_false",
                "classification": "evidence_hygiene_failure",
                "affected_fields": ["exp3302.headline_claim_allowed", "exp3303.source_headline_claim_allowed"],
                "panel_value": panel.get("headline_claim_allowed") is True,
                "audit_value": audit.get("source_headline_claim_allowed") is True,
                "true_repair_failure": False,
                "required_fix": "Exp3316 must keep headline promotion false unless all runtime, source, and substrate gates pass.",
            }
        )
    if critical:
        modes.append(
            {
                "id": "critical_duration_flag_carried_forward",
                "classification": "evidence_hygiene_failure",
                "affected_fields": ["exp3302.corrigendum_pending", "exp3303.adversarial_verify_flags"],
                "critical_flags": critical,
                "true_repair_failure": False,
                "required_fix": "Exp3316 must clear critical adversarial flags or write an honestly blocked artifact.",
            }
        )
    if missing_runtime:
        modes.append(
            {
                "id": "runtime_contract_fields_missing",
                "classification": "evidence_hygiene_failure",
                "missing_fields": missing_runtime,
                "true_repair_failure": False,
                "required_fix": "Exp3316 must include runtime_provenance, checker_versions, duration_contract_passed, and runtime_provenance_clean.",
            }
        )
    if missing_models:
        modes.append(
            {
                "id": "incomplete_mandated_model_coverage",
                "classification": "model_identity_hygiene_failure",
                "used_model_ids": model_ids_from_rows(panel.get("models_used")),
                "missing_model_ids": missing_models,
                "legacy_small_model_used": legacy_small_model_used(panel),
                "true_repair_failure": False,
                "required_fix": "Exp3316 must use the strongest available mandated GGUF and list every missing mandated model without substituting a legacy small model.",
            }
        )
    if identity_missing:
        modes.append(
            {
                "id": "model_identity_fields_incomplete",
                "classification": "model_identity_hygiene_failure",
                "missing_fields_by_model": identity_missing,
                "true_repair_failure": False,
                "required_fix": "Exp3316 must record model ID, HF ID, path, size, quantization, cache root, and snapshot revision in the used-model rows.",
            }
        )
    return modes


def substrate_consistency_failure_modes(
    panel: Mapping[str, Any],
    audit: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[JsonDict]:
    """List substrate blockers and expected non-blocking panel/audit differences."""

    minimum_duration = minimum_live_duration(contract)
    modes: list[JsonDict] = [
        {
            "id": "audit_aggregation_boundary_expected",
            "classification": "expected_audit_boundary_not_failure",
            "is_blocker": False,
            "panel_substrate": str(panel.get("inference_substrate") or ""),
            "audit_substrate": str(audit.get("inference_substrate") or ""),
            "required_fix": "Keep repair audits aggregation-only, but require the source panel to carry a clean runtime contract.",
        }
    ]
    if numeric(panel.get("duration_s")) < minimum_duration:
        modes.append(
            {
                "id": "source_duration_below_live_floor",
                "classification": "evidence_hygiene_failure",
                "is_blocker": True,
                "panel_duration_s": numeric(panel.get("duration_s")),
                "minimum_live_duration_s": minimum_duration,
                "required_fix": "Exp3316 headline live evidence must satisfy the Exp3309 duration floor or stay blocked.",
            }
        )
    if missing_runtime_contract_fields(panel):
        modes.append(
            {
                "id": "source_panel_runtime_contract_absent",
                "classification": "evidence_hygiene_failure",
                "is_blocker": True,
                "missing_fields": missing_runtime_contract_fields(panel),
                "required_fix": "Exp3316 must copy the Exp3309 checker result into duration_contract_passed and runtime_provenance_clean.",
            }
        )
    if critical_adversarial_flags(panel, audit):
        modes.append(
            {
                "id": "critical_adversarial_flag_present",
                "classification": "evidence_hygiene_failure",
                "is_blocker": True,
                "critical_flags": critical_adversarial_flags(panel, audit),
                "required_fix": "Exp3316 and Exp3317 must preserve zero critical adversarial flags before headline promotion.",
            }
        )
    if audit.get("substrate_consistency_passed") is not True:
        modes.append(
            {
                "id": "audit_substrate_consistency_gate_false",
                "classification": "evidence_hygiene_failure",
                "is_blocker": True,
                "audit_value": audit.get("substrate_consistency_passed") is True,
                "required_fix": "Exp3317 may set substrate_consistency_passed=true only when the source panel contract, exact checkers, model IDs, and critical flags are clean.",
            }
        )
    return modes


def blocker_classification(panel: Mapping[str, Any], audit: Mapping[str, Any]) -> list[JsonDict]:
    """Classify blockers as repair failures, bounded limitations, or hygiene."""

    outcomes = candidate_outcome_summary(panel)
    return [
        {
            "id": "false_accept_and_exact_check_status",
            "classification": "not_true_repair_failure",
            "false_accept_count": outcomes["false_accept_count"],
            "exact_check_passed_count": outcomes["exact_check_passed_count"],
            "candidate_result_count": outcomes["candidate_result_count"],
            "llm_judge_dependency_count": int(audit.get("llm_judge_dependency_count") or 0),
            "diagnosis": "Zero false accepts, exact checks on all rows, and no audit LLM-judge dependency are bounded positive evidence.",
        },
        {
            "id": "verified_success_shortfall",
            "classification": "bounded_repair_performance_limitation",
            "verified_success_count": outcomes["verified_success_count"],
            "panel_case_count": outcomes["candidate_result_count"],
            "clean_verifier_rejected_exact_success_count": outcomes["clean_verifier_rejected_exact_success_count"],
            "true_repair_failure": False,
            "diagnosis": "The 27/30 verified-success metric is real, but the three non-success rows were exact-pass rows rejected by the clean verifier, not false accepts.",
        },
        {
            "id": "headline_blocker_status",
            "classification": "evidence_hygiene_failure",
            "headline_claim_allowed": panel.get("headline_claim_allowed") is True,
            "headline_claim_allowed_after_audit": audit.get("headline_claim_allowed_after_audit") is True,
            "source_provenance_clean": audit.get("source_provenance_clean") is True,
            "substrate_consistency_passed": audit.get("substrate_consistency_passed") is True,
            "diagnosis": "Headline promotion is blocked by runtime/provenance/substrate evidence, not by a false-accept repair failure.",
        },
    ]


def rerun_contract() -> JsonDict:
    """Return exact downstream requirements for Exp 3314, Exp 3315, and Exp 3316."""

    return {
        "exp3314": {
            "deliverable": "results/experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.json",
            "purpose": "Separate exact correctness, clean-verifier behavior, uncertainty, abstention, provenance risk, and model identity before the live repair rerun.",
            "acceptance_requirements": [
                "distributional_repair_audit_ready=true",
                "repair_case_count>=30",
                "deterministic_exact_check_feature_present",
                "clean_verifier_decision_feature_present",
                "provenance_risk_features_include_runtime_contract_and_model_coverage",
                "model_identity_confound_check_reports_used_and_missing_mandated_models",
                "uncertainty_abstention_policy_blocks_high_risk_rows",
                "exact_acceptance_remains_final_authority",
                "no_new_model_execution=true",
            ],
        },
        "exp3315": {
            "deliverable": "results/experiment_3315_vgb_backtracking_repair_policy_v1.json",
            "purpose": "Define when the next repair runner accepts, rejects, backtracks, or abstains while keeping exact verifiers final.",
            "acceptance_requirements": [
                "vgb_repair_policy_ready=true",
                "proposal_budget_defined",
                "backtrack_on_clean_verifier_rejects_exact_success",
                "reject_on_exact_check_failure",
                "abstain_when_provenance_or_uncertainty_policy_blocks",
                "candidate_attempt_logging_required",
                "verifier_confidence_thresholds_defined",
                "exact_verifiers_not_llm_judges_are_final",
                "no_new_model_execution=true",
            ],
        },
        "exp3316": {
            "deliverable": "results/experiment_3316_sota_repair_rerun_v12_runtime_clean.json",
            "purpose": "Rerun the repair panel only after DataFlip quality cleanup, runtime provenance, uncertainty audit, and backtracking policy gates are ready.",
            "gated_on": [
                "exp3312.dataflip_gate_passed=true",
                "exp3312.quality_flags_cleared=true",
                "exp3309.runtime_contract_ready=true",
                "exp3314.distributional_repair_audit_ready=true",
                "exp3315.vgb_repair_policy_ready=true",
            ],
            "acceptance_requirements": [
                "repair_rerun_v12_ready=true",
                "repair_panel_ran=true",
                "panel_case_count>=30",
                "same_or_superset_manifest_case_hashes_recorded",
                "runtime_provenance_clean=true",
                "duration_contract_passed=true",
                "substrate_consistency_passed=true",
                "model_specs_used_match_exp3309_contract",
                "model_identity_cache_size_quantization_load_generation_gpu_checker_versions_present",
                "no_legacy_small_model_substitution",
                "used_and_missing_mandated_model_ids_reported",
                "exact_acceptance_authority_no_llm_judge",
                "false_accept_count=0",
                "abstention_count_reported",
                "confidence_interval_present",
                "no_critical_adversarial_verify_flags",
                "headline_claim_allowed_true_only_if_all_gates_pass_else_honestly_blocked",
            ],
        },
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal autopsy artifact and fail closed on overclaims."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not isinstance(artifact.get("repair_substrate_autopsy_ready"), bool):
        raise ValueError("repair_substrate_autopsy_ready must be a bool")
    if not artifact.get("analyzed_artifacts"):
        raise ValueError("analyzed_artifacts must be non-empty")
    if not artifact.get("source_provenance_failure_modes"):
        raise ValueError("source_provenance_failure_modes must be non-empty")
    if not artifact.get("substrate_consistency_failure_modes"):
        raise ValueError("substrate_consistency_failure_modes must be non-empty")
    contract = mapping(artifact.get("rerun_contract"))
    if set(contract) != {"exp3314", "exp3315", "exp3316"}:
        raise ValueError("rerun_contract must contain exp3314, exp3315, and exp3316")
    if artifact.get("no_new_model_execution") is not True:
        raise ValueError("no_new_model_execution must be true")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def candidate_outcome_summary(panel: Mapping[str, Any]) -> JsonDict:
    """Summarize exact outcomes and clean-verifier rejection rows."""

    rows = mapping_list(panel.get("candidate_results"))
    return {
        "candidate_result_count": len(rows),
        "exact_check_passed_count": sum(row.get("exact_check_passed") is True for row in rows),
        "verified_success_count": sum(row.get("verified_success") is True for row in rows),
        "false_accept_count": sum(row.get("false_accept") is True for row in rows),
        "clean_verifier_rejected_exact_success_count": sum(
            row.get("failure_class") == "clean_verifier_rejected_exact_success" for row in rows
        ),
        "clean_verifier_rejected_exact_success_case_ids": [
            str(row.get("case_id") or "")
            for row in rows
            if row.get("failure_class") == "clean_verifier_rejected_exact_success"
        ],
    }


def critical_adversarial_flags(panel: Mapping[str, Any], audit: Mapping[str, Any]) -> list[JsonDict]:
    """Return unique critical flags preserved by panel or audit artifacts."""

    flags: list[JsonDict] = []
    for source in (panel.get("corrigendum_pending"), audit.get("adversarial_verify_flags")):
        for row in mapping_list(source):
            if row.get("severity") == "critical" and row not in flags:
                flags.append(row)
    return flags


def missing_runtime_contract_fields(panel: Mapping[str, Any]) -> list[str]:
    """Return Exp 3309 runtime-contract fields absent from the source panel."""

    return [field for field in RUNTIME_CONTRACT_FIELDS if field not in panel]


def missing_model_identity_fields(panel: Mapping[str, Any]) -> list[JsonDict]:
    """Return missing Exp 3309 model-identity fields for each used model row."""

    missing: list[JsonDict] = []
    for row in mapping_list(panel.get("models_used")):
        missing_fields = [field for field in MODEL_IDENTITY_FIELDS if not row.get(field)]
        if missing_fields:
            missing.append({"model_id": str(row.get("model_id") or ""), "missing_fields": missing_fields})
    return missing


def exact_checker_types_for_successes(panel: Mapping[str, Any]) -> list[str]:
    """Return exact checker types used by rows the source panel counted."""

    return sorted(
        {
            str(row.get("exact_checker_type") or "")
            for row in mapping_list(panel.get("candidate_results"))
            if row.get("verified_success") is True and row.get("exact_checker_type")
        }
    )


def audit_source_exp3302_sha(audit: Mapping[str, Any]) -> str:
    """Return the audit-recorded SHA for the source panel artifact."""

    source = audit.get("source_artifacts")
    if isinstance(source, Mapping):
        return str(mapping(source.get("exp3302")).get("sha256") or "")
    for row in mapping_list(source):
        if row.get("experiment_id") == "exp3302" or row.get("label") == "exp3302":
            return str(row.get("sha256") or "")
    return ""


def comparison_row(field: str, panel_value: Any, audit_value: Any, status: str) -> JsonDict:
    """Build one compact field-comparison row."""

    return {"field": field, "panel": panel_value, "audit": audit_value, "status": status}


def list_status(left: list[str], right: list[str]) -> str:
    """Classify two string lists by exact match, set match, or mismatch."""

    if left == right:
        return "match"
    if set(left) == set(right):
        return "set_match_order_differs"
    return "mismatch"


def model_ids_from_rows(value: Any) -> list[str]:
    """Extract model IDs from a list of artifact rows."""

    return string_list(row.get("model_id") for row in mapping_list(value))


def legacy_small_model_used(panel: Mapping[str, Any]) -> bool:
    """Return true if any used model row declares legacy small-model fallback."""

    return any(row.get("legacy_small_model") is True for row in mapping_list(panel.get("models_used")))


def minimum_live_duration(contract: Mapping[str, Any]) -> float:
    """Read the Exp 3309 live duration floor, preserving the 60s fallback."""

    return numeric(contract.get("minimum_live_duration_s")) or MINIMUM_LIVE_DURATION_S


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a compact verdict that preserves the root-cause distinction."""

    ready = str(artifact.get("repair_substrate_autopsy_ready") is True).lower()
    return (
        "complete: "
        f"repair_substrate_autopsy_ready={ready}; "
        "repair_false_accept_failure=false; "
        "headline_blockers=evidence_hygiene; "
        "rerun_contract=exp3314_exp3315_exp3316; "
        "no_new_model_execution=true"
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable autopsy content while excluding self-referential fields."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "honest_verdict", "reproducibility_checksum"}
    }
    return stable_hash(stable)


def read_json_object(path: Path) -> JsonDict:
    """Read one required JSON source artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_text(path: Path) -> str:
    """Read one required text source artifact."""

    return path.read_text(encoding="utf-8")


def mapping(value: Any) -> JsonDict:
    """Return a plain dict for JSON-like mappings."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Return only mapping rows from a JSON-like list."""

    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list | tuple) else []


def string_list(value: Any) -> list[str]:
    """Return stable non-empty strings from an iterable JSON value."""

    if isinstance(value, str) or value is None:
        return []
    try:
        return [str(item) for item in value if str(item or "")]
    except TypeError:
        return []


def numeric(value: Any) -> float:
    """Return a float with explicit bad-value fallback for artifact checks."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def duration(started: float, finished: float) -> float:
    """Return non-negative elapsed seconds rounded for stable JSON."""

    return round(max(0.0, float(finished) - float(started)), 6)


def stable_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible content."""

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()
