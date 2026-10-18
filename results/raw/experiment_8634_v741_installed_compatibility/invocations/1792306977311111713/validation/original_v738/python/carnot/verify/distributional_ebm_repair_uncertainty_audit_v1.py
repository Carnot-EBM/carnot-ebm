"""Build the Exp 3314 distributional repair uncertainty audit artifact.

Spec refs: REQ-VERIFY-3314, SCENARIO-VERIFY-3314.

This module borrows the Distributional-EBM shape of "separate energy terms plus
uncertainty" without pretending that a new model has been trained. It replays
only checked-in repair rows, records deterministic proxy scores, and leaves the
exact checker result as the final correctness authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "carnot.distributional_ebm_repair_uncertainty_audit.v1"
EXPERIMENT_ID = "exp3314"
TASK_ID = "exp3314-distributional-ebm-repair-uncertainty-audit-v1"
ARTIFACT = "experiment_3314_distributional_ebm_repair_uncertainty_audit_v1"
MILESTONE = "2026.05.306"
RUN_DATE = "20260529"
RANDOM_SEED = 3314
INFERENCE_SUBSTRATE = "deterministic_artifact_replay_no_model_calls"

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.json")
EXP3302_REL_PATH = Path("results/experiment_3302_headline_sota_repair_panel_v11.json")
EXP3303_REL_PATH = Path("results/experiment_3303_repair_headline_evidence_audit_v1.json")
EXP3313_REL_PATH = Path("results/experiment_3313_repair_substrate_root_cause_autopsy_v1.json")

MIN_REPAIR_CASES = 30
UNCERTAINTY_SCORE_BLOCK_THRESHOLD = 0.60
PROVENANCE_RISK_SCORE_BLOCK_THRESHOLD = 0.50
MODEL_IDENTITY_COVERAGE_RISK_BLOCK_THRESHOLD = 0.50
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "distributional_repair_audit_ready",
    "uncertainty_abstention_policy",
    "distributional_energy_schema",
    "model_identity_confound_check",
    "provenance_risk_features",
    "repair_case_count",
    "no_new_model_execution",
    "honest_verdict",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """SCENARIO-VERIFY-3314: score repair rows without making new calls."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    panel = read_json_object(root_path / EXP3302_REL_PATH)
    audit = read_json_object(root_path / EXP3303_REL_PATH)
    autopsy = read_json_object(root_path / EXP3313_REL_PATH)
    rows = mapping_list(panel.get("candidate_results"))
    source_status = source_artifacts(root_path)
    provenance = provenance_risk_features(panel, audit, autopsy, source_status)
    model_check = model_identity_confound_check(panel, audit)
    row_scores = repair_row_scores(rows, provenance, model_check)
    schema = distributional_energy_schema()
    policy = uncertainty_abstention_policy(row_scores, provenance, model_check, bool(panel and audit and autopsy))
    exact_fields_present = bool(rows) and all("exact_check_passed" in row for row in rows)
    ready = (
        bool(panel and audit and autopsy)
        and len(row_scores) >= MIN_REPAIR_CASES
        and exact_fields_present
        and bool(schema)
        and bool(policy)
        and bool(model_check.get("used_model_ids") or model_check.get("missing_mandated_model_ids"))
        and provenance.get("runtime_contract_ready") is True
    )
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3314", "SCENARIO-VERIFY-3314"],
        "distributional_repair_audit_ready": ready,
        "uncertainty_abstention_policy": policy,
        "distributional_energy_schema": schema,
        "model_identity_confound_check": model_check,
        "provenance_risk_features": provenance,
        "repair_case_count": len(row_scores),
        "repair_row_scores": row_scores,
        "source_artifacts": source_status,
        "exact_acceptance_authority_preserved": True,
        "no_new_model_execution": True,
        "no_new_repair_generation": True,
        "no_new_verifier_run": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "duration_s": duration(started, finished),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3314 JSON deliverable."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def distributional_energy_schema() -> JsonDict:
    """Define the sidecar row schema and name which component has authority."""

    return {
        "schema_kind": "distributional_ebm_inspired_sidecar_not_acceptance_authority",
        "row_components": {
            "deterministic_constraint_penalty": {
                "source_fields": ["exact_check_passed", "exact_checker_type", "false_accept"],
                "formula": "0.0 when exact_check_passed is true, else 1.0",
                "authority": "final_exact_checker",
            },
            "learned_proxy_quality": {
                "source_fields": [
                    "calibrated_clean_verifier_decision",
                    "candidate_answer",
                    "token_counts",
                ],
                "formula": "deterministic proxy over recorded clean-verifier output and answer-shape evidence",
                "authority": "diagnostic_proxy_only",
            },
            "provenance_risk": {
                "source_fields": [
                    "provenance_clean",
                    "adversarial_verify_flags",
                    "runtime_contract_reference",
                    "substrate_consistency_failure_modes",
                ],
                "formula": "bounded max-risk score from source hygiene and runtime evidence",
                "authority": "headline_promotion_guard",
            },
            "model_identity": {
                "source_fields": ["models_used", "missing_model_specs", "model_invocation_summary"],
                "formula": "used and missing mandated model families are reported without imputation",
                "authority": "confound_diagnostic_only",
            },
            "uncertainty": {
                "source_fields": [
                    "exact_check_passed",
                    "calibrated_clean_verifier_decision",
                    "false_accept",
                    "candidate_answer",
                ],
                "formula": "high when exact checks and clean verifier disagree or the row lacks usable evidence",
                "authority": "triage_metadata_only",
            },
            "abstention": {
                "source_fields": ["uncertainty", "provenance_risk", "model_identity"],
                "formula": "block headline promotion when uncertainty or provenance/model risk is above threshold",
                "authority": "advisory_policy_only",
            },
        },
    }


def provenance_risk_features(
    panel: Mapping[str, Any],
    audit: Mapping[str, Any],
    autopsy: Mapping[str, Any],
    source_status: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Summarize source and runtime hygiene as a bounded risk vector."""

    artifacts_readable = all(mapping(row).get("readable") is True for row in source_status.values())
    flags = critical_flags(panel, audit, autopsy)
    runtime_ref = mapping(autopsy.get("runtime_contract_reference"))
    live_floor = numeric(runtime_ref.get("minimum_live_duration_s"))
    panel_duration = numeric(panel.get("duration_s"))
    duration_below_floor = bool(panel_duration and live_floor and panel_duration < live_floor)
    blocker_modes = [
        str(row.get("id") or "")
        for row in mapping_list(autopsy.get("substrate_consistency_failure_modes"))
        if row.get("is_blocker") is True
    ]
    score = 0.0
    if not artifacts_readable:
        score += 0.50
    if panel.get("provenance_clean") is not True or audit.get("source_provenance_clean") is not True:
        score += 0.25
    if audit.get("substrate_consistency_passed") is not True:
        score += 0.25
    if flags:
        score += 0.35
    if duration_below_floor:
        score += 0.25
    if runtime_ref.get("runtime_contract_ready") is not True:
        score += 0.25
    if blocker_modes:
        score += 0.20
    return {
        "source_artifacts_readable": artifacts_readable,
        "source_provenance_clean": panel.get("provenance_clean") is True
        and audit.get("source_provenance_clean") is True,
        "substrate_consistency_passed": audit.get("substrate_consistency_passed") is True,
        "critical_adversarial_flag_count": len(flags),
        "critical_adversarial_flags": flags,
        "runtime_contract_ready": runtime_ref.get("runtime_contract_ready") is True,
        "live_duration_floor_s": live_floor,
        "source_duration_s": panel_duration,
        "source_duration_below_live_floor": duration_below_floor,
        "substrate_blocker_modes": blocker_modes,
        "provenance_risk_score": round(min(1.0, score), 6),
    }


def critical_flags(panel: Mapping[str, Any], audit: Mapping[str, Any], autopsy: Mapping[str, Any]) -> list[JsonDict]:
    """Collect unique critical flags carried by the panel, audit, or autopsy."""

    flags: list[JsonDict] = []
    sources: list[Any] = [panel.get("corrigendum_pending"), audit.get("adversarial_verify_flags")]
    for row in mapping_list(autopsy.get("source_provenance_failure_modes")):
        sources.append(row.get("critical_flags"))
    for row in mapping_list(autopsy.get("substrate_consistency_failure_modes")):
        sources.append(row.get("critical_flags"))
    seen: set[tuple[str, str]] = set()
    for source in sources:
        for row in mapping_list(source):
            flag = {
                "kind": str(row.get("kind") or "UNKNOWN"),
                "severity": str(row.get("severity") or "warn"),
                "detail": str(row.get("detail") or ""),
            }
            key = (flag["kind"], flag["severity"])
            if flag["severity"] == "critical" and key not in seen:
                seen.add(key)
                flags.append(flag)
    return flags


def model_identity_confound_check(panel: Mapping[str, Any], audit: Mapping[str, Any]) -> JsonDict:
    """Report whether row outcomes are tied to a narrow model-family slice."""

    summary = mapping(audit.get("model_invocation_summary"))
    used_ids = string_list(summary.get("used_model_ids")) or model_ids_from_rows(panel.get("models_used"))
    missing_ids = string_list(summary.get("missing_model_ids")) or model_ids_from_rows(panel.get("missing_model_specs"))
    mandated_ids = string_list(summary.get("mandated_model_ids")) or string_list(
        mapping(panel.get("model_specs")).get("mandated_model_ids")
    )
    used_families = sorted({model_family(model_id) for model_id in used_ids})
    missing_families = sorted({model_family(model_id) for model_id in missing_ids})
    mandated_families = sorted({model_family(model_id) for model_id in mandated_ids})
    missing_fraction = rate(len(missing_ids), len(mandated_ids)) if mandated_ids else 1.0
    single_family_slice = bool(used_families) and len(used_families) <= 1 and len(mandated_families) > 1
    coverage_risk = 1.0 if not used_ids else missing_fraction
    if single_family_slice:
        coverage_risk = max(coverage_risk, 0.50)
    confound_detected = coverage_risk >= MODEL_IDENTITY_COVERAGE_RISK_BLOCK_THRESHOLD
    return {
        "confound_detected": confound_detected,
        "used_model_ids": used_ids,
        "missing_mandated_model_ids": missing_ids,
        "mandated_model_ids": mandated_ids,
        "used_model_families": used_families,
        "missing_model_families": missing_families,
        "mandated_model_families": mandated_families,
        "single_used_family_slice": single_family_slice,
        "model_identity_coverage_risk": round(min(1.0, coverage_risk), 6),
        "model_family_confounds": model_family_confounds(used_ids, missing_ids, used_families),
        "exp3316_report_required_fields": [
            "model_id",
            "hf_id",
            "model_path",
            "size_bytes",
            "quantization",
            "cache_root",
            "snapshot_revision",
            "model_family",
            "role",
        ],
    }


def model_family_confounds(
    used_ids: Sequence[str],
    missing_ids: Sequence[str],
    used_families: Sequence[str],
) -> list[str]:
    """Turn model coverage gaps into reportable warnings for Exp 3316."""

    messages: list[str] = []
    if used_ids and len(set(used_families)) <= 1 and missing_ids:
        messages.append(
            "All scored repair rows came from one used model family; Exp3316 must not report cross-family repair evidence from this audit."
        )
    if missing_ids:
        messages.append(
            "Missing mandated model ids must be reported beside any headline repair metric: "
            + ", ".join(missing_ids)
        )
    if not used_ids:
        messages.append("No used model id is recorded, so model identity risk is maximal.")
    return messages


def repair_row_scores(
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    model_check: Mapping[str, Any],
) -> list[JsonDict]:
    """Apply the sidecar schema to every available repair row."""

    return [repair_row_score(row, provenance, model_check) for row in rows]


def repair_row_score(
    row: Mapping[str, Any],
    provenance: Mapping[str, Any],
    model_check: Mapping[str, Any],
) -> JsonDict:
    """Compute one deterministic row score while preserving exact authority."""

    exact_passed = row.get("exact_check_passed") is True
    false_accept = row.get("false_accept") is True
    decision = str(row.get("calibrated_clean_verifier_decision") or row.get("calibrated_clean_verifier_output") or "")
    clean_quality = score_clean_verifier_quality(decision)
    answer_present = bool(str(row.get("candidate_answer") or "").strip())
    token_counts = mapping(row.get("token_counts"))
    token_count_present = numeric(token_counts.get("total_tokens")) > 0
    deterministic_penalty = 0.0 if exact_passed else 1.0
    learned_quality = (clean_quality + (1.0 if answer_present else 0.0) + (1.0 if token_count_present else 0.0)) / 3.0
    exact_clean_disagreement = exact_passed and decision.casefold() in {"reject", "abstain"}
    unknown_clean_decision = decision.casefold() not in {"accept", "reject", "abstain"}
    uncertainty_score = 0.10
    if not exact_passed or false_accept:
        uncertainty_score = 1.0
    elif exact_clean_disagreement:
        uncertainty_score = 0.70
    elif unknown_clean_decision or not answer_present:
        uncertainty_score = 0.65
    reason_codes: list[str] = []
    if uncertainty_score >= UNCERTAINTY_SCORE_BLOCK_THRESHOLD:
        if exact_clean_disagreement:
            reason_codes.append("exact_clean_disagreement")
        elif not exact_passed:
            reason_codes.append("deterministic_constraint_failure")
        elif unknown_clean_decision:
            reason_codes.append("unknown_clean_verifier_decision")
        else:
            reason_codes.append("row_uncertainty")
    if numeric(provenance.get("provenance_risk_score")) >= PROVENANCE_RISK_SCORE_BLOCK_THRESHOLD:
        reason_codes.append("source_provenance_risk")
    if numeric(model_check.get("model_identity_coverage_risk")) >= MODEL_IDENTITY_COVERAGE_RISK_BLOCK_THRESHOLD:
        reason_codes.append("model_identity_coverage_risk")
    return {
        "case_id": str(row.get("case_id") or ""),
        "case_hash": str(row.get("case_hash") or ""),
        "family": str(row.get("family") or ""),
        "deterministic_constraint_penalty": deterministic_penalty,
        "learned_proxy_quality_score": round(learned_quality, 6),
        "provenance_risk_score": numeric(provenance.get("provenance_risk_score")),
        "model_identity": {
            "model_id": str(row.get("model_id") or ""),
            "model_family": model_family(str(row.get("model_id") or "")),
        },
        "uncertainty": {
            "exact_clean_disagreement": exact_clean_disagreement,
            "unknown_clean_verifier_decision": unknown_clean_decision,
            "answer_present": answer_present,
            "token_count_present": token_count_present,
        },
        "uncertainty_score": round(uncertainty_score, 6),
        "abstention": {
            "policy_blocked": bool(reason_codes),
            "reason_codes": reason_codes,
        },
        "exact_check_passed": exact_passed,
        "exact_checker_type": str(row.get("exact_checker_type") or ""),
        "calibrated_clean_verifier_decision": decision.casefold(),
        "verified_success": row.get("verified_success") is True,
        "false_accept": false_accept,
        "exact_acceptance_authority": exact_passed,
    }


def uncertainty_abstention_policy(
    row_scores: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    model_check: Mapping[str, Any],
    sources_readable: bool,
) -> JsonDict:
    """Define the Exp 3316 headline block policy from sidecar metadata."""

    high_uncertainty_rows = [
        str(row.get("case_id") or "")
        for row in row_scores
        if numeric(row.get("uncertainty_score")) >= UNCERTAINTY_SCORE_BLOCK_THRESHOLD
    ]
    row_abstentions = [row for row in row_scores if mapping(row.get("abstention")).get("policy_blocked") is True]
    provenance_blocks = (
        numeric(provenance.get("provenance_risk_score")) >= PROVENANCE_RISK_SCORE_BLOCK_THRESHOLD
        or count_value(provenance.get("critical_adversarial_flag_count")) > 0
    )
    model_blocks = numeric(model_check.get("model_identity_coverage_risk")) >= MODEL_IDENTITY_COVERAGE_RISK_BLOCK_THRESHOLD
    headline_blocked = bool(high_uncertainty_rows or provenance_blocks or model_blocks or not sources_readable)
    return {
        "policy_name": "exp3316_headline_promotion_abstention_policy_v1",
        "uncertainty_score_block_threshold": UNCERTAINTY_SCORE_BLOCK_THRESHOLD,
        "provenance_risk_score_block_threshold": PROVENANCE_RISK_SCORE_BLOCK_THRESHOLD,
        "model_identity_coverage_risk_block_threshold": MODEL_IDENTITY_COVERAGE_RISK_BLOCK_THRESHOLD,
        "high_uncertainty_case_count": len(high_uncertainty_rows),
        "high_uncertainty_case_ids": high_uncertainty_rows,
        "row_abstention_count": len(row_abstentions),
        "provenance_risk_blocks_headline": provenance_blocks,
        "model_identity_risk_blocks_headline": model_blocks,
        "headline_promotion_blocked": headline_blocked,
        "exact_acceptance_remains_final_authority": True,
        "exp3316_required_fields": [
            "uncertainty_abstention_policy",
            "distributional_energy_schema",
            "model_identity_confound_check",
            "provenance_risk_features",
            "row_abstention_count",
            "critical_adversarial_flag_count",
        ],
        "exp3316_reporting_rule": (
            "Exp3316 may promote a repair headline only when exact checks pass, false_accept_count=0, "
            "critical flags are zero, provenance/model risk are below threshold, and no row exceeds the uncertainty threshold."
        ),
    }


def score_clean_verifier_quality(decision: str) -> float:
    """Map the recorded clean-verifier decision to a proxy quality score."""

    normalized = decision.casefold()
    if normalized == "accept":
        return 1.0
    if normalized == "abstain":
        return 0.5
    if normalized == "reject":
        return 0.25
    return 0.0


def source_artifacts(root: Path) -> JsonDict:
    """Return source file status rows and hashes for reproducibility."""

    return {
        "exp3302": file_status(root / EXP3302_REL_PATH),
        "exp3303": file_status(root / EXP3303_REL_PATH),
        "exp3313": file_status(root / EXP3313_REL_PATH),
    }


def file_status(path: Path) -> JsonDict:
    """Inspect a source file without interpreting its scientific meaning."""

    if not path.is_file():
        return {"path": str(path), "present": path.exists(), "readable": False, "sha256": None}
    return {"path": str(path), "present": True, "readable": True, "sha256": sha256_file(path)}


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, returning empty evidence on malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def model_ids_from_rows(rows: Any) -> list[str]:
    """Extract model ids from dictionaries while preserving source order."""

    ids: list[str] = []
    for row in mapping_list(rows):
        value = str(row.get("model_id") or row.get("hf_id") or "")
        if value and value not in ids:
            ids.append(value)
    return ids


def model_family(model_id: str) -> str:
    """Classify a model id into the coarse family needed for confound reports."""

    normalized = model_id.casefold()
    if "qwen" in normalized:
        return "qwen"
    if "gemma" in normalized:
        return "gemma"
    if "llama" in normalized:
        return "llama"
    return "unknown"


def mapping(value: Any) -> JsonDict:
    """Normalize a maybe-dict value to a plain dictionary."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Normalize a list of dict-like values to JSON dictionaries."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def string_list(value: Any) -> list[str]:
    """Normalize a sequence to strings while dropping non-sequence scalars."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value]


def numeric(value: Any) -> float:
    """Convert JSON scalar values to finite floats when possible."""

    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def count_value(value: Any) -> int:
    """Convert JSON scalar counts to integers while treating bools as invalid."""

    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate, avoiding divide-by-zero promotion bugs."""

    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def duration(started_s: float, finished_s: float) -> float:
    """Compute non-negative elapsed seconds for deterministic tests."""

    return round(max(0.0, finished_s - started_s), 6)


def sha256_file(path: Path) -> str | None:
    """Hash a source file, returning None when it is unavailable."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash artifact content while excluding volatile runtime fields."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Report the audit outcome without promoting the sidecar to authority."""

    policy = mapping(artifact.get("uncertainty_abstention_policy"))
    if artifact.get("distributional_repair_audit_ready") is True and policy.get("headline_promotion_blocked") is True:
        return "complete: distributional repair audit ready; Exp3316 headline promotion blocked by advisory risk policy"
    if artifact.get("distributional_repair_audit_ready") is True:
        return "complete: distributional repair audit ready; no advisory abstention block found"
    return "complete: distributional repair audit incomplete; Exp3316 headline promotion remains blocked"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal artifact and fail closed on overclaims."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not isinstance(artifact.get("distributional_repair_audit_ready"), bool):
        raise ValueError("distributional_repair_audit_ready must be a bool")
    if not mapping(artifact.get("uncertainty_abstention_policy")):
        raise ValueError("uncertainty_abstention_policy must be non-empty")
    if not mapping(artifact.get("distributional_energy_schema")):
        raise ValueError("distributional_energy_schema must be non-empty")
    if not mapping(artifact.get("model_identity_confound_check")):
        raise ValueError("model_identity_confound_check must be non-empty")
    if not mapping(artifact.get("provenance_risk_features")):
        raise ValueError("provenance_risk_features must be non-empty")
    if not isinstance(artifact.get("repair_case_count"), int):
        raise ValueError("repair_case_count must be an int")
    if artifact.get("no_new_model_execution") is not True:
        raise ValueError("no_new_model_execution must be true")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")
