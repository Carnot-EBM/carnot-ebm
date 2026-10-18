"""Build the Exp 3315 verifier-guided backtracking repair policy artifact.

Spec refs: REQ-VERIFY-3315, SCENARIO-VERIFY-3315.

The policy is executable, but it is not a new model run. It turns the Exp 3314
uncertainty audit into a deterministic routing contract for Exp 3316: exact
checks decide whether a candidate is correct, while process-verifier confidence
only decides whether to keep proposing, backtrack, or abstain.
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
SCHEMA_VERSION = "carnot.vgb_backtracking_repair_policy.v1"
EXPERIMENT_ID = "exp3315"
TASK_ID = "exp3315-vgb-backtracking-repair-policy-v1"
ARTIFACT = "experiment_3315_vgb_backtracking_repair_policy_v1"
MILESTONE = "2026.05.306"
RUN_DATE = "20260529"
RANDOM_SEED = 3315
INFERENCE_SUBSTRATE = "deterministic_policy_artifact_no_model_calls"

SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
OUTPUT_REL_PATH = Path("results/experiment_3315_vgb_backtracking_repair_policy_v1.json")
EXP3313_REL_PATH = Path("results/experiment_3313_repair_substrate_root_cause_autopsy_v1.json")
EXP3314_REL_PATH = Path("results/experiment_3314_distributional_ebm_repair_uncertainty_audit_v1.json")

MAX_ATTEMPTS_PER_CASE = 4
MAX_BACKTRACKS_PER_CASE = MAX_ATTEMPTS_PER_CASE - 1
PROCESS_ACCEPT_CONFIDENCE_MIN = 0.80
PROCESS_BACKTRACK_CONFIDENCE_FLOOR = 0.60
ROW_UNCERTAINTY_ABSTAIN_THRESHOLD = 0.60
PROVENANCE_RISK_ABSTAIN_THRESHOLD = 0.50
MODEL_IDENTITY_RISK_ABSTAIN_THRESHOLD = 0.50
SUCCESS_PREFIXES = ("complete:", "success:", "passed:", "shipped:")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "vgb_repair_policy_ready",
    "backtracking_policy",
    "proposal_budget",
    "exact_acceptance_rules",
    "verifier_confidence_thresholds",
    "no_new_model_execution",
    "honest_verdict",
)

REQUIRED_ATTEMPT_LOG_FIELDS: tuple[str, ...] = (
    "case_id",
    "case_hash",
    "attempt_index",
    "proposal_id",
    "parent_attempt_id",
    "backtrack_depth",
    "candidate_hash",
    "model_id",
    "model_family",
    "prompt_hash",
    "exact_check_passed",
    "exact_checker_type",
    "exact_outcome",
    "false_accept",
    "clean_process_verifier_decision",
    "process_verifier_confidence",
    "verifier_confidence_thresholds",
    "policy_action",
    "action_reason_codes",
    "abstained",
    "abstention_reason_codes",
    "exact_acceptance_authority",
    "no_llm_judge_final_acceptance",
    "source_artifact_hashes",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """SCENARIO-VERIFY-3315: build the policy without proposing new repairs."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    autopsy = read_json_object(root_path / EXP3313_REL_PATH)
    audit = read_json_object(root_path / EXP3314_REL_PATH)
    source_status = source_artifacts(root_path)
    summary = source_policy_summary(autopsy, audit, source_status)
    thresholds = default_verifier_confidence_thresholds(audit)
    budget = proposal_budget(count_value(audit.get("repair_case_count")))
    exact_rules = exact_acceptance_rules()
    policy = backtracking_policy()
    handoff = exp3316_handoff(summary, thresholds)
    sample_routing = sample_candidate_routing(audit, thresholds, summary)
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3315", "SCENARIO-VERIFY-3315"],
        "vgb_repair_policy_ready": policy_ready(summary, budget, exact_rules, policy),
        "backtracking_policy": policy,
        "proposal_budget": budget,
        "exact_acceptance_rules": exact_rules,
        "verifier_confidence_thresholds": thresholds,
        "source_policy_summary": summary,
        "candidate_attempt_logging_requirements": policy["candidate_attempt_logging"],
        "sample_candidate_routing": sample_routing,
        "exp3316_handoff": handoff,
        "source_artifacts": source_status,
        "no_new_model_execution": True,
        "no_new_repair_generation": True,
        "no_new_verifier_run": True,
        "no_llm_judge_final_acceptance": True,
        "scripts_research_conductor_modified": False,
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
    """Build, validate, and persist the Exp 3315 terminal artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def default_verifier_confidence_thresholds(audit: Mapping[str, Any]) -> JsonDict:
    """Carry forward Exp 3314 risk thresholds and add the VGB process thresholds."""

    policy = mapping(audit.get("uncertainty_abstention_policy"))
    return {
        "exact_acceptance_required": 1.0,
        "process_accept_confidence_min": PROCESS_ACCEPT_CONFIDENCE_MIN,
        "process_backtrack_confidence_floor": PROCESS_BACKTRACK_CONFIDENCE_FLOOR,
        "row_uncertainty_abstain_threshold": numeric(
            policy.get("uncertainty_score_block_threshold"), ROW_UNCERTAINTY_ABSTAIN_THRESHOLD
        ),
        "provenance_risk_abstain_threshold": numeric(
            policy.get("provenance_risk_score_block_threshold"), PROVENANCE_RISK_ABSTAIN_THRESHOLD
        ),
        "model_identity_coverage_risk_abstain_threshold": numeric(
            policy.get("model_identity_coverage_risk_block_threshold"),
            MODEL_IDENTITY_RISK_ABSTAIN_THRESHOLD,
        ),
        "critical_adversarial_flag_count_max": 0,
        "llm_judge_acceptance_confidence": "not_applicable_llm_judges_are_never_final",
    }


def exact_acceptance_rules() -> JsonDict:
    """Define the non-negotiable final acceptance gate for a candidate."""

    return {
        "final_acceptance_authority": "exact_verifier_only",
        "llm_judge_final_acceptance_allowed": False,
        "clean_process_verifier_final_acceptance_allowed": False,
        "required_acceptance_conditions": [
            "exact_check_passed=true",
            "exact_checker_type_present",
            "false_accept=false",
            "candidate_answer_present",
            "candidate_attempt_logged",
        ],
        "acceptance_disallowed_when": [
            "exact_check_passed=false",
            "exact_checker_type_missing",
            "false_accept=true",
            "llm_judge_is_only_positive_signal",
            "source_provenance_or_uncertainty_policy_blocks",
        ],
        "authority_note": (
            "Process verifiers may route search, but only an exact checker can "
            "turn a candidate into an accepted repair."
        ),
    }


def backtracking_policy() -> JsonDict:
    """Return the executable accept/reject/backtrack/abstain action contract."""

    return {
        "policy_name": "exp3315_verifier_guided_backtracking_repair_policy_v1",
        "actions": {
            "accepted": {
                "requires_exact_acceptance": True,
                "trigger_conditions": [
                    "exact_check_passed=true",
                    "false_accept=false",
                    "exact_checker_type_present",
                    "clean_process_verifier_decision=accept",
                    "process_verifier_confidence>=process_accept_confidence_min",
                    "row_uncertainty<row_uncertainty_abstain_threshold",
                    "no_advisory_risk_gate_closed",
                ],
            },
            "rejected": {
                "requires_exact_acceptance": False,
                "trigger_conditions": [
                    "exact_check_passed=false",
                    "false_accept=true",
                ],
                "next_step": "backtrack_if_budget_remains_else_abstain_case",
            },
            "backtracked": {
                "requires_exact_acceptance": False,
                "trigger_conditions": [
                    "clean_process_verifier_rejects_exact_success",
                    "clean_process_verifier_abstains_on_exact_success",
                    "process_verifier_confidence_below_accept_threshold",
                    "row_uncertainty_at_or_above_threshold",
                ],
                "next_step": "request_new_candidate_with_localized_feedback_until_budget_exhausted",
            },
            "abstained": {
                "requires_exact_acceptance": False,
                "trigger_conditions": [
                    "source_provenance_or_uncertainty_policy_blocks",
                    "model_identity_coverage_risk_blocks",
                    "critical_adversarial_flag_present",
                    "exact_checker_type_missing",
                    "proposal_budget_exhausted",
                    "upstream_policy_artifact_not_ready",
                ],
                "next_step": "do_not_promote_headline_claim",
            },
        },
        "candidate_attempt_logging": {
            "required_fields": list(REQUIRED_ATTEMPT_LOG_FIELDS),
            "log_each_candidate_before_route_decision": True,
            "log_exact_outcome_even_when_abstained": True,
            "log_verifier_confidence_even_when_exact_checker_passes": True,
            "log_abstention_reason_codes": True,
        },
    }


def proposal_budget(repair_case_count: int) -> JsonDict:
    """Define a finite proposal budget so search cannot hide weak reasoning."""

    case_count = max(0, repair_case_count)
    return {
        "max_attempts_per_case": MAX_ATTEMPTS_PER_CASE,
        "max_backtracks_per_case": MAX_BACKTRACKS_PER_CASE,
        "max_total_attempts": case_count * MAX_ATTEMPTS_PER_CASE,
        "budget_scope": "per_manifest_case_plus_global_panel_cap",
        "stop_conditions": [
            "stop_on_exact_acceptance",
            "stop_on_proposal_budget_exhausted",
            "stop_on_false_accept_hard_failure",
            "stop_on_advisory_gate_closed_before_generation",
            "stop_on_missing_exact_checker",
            "stop_on_global_attempt_cap",
        ],
    }


def route_candidate_attempt(
    attempt: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
    max_attempts_per_case: int,
) -> JsonDict:
    """Route one candidate attempt using exact outcomes first and confidence second."""

    attempt_index = max(1, count_value(attempt.get("attempt_index")) or count_value(attempt.get("attempt_number")) or 1)
    attempts_remaining = attempt_index < max_attempts_per_case
    exact_passed = attempt.get("exact_check_passed") is True
    false_accept = attempt.get("false_accept") is True
    exact_type = str(attempt.get("exact_checker_type") or "").strip()
    decision = str(
        attempt.get("calibrated_clean_verifier_decision")
        or attempt.get("clean_process_verifier_decision")
        or attempt.get("calibrated_clean_verifier_output")
        or ""
    ).casefold()
    confidence = numeric(
        attempt.get("process_verifier_confidence", attempt.get("verifier_confidence")),
        0.0,
    )
    uncertainty = numeric(attempt.get("uncertainty_score"), 0.0)
    advisory_blocked = advisory_risk_gate_closed(attempt, thresholds)
    base = {
        "attempt_index": attempt_index,
        "attempts_remaining": max(0, max_attempts_per_case - attempt_index),
        "exact_check_passed": exact_passed,
        "exact_checker_type": exact_type,
        "false_accept": false_accept,
        "clean_process_verifier_decision": decision,
        "process_verifier_confidence": round(confidence, 6),
        "uncertainty_score": round(uncertainty, 6),
        "exact_acceptance_authority": exact_passed and bool(exact_type) and not false_accept,
        "no_llm_judge_final_acceptance": True,
    }
    if not exact_passed:
        return routed(base, "rejected", ["deterministic_exact_check_failed"], attempts_remaining)
    if false_accept:
        return routed(base, "rejected", ["false_accept_recorded"], attempts_remaining)
    if not exact_type:
        return routed(base, "abstained", ["missing_exact_checker_type"], False)
    if advisory_blocked:
        return routed(base, "abstained", ["advisory_risk_gate_closed"], False)
    reasons: list[str] = []
    if decision == "reject":
        reasons.append("clean_process_verifier_rejects_exact_success")
    elif decision == "abstain":
        reasons.append("clean_process_verifier_abstains_on_exact_success")
    elif decision != "accept":
        reasons.append("unknown_clean_process_verifier_decision")
    if confidence < numeric(thresholds.get("process_accept_confidence_min"), PROCESS_ACCEPT_CONFIDENCE_MIN):
        reasons.append("process_verifier_confidence_below_accept_threshold")
    if uncertainty >= numeric(thresholds.get("row_uncertainty_abstain_threshold"), ROW_UNCERTAINTY_ABSTAIN_THRESHOLD):
        reasons.append("row_uncertainty_at_or_above_threshold")
    if not reasons:
        return routed(base, "accepted", ["exact_acceptance_confirmed"], False)
    if attempts_remaining:
        return routed(base, "backtracked", reasons, True)
    return routed(base, "abstained", [*reasons, "proposal_budget_exhausted"], False)


def advisory_risk_gate_closed(attempt: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    """Check advisory risk gates that must abstain instead of headline-accepting."""

    return (
        attempt.get("advisory_policy_blocked") is True
        or numeric(attempt.get("provenance_risk_score"), 0.0)
        >= numeric(thresholds.get("provenance_risk_abstain_threshold"), PROVENANCE_RISK_ABSTAIN_THRESHOLD)
        or numeric(attempt.get("model_identity_coverage_risk"), 0.0)
        >= numeric(
            thresholds.get("model_identity_coverage_risk_abstain_threshold"),
            MODEL_IDENTITY_RISK_ABSTAIN_THRESHOLD,
        )
        or count_value(attempt.get("critical_adversarial_flag_count"))
        > count_value(thresholds.get("critical_adversarial_flag_count_max"))
    )


def routed(base: Mapping[str, Any], action: str, reasons: Sequence[str], backtrack_next: bool) -> JsonDict:
    """Attach one policy decision to the common attempt fields."""

    payload = dict(base)
    payload.update(
        {
            "policy_action": action,
            "reason_codes": list(reasons),
            "action_reason_codes": list(reasons),
            "backtrack_next": backtrack_next,
            "abstained": action == "abstained",
            "abstention_reason_codes": list(reasons) if action == "abstained" else [],
        }
    )
    return payload


def source_policy_summary(
    autopsy: Mapping[str, Any],
    audit: Mapping[str, Any],
    source_status: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Summarize the upstream gates that decide whether the policy is usable."""

    policy = mapping(audit.get("uncertainty_abstention_policy"))
    provenance = mapping(audit.get("provenance_risk_features"))
    model = mapping(audit.get("model_identity_confound_check"))
    sources_readable = all(mapping(row).get("readable") is True for row in source_status.values())
    return {
        "source_artifacts_readable": sources_readable,
        "exp3313_ready": autopsy.get("repair_substrate_autopsy_ready") is True,
        "exp3314_ready": audit.get("distributional_repair_audit_ready") is True,
        "exp3314_exact_acceptance_authority_preserved": audit.get("exact_acceptance_authority_preserved") is True
        or policy.get("exact_acceptance_remains_final_authority") is True,
        "repair_case_count": count_value(audit.get("repair_case_count")),
        "headline_promotion_blocked_by_exp3314": policy.get("headline_promotion_blocked") is True,
        "high_uncertainty_case_count": count_value(policy.get("high_uncertainty_case_count")),
        "row_abstention_count": count_value(policy.get("row_abstention_count")),
        "provenance_risk_score": numeric(provenance.get("provenance_risk_score"), 1.0),
        "critical_adversarial_flag_count": count_value(provenance.get("critical_adversarial_flag_count")),
        "model_identity_coverage_risk": numeric(model.get("model_identity_coverage_risk"), 1.0),
        "missing_mandated_model_ids": string_list(model.get("missing_mandated_model_ids")),
    }


def exp3316_handoff(summary: Mapping[str, Any], thresholds: Mapping[str, Any]) -> JsonDict:
    """Define the fields Exp 3316 must log when it consumes this policy."""

    blocked = (
        summary.get("headline_promotion_blocked_by_exp3314") is True
        or summary.get("exp3314_ready") is not True
        or numeric(summary.get("provenance_risk_score"), 1.0)
        >= numeric(thresholds.get("provenance_risk_abstain_threshold"), PROVENANCE_RISK_ABSTAIN_THRESHOLD)
        or numeric(summary.get("model_identity_coverage_risk"), 1.0)
        >= numeric(
            thresholds.get("model_identity_coverage_risk_abstain_threshold"),
            MODEL_IDENTITY_RISK_ABSTAIN_THRESHOLD,
        )
        or count_value(summary.get("critical_adversarial_flag_count")) > 0
    )
    return {
        "headline_promotion_blocked_until_policy_clears": blocked,
        "required_artifact_fields": [
            "vgb_repair_policy_ready",
            "candidate_attempts",
            "proposal_budget",
            "exact_acceptance_rules",
            "verifier_confidence_thresholds",
            "exact_outcome_summary",
            "abstention_count",
            "abstention_reason_codes",
        ],
        "attempt_logging_rule": (
            "Every candidate attempt must record verifier confidence and exact "
            "outcome before the policy action is applied."
        ),
        "acceptance_rule": (
            "Exp3316 may accept a candidate only through exact verifier success; "
            "process verifiers guide backtracking and abstention only."
        ),
    }


def sample_candidate_routing(
    audit: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> list[JsonDict]:
    """Route a small source-row sample to prove the policy is executable."""

    rows = mapping_list(audit.get("repair_row_scores"))
    if not rows:
        return []
    routed_rows: list[JsonDict] = []
    advisory_blocked = summary.get("headline_promotion_blocked_by_exp3314") is True
    for row in rows[:5]:
        decision = str(row.get("calibrated_clean_verifier_decision") or "")
        attempt = {
            "attempt_index": 1,
            "exact_check_passed": row.get("exact_check_passed") is True,
            "exact_checker_type": str(row.get("exact_checker_type") or ""),
            "false_accept": row.get("false_accept") is True,
            "calibrated_clean_verifier_decision": decision,
            "process_verifier_confidence": inferred_process_confidence(decision),
            "uncertainty_score": numeric(row.get("uncertainty_score"), 0.0),
            "advisory_policy_blocked": advisory_blocked,
        }
        route = route_candidate_attempt(
            attempt,
            thresholds=thresholds,
            max_attempts_per_case=MAX_ATTEMPTS_PER_CASE,
        )
        route["case_id"] = str(row.get("case_id") or "")
        route["case_hash"] = str(row.get("case_hash") or "")
        routed_rows.append(route)
    return routed_rows


def inferred_process_confidence(decision: str) -> float:
    """Use a deterministic proxy when only the clean-verifier decision is logged."""

    normalized = decision.casefold()
    if normalized in {"accept", "reject"}:
        return 0.90
    if normalized == "abstain":
        return 0.50
    return 0.0


def policy_ready(
    summary: Mapping[str, Any],
    budget: Mapping[str, Any],
    exact_rules: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    """Fail closed unless the policy and upstream evidence are complete enough."""

    actions = set(mapping(policy.get("actions")))
    logged_fields = set(string_list(mapping(policy.get("candidate_attempt_logging")).get("required_fields")))
    return (
        summary.get("source_artifacts_readable") is True
        and summary.get("exp3313_ready") is True
        and summary.get("exp3314_ready") is True
        and summary.get("exp3314_exact_acceptance_authority_preserved") is True
        and count_value(summary.get("repair_case_count")) >= 30
        and count_value(budget.get("max_attempts_per_case")) > 0
        and count_value(budget.get("max_total_attempts")) > 0
        and actions == {"accepted", "rejected", "backtracked", "abstained"}
        and set(REQUIRED_ATTEMPT_LOG_FIELDS) <= logged_fields
        and exact_rules.get("final_acceptance_authority") == "exact_verifier_only"
        and exact_rules.get("llm_judge_final_acceptance_allowed") is False
    )


def source_artifacts(root: Path) -> JsonDict:
    """Return source file status rows and hashes for reproducibility."""

    return {
        "exp3313": file_status(root / EXP3313_REL_PATH),
        "exp3314": file_status(root / EXP3314_REL_PATH),
    }


def file_status(path: Path) -> JsonDict:
    """Inspect a source artifact without treating presence as correctness."""

    if not path.is_file():
        return {"path": str(path), "present": path.exists(), "readable": False, "sha256": None}
    return {"path": str(path), "present": True, "readable": True, "sha256": sha256_file(path)}


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, returning empty evidence on missing or malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def mapping(value: Any) -> JsonDict:
    """Normalize maybe-dict data to a mutable JSON dictionary."""

    return dict(value) if isinstance(value, Mapping) else {}


def mapping_list(value: Any) -> list[JsonDict]:
    """Normalize maybe-list data to a list of JSON dictionaries."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def string_list(value: Any) -> list[str]:
    """Normalize a JSON sequence to strings while dropping scalar inputs."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value]


def numeric(value: Any, default: float = 0.0) -> float:
    """Convert JSON scalar values to finite floats while treating bools as invalid."""

    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def count_value(value: Any) -> int:
    """Convert JSON scalar counts to integers while treating bools as invalid."""

    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate without letting divide-by-zero imply success."""

    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def duration(started_s: float, finished_s: float) -> float:
    """Compute non-negative elapsed seconds for deterministic tests."""

    return round(max(0.0, finished_s - started_s), 6)


def sha256_file(path: Path) -> str | None:
    """Hash an input artifact, returning None when the file is unavailable."""

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
    """Report policy readiness without turning advisory signals into authority."""

    handoff = mapping(artifact.get("exp3316_handoff"))
    if artifact.get("vgb_repair_policy_ready") is True and handoff.get("headline_promotion_blocked_until_policy_clears"):
        return "complete: vgb repair policy ready; Exp3316 must abstain until advisory gates clear"
    if artifact.get("vgb_repair_policy_ready") is True:
        return "complete: vgb repair policy ready; no Exp3315 handoff block remains"
    return "complete: vgb repair policy incomplete; Exp3316 headline promotion remains blocked"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal policy artifact and fail closed on overclaims."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if not isinstance(artifact.get("vgb_repair_policy_ready"), bool):
        raise ValueError("vgb_repair_policy_ready must be a bool")
    if not mapping(artifact.get("backtracking_policy")):
        raise ValueError("backtracking_policy must be non-empty")
    if not mapping(artifact.get("proposal_budget")):
        raise ValueError("proposal_budget must be non-empty")
    exact_rules = mapping(artifact.get("exact_acceptance_rules"))
    if exact_rules.get("final_acceptance_authority") != "exact_verifier_only":
        raise ValueError("exact verifier must be final authority")
    if exact_rules.get("llm_judge_final_acceptance_allowed") is not False:
        raise ValueError("LLM judges cannot be final acceptance authority")
    if not mapping(artifact.get("verifier_confidence_thresholds")):
        raise ValueError("verifier_confidence_thresholds must be non-empty")
    if artifact.get("no_new_model_execution") is not True:
        raise ValueError("no_new_model_execution must be true")
    actions = set(mapping(mapping(artifact.get("backtracking_policy")).get("actions")))
    if actions != {"accepted", "rejected", "backtracked", "abstained"}:
        raise ValueError("backtracking_policy must define all four actions")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")
