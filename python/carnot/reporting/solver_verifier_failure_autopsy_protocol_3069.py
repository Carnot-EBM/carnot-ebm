"""Build the Exp 3069 solver-verifier failure autopsy protocol.

Spec refs: REQ-REPORT-3069, SCENARIO-REPORT-3069.

This module is an accounting and protocol step, not a new model run. The prior
experiments already produced the key negative evidence: the verifier rejected
the only exact-good solution in Exp 3057, and the LLM-guided SMT pilot in Exp
3058 matched the exact solver's own fallback. The job here is to preserve that
evidence in a form the next experiments can consume directly, so a repair or
SMT guidance retry cannot quietly relabel a failed calibration as readiness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.287"
SCHEMA = "carnot.solver_verifier_failure_autopsy_protocol.v1"
ARTIFACT = "experiment_3069_solver_verifier_failure_autopsy_protocol_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCRIPT_FILENAME = f"{ARTIFACT}.py"
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME

EXP3057_REL_PATH = Path("results/experiment_3057_local_sota_solution_verifier_gain_panel_v1.json")
EXP3058_REL_PATH = Path("results/experiment_3058_aquaforte_style_llm_guided_smt_pilot_v1.json")
MATRIX_V20_REL_PATH = Path("results/experiment_3065_cross_corpus_matrix_v20.json")
CAPSTONE_V286_REL_PATH = Path("results/experiment_3066_capstone_v286.json")
CODEX_REL_PATH = Path("CODEX.md")
CLAUDE_REL_PATH = Path("CLAUDE.md")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")

REQUIRED_ARTIFACT_FIELDS = (
    "verifier_failure_autopsy_ready",
    "root_cause_hypotheses",
    "recovery_protocol",
    "abstention_policy",
    "candidate_signals",
    "promotion_disqualifiers",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

JSON_SOURCES = (
    ("exp3057", EXP3057_REL_PATH, "local_sota_solution_verifier_gain_panel"),
    ("exp3058", EXP3058_REL_PATH, "aquaforte_style_llm_guided_smt_pilot"),
    ("exp3065", MATRIX_V20_REL_PATH, "matrix_v20_solver_grounding_status"),
    ("exp3066", CAPSTONE_V286_REL_PATH, "capstone_v286_solver_grounding_status"),
)
TEXT_SOURCES = (
    ("codex", CODEX_REL_PATH, "repo_workflow_instructions"),
    ("claude", CLAUDE_REL_PATH, "verifier_authenticity_and_workflow_instructions"),
    ("research_references", RESEARCH_REFERENCES_REL_PATH, "confidence_diagnostic_references"),
)
FAILURE_MODES = (
    "false_negatives",
    "no_verifier_gain",
    "no_smt_lift",
    "self_verification_risk",
    "solver_only_equivalence",
)
CONSUMER_EXPERIMENTS = ("exp3070", "exp3071", "exp3072", "exp3075")


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and treat missing or malformed files as no evidence."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_text(path: Path) -> str:
    """Read text evidence while treating absent workflow docs as unavailable."""

    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    """Return a source checksum without mutating any prior artifact."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3069: build the artifact-only autopsy and recovery protocol."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    json_payloads = {
        experiment_id: read_json_object(root_path / rel_path)
        for experiment_id, rel_path, _role in JSON_SOURCES
    }
    text_payloads = {
        experiment_id: read_text(root_path / rel_path)
        for experiment_id, rel_path, _role in TEXT_SOURCES
    }
    source_artifacts = _source_artifacts(root_path, json_payloads, text_payloads)
    exp3057 = json_payloads["exp3057"]
    exp3058 = json_payloads["exp3058"]
    metrics_summary = _metrics_summary(exp3057, exp3058)
    failure_modes = _failure_mode_classification(exp3057, exp3058)
    candidate_signals = _candidate_signals(text_payloads["research_references"])
    promotion_disqualifiers = _promotion_disqualifiers()
    recovery_protocol = _recovery_protocol()
    ready = _ready(
        source_artifacts=source_artifacts,
        failure_modes=failure_modes,
        candidate_signals=candidate_signals,
        promotion_disqualifiers=promotion_disqualifiers,
        recovery_protocol=recovery_protocol,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "verifier_failure_autopsy_ready": ready,
        "metrics_summary": metrics_summary,
        "failure_mode_classification": failure_modes,
        "root_cause_hypotheses": _root_cause_hypotheses(metrics_summary),
        "recovery_protocol": recovery_protocol,
        "abstention_policy": _abstention_policy(),
        "candidate_signals": candidate_signals,
        "promotion_disqualifiers": promotion_disqualifiers,
        "matrix_capstone_context": _matrix_capstone_context(
            json_payloads["exp3065"], json_payloads["exp3066"]
        ),
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row["sha256"] for row in source_artifacts},
        "inference_substrate": _inference_substrate(),
        "no_live_model_inference": True,
        "no_live_llm_inference": True,
        "no_new_solver_run": True,
        "no_new_verifier_run": True,
        "no_repair_promotion": True,
        "no_historical_artifact_rewrite": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
        "duration_s": _duration(start, now_s),
        "honest_verdict": _honest_verdict(ready, source_artifacts),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3069 terminal artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise if the autopsy can be misread as a live verifier success."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("live_llm_inference") is not False:
        raise ValueError("inference_substrate.live_llm_inference must be false")
    if not artifact.get("promotion_disqualifiers"):
        raise ValueError("promotion_disqualifiers must not be empty")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("verifier_failure_autopsy_ready") is True:
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must start with a terminal success prefix")
        if not _contains_all_failure_modes(artifact.get("failure_mode_classification")):
            raise ValueError("failure_mode_classification must cover all required modes")
        return
    if not verdict.startswith("blocked_missing_source:"):
        raise ValueError("honest_verdict must disclose blocked_missing_source")


def _source_artifacts(
    root: Path,
    json_payloads: Mapping[str, Mapping[str, Any]],
    text_payloads: Mapping[str, str],
) -> list[JsonDict]:
    rows = []
    for experiment_id, rel_path, role in JSON_SOURCES:
        path = root / rel_path
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": rel_path.as_posix(),
                "role": role,
                "kind": "json",
                "present": path.is_file(),
                "readable": bool(json_payloads[experiment_id]),
                "sha256": sha256_file(path),
            }
        )
    for experiment_id, rel_path, role in TEXT_SOURCES:
        path = root / rel_path
        rows.append(
            {
                "experiment_id": experiment_id,
                "path": rel_path.as_posix(),
                "role": role,
                "kind": "text",
                "present": path.is_file(),
                "readable": bool(text_payloads[experiment_id]),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _metrics_summary(exp3057: Mapping[str, Any], exp3058: Mapping[str, Any]) -> JsonDict:
    exp3058_lift = exp3058.get("guidance_vs_solver_only", {})
    lift = exp3058_lift if isinstance(exp3058_lift, Mapping) else {}
    return {
        "exp3057": {
            "false_negative_rate": _float(exp3057.get("false_negative_rate")),
            "false_positive_rate": _float(exp3057.get("false_positive_rate")),
            "one_shot_solver_accuracy": _float(exp3057.get("one_shot_solver_accuracy")),
            "verifier_selected_accuracy": _float(exp3057.get("verifier_selected_accuracy")),
            "verifier_gain_delta": _float(exp3057.get("verifier_gain_delta")),
            "exact_solver_agreement": _float(exp3057.get("exact_solver_agreement")),
            "exact_solver_authority": str(exp3057.get("exact_solver_authority") or ""),
            "flagged_adversarial": bool(exp3057.get("flagged_adversarial")),
        },
        "exp3058": {
            "guided_success_count": int(exp3058.get("guided_success_count") or 0),
            "solver_only_success_count": int(exp3058.get("solver_only_success_count") or 0),
            "guided_minus_solver_only_success_count": int(
                lift.get("guided_minus_solver_only_success_count") or 0
            ),
            "invalid_llm_proposal_count": int(exp3058.get("invalid_llm_proposal_count") or 0),
            "formal_fallback_preserved": bool(exp3058.get("formal_fallback_preserved")),
            "flagged_adversarial": bool(exp3058.get("flagged_adversarial")),
        },
    }


def _failure_mode_classification(
    exp3057: Mapping[str, Any],
    exp3058: Mapping[str, Any],
) -> list[JsonDict]:
    metrics = _metrics_summary(exp3057, exp3058)
    lift = metrics["exp3058"]["guided_minus_solver_only_success_count"]
    return [
        {
            "failure_mode": "false_negatives",
            "present": metrics["exp3057"]["false_negative_rate"] > 0.0,
            "source_artifact": EXP3057_REL_PATH.as_posix(),
            "evidence": "Exp 3057 false_negative_rate is nonzero.",
        },
        {
            "failure_mode": "no_verifier_gain",
            "present": metrics["exp3057"]["verifier_gain_delta"] <= 0.0,
            "source_artifact": EXP3057_REL_PATH.as_posix(),
            "evidence": "Verifier-selected accuracy did not exceed one-shot solver accuracy.",
        },
        {
            "failure_mode": "no_smt_lift",
            "present": lift <= 0,
            "source_artifact": EXP3058_REL_PATH.as_posix(),
            "evidence": "Guided SMT success did not exceed solver-only fallback success.",
        },
        {
            "failure_mode": "self_verification_risk",
            "present": True,
            "source_artifact": EXP3057_REL_PATH.as_posix(),
            "evidence": "Same-family or single-model verifier evidence needs exact-solver arbitration.",
        },
        {
            "failure_mode": "solver_only_equivalence",
            "present": metrics["exp3058"]["guided_success_count"]
            == metrics["exp3058"]["solver_only_success_count"],
            "source_artifact": EXP3058_REL_PATH.as_posix(),
            "evidence": "Guided SMT and solver-only counts are identical.",
        },
    ]


def _root_cause_hypotheses(metrics: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "id": "h1_over_rejection",
            "source_artifact": EXP3057_REL_PATH.as_posix(),
            "metric": "false_negative_rate",
            "observed_value": metrics["exp3057"]["false_negative_rate"],
            "hypothesis": "The verifier is calibrated as an over-rejector and suppresses exact-good solutions.",
        },
        {
            "id": "h2_negative_selection_delta",
            "source_artifact": EXP3057_REL_PATH.as_posix(),
            "metric": "verifier_gain_delta",
            "observed_value": metrics["exp3057"]["verifier_gain_delta"],
            "hypothesis": "Verifier selection is worse than the solver-only one-shot baseline.",
        },
        {
            "id": "h3_guidance_not_causal",
            "source_artifact": EXP3058_REL_PATH.as_posix(),
            "metric": "guided_minus_solver_only_success_count",
            "observed_value": metrics["exp3058"]["guided_minus_solver_only_success_count"],
            "hypothesis": "LLM guidance is not adding solving power beyond exact fallback enumeration.",
        },
        {
            "id": "h4_artifact_flag_contamination",
            "source_artifact": MATRIX_V20_REL_PATH.as_posix(),
            "metric": "flagged_solver_grounded_no_gain",
            "observed_value": True,
            "hypothesis": "Matrix and capstone aggregation correctly carry this evidence as flagged, not clean.",
        },
    ]


def _candidate_signals(research_references: str) -> list[JsonDict]:
    has_references = bool(research_references.strip())
    return [
        {
            "name": "first_token_entropy",
            "source_reference": "The First Token Knows",
            "local_diagnostic": "Record normalized entropy for the first content token when logits exist.",
            "required_for_recovery": True,
            "accessible_now": False,
            "blocked_reason": "Exp 3069 performs no live inference and prior artifacts lack token logits.",
            "reference_trace_present": has_references,
        },
        {
            "name": "abstention_precision",
            "source_reference": "Distributional uncertainty and abstention framing",
            "local_diagnostic": "Measure the precision of abstained cases among exact-solver failures.",
            "required_for_recovery": True,
            "accessible_now": True,
            "reference_trace_present": has_references,
        },
        {
            "name": "rejection_recall",
            "source_reference": "Verifier calibration failure analysis",
            "local_diagnostic": "Measure recall for rejecting exact-solver-bad candidates.",
            "required_for_recovery": True,
            "accessible_now": True,
            "reference_trace_present": has_references,
        },
        {
            "name": "confidence_coverage",
            "source_reference": "Verifiability as a metric",
            "local_diagnostic": "Report accepted, rejected, and abstained fractions by confidence bin.",
            "required_for_recovery": True,
            "accessible_now": True,
            "reference_trace_present": has_references,
        },
        {
            "name": "lyapunov_perturbation_sensitivity",
            "source_reference": "Lyapunov-style stability under perturbation",
            "local_diagnostic": "Perturb prompts or candidate traces and measure decision stability.",
            "required_for_recovery": False,
            "accessible_now": False,
            "blocked_reason": "Requires logits, trajectory states, or controlled rerun traces.",
            "reference_trace_present": has_references,
        },
        {
            "name": "verge_mcs_feedback",
            "source_reference": "VERGE/MCS repair feedback",
            "local_diagnostic": "Attach minimal correction-set feedback to rejected candidates.",
            "required_for_recovery": True,
            "accessible_now": True,
            "reference_trace_present": has_references,
        },
    ]


def _recovery_protocol() -> JsonDict:
    minimum_fields = [
        *REQUIRED_ARTIFACT_FIELDS,
        "metrics_summary",
        "failure_mode_classification",
        "candidate_rows",
        "exact_solver_authority",
        "exact_checked_count",
        "abstained_count",
        "accepted_count",
        "rejected_count",
        "verifier_gain_delta",
        "false_positive_rate",
        "false_negative_rate",
        "guided_minus_solver_only_success_count",
        "reproducibility_checksum",
    ]
    return {
        "protocol_name": "bounded_verifier_gain_recovery",
        "consumer_ready": True,
        "consumer_experiments": list(CONSUMER_EXPERIMENTS),
        "minimum_artifact_fields": minimum_fields,
        "exact_solver_authority_requirements": {
            "primary_authority": "z3_or_exact_solver",
            "llm_must_not_be_authority": True,
            "accepted_rows_require_exact_checked": True,
            "solver_only_baseline_required": True,
            "candidate_level_counterexamples_required": True,
            "self_verification_allowed_only_as_signal": True,
        },
        "acceptance_gates": {
            "verifier_gain_delta_min_exclusive": 0.0,
            "false_negative_rate_max": 0.25,
            "false_positive_rate_max": 0.05,
            "guided_minus_solver_only_success_count_min": 1,
            "exact_solver_agreement_min": 1.0,
            "abstention_precision_min": 0.80,
            "confidence_coverage_bins_min": 3,
        },
        "required_row_fields": [
            "fixture_id",
            "candidate_id",
            "model_id",
            "verifier_decision",
            "abstention_decision",
            "confidence_signals",
            "exact_solver_status",
            "exact_checked",
            "mcs_feedback",
            "accepted_by_exact_solver",
        ],
    }


def _abstention_policy() -> JsonDict:
    return {
        "enabled": True,
        "forced_accept_reject_disallowed": True,
        "allowed_decisions": ["accept", "reject", "abstain"],
        "abstain_when": [
            "confidence signals disagree",
            "exact solver authority is unavailable",
            "first-token entropy or proxy confidence falls in gray zone",
            "VERGE/MCS feedback is missing for a rejection",
        ],
        "minimum_reported_metrics": [
            "abstention_precision",
            "rejection_recall",
            "confidence_coverage",
        ],
        "promotion_rule": "abstention improves safety only if it reduces false accepts without hiding false negatives",
    }


def _promotion_disqualifiers() -> list[JsonDict]:
    return [
        {
            "experiment_id": "exp3070",
            "id": "exp3070_no_positive_gain",
            "blocks": "local_sota_verifier_repair",
            "condition": "verifier_gain_delta <= 0 or false_negative_rate >= 1.0",
            "reason": "A repaired verifier must first beat solver-only selection under exact labels.",
        },
        {
            "experiment_id": "exp3071",
            "id": "exp3071_no_abstention_calibration",
            "blocks": "confidence_gate_promotion",
            "condition": "abstention_precision missing or confidence_coverage missing",
            "reason": "Uncertain cases must not be forced into accept/reject decisions.",
        },
        {
            "experiment_id": "exp3072",
            "id": "exp3072_no_smt_lift",
            "blocks": "llm_guided_smt_promotion",
            "condition": "guided_minus_solver_only_success_count <= 0",
            "reason": "Guidance that equals solver-only fallback is not a promotable SMT result.",
        },
        {
            "experiment_id": "exp3075",
            "id": "exp3075_missing_exact_authority",
            "blocks": "repair_or_capstone_promotion",
            "condition": "any accepted row lacks exact_checked=true or exact solver authority",
            "reason": "Repair promotion requires exact-solver authority over accepted candidates.",
        },
        {
            "experiment_id": "exp3075",
            "id": "exp3075_self_verification_only",
            "blocks": "repair_or_capstone_promotion",
            "condition": "verifier and generator agree without independent solver evidence",
            "reason": "Self-verification is a diagnostic signal, not an authority.",
        },
    ]


def _matrix_capstone_context(matrix: Mapping[str, Any], capstone: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_v20_ready": bool(matrix.get("matrix_v20_ready")),
        "matrix_solver_grounding_status": _nested_status(
            matrix.get("status_summaries"), "solver_grounded_verification"
        ),
        "capstone_ready": bool(capstone.get("capstone_ready")),
        "paper_ready": bool(capstone.get("paper_ready")),
        "solver_grounding_status": str(capstone.get("solver_grounding_status") or ""),
    }


def _inference_substrate() -> JsonDict:
    return {
        "mode": "artifact_only_protocol_autopsy",
        "protocol_only": True,
        "live_llm_inference": False,
        "local_gguf_inference": False,
        "fresh_solver_execution": False,
        "fresh_verifier_scoring": False,
        "source_artifacts_only": True,
    }


def _ready(
    *,
    source_artifacts: list[Mapping[str, Any]],
    failure_modes: list[Mapping[str, Any]],
    candidate_signals: list[Mapping[str, Any]],
    promotion_disqualifiers: list[Mapping[str, Any]],
    recovery_protocol: Mapping[str, Any],
) -> bool:
    return (
        all(row.get("present") is True and row.get("readable") is True for row in source_artifacts)
        and _contains_all_failure_modes(failure_modes)
        and len(candidate_signals) >= 6
        and set(row.get("experiment_id") for row in promotion_disqualifiers).issuperset(
            CONSUMER_EXPERIMENTS
        )
        and recovery_protocol.get("consumer_ready") is True
    )


def _honest_verdict(ready: bool, source_artifacts: list[Mapping[str, Any]]) -> str:
    if ready:
        return "complete: verifier_failure_autopsy_ready=true; promotions blocked until recovery gates pass"
    missing = [
        str(row["path"])
        for row in source_artifacts
        if row.get("present") is not True or row.get("readable") is not True
    ]
    return f"blocked_missing_source: verifier_failure_autopsy_ready=false; missing={missing}"


def _contains_all_failure_modes(rows: Any) -> bool:
    if not isinstance(rows, list):
        return False
    return {str(row.get("failure_mode")) for row in rows if isinstance(row, Mapping)}.issuperset(
        FAILURE_MODES
    )


def _nested_status(payload: Any, key: str) -> str:
    if not isinstance(payload, Mapping):
        return ""
    row = payload.get(key)
    if not isinstance(row, Mapping):
        return ""
    return str(row.get("status") or "")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
