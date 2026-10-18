"""Build the Exp 3083 verifier-hardness autopsy protocol.

Spec refs: REQ-REPORT-3083, SCENARIO-REPORT-3083.

This module turns the `.287` verifier failures into a machine-readable
recovery protocol for `.288`. It deliberately performs no new solving,
verification, repair, or model inference because the point is to keep the next
reruns honest: later experiments must first cite the known failure evidence
and then report accept, reject, abstain, and formal-feedback behavior as
separate quantities.
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
MILESTONE = "2026.05.288"
SCHEMA = "carnot.verifier_hardness_autopsy_protocol.v1"
ARTIFACT = "experiment_3083_verifier_hardness_autopsy_protocol_v1"
OUTPUT_REL_PATH = Path("results/experiment_3083_verifier_hardness_autopsy_protocol_v1.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts/experiment_3083_verifier_hardness_autopsy_protocol_v1.py"

EXP3057_REL_PATH = Path("results/experiment_3057_local_sota_solution_verifier_gain_panel_v1.json")
EXP3070_REL_PATH = Path("results/experiment_3070_first_token_abstention_sota_panel_v1.json")
EXP3071_REL_PATH = Path("results/experiment_3071_verge_mcs_smt_correction_pilot_v1.json")
EXP3080_REL_PATH = Path("results/experiment_3080_capstone_v287.json")
AGENTS_REL_PATH = Path("AGENTS.md")
CODEX_REL_PATH = Path("CODEX.md")
CLAUDE_REL_PATH = Path("CLAUDE.md")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")

REQUIRED_ARTIFACT_FIELDS = (
    "verifier_hardness_protocol_ready",
    "prior_failure_modes",
    "perturbation_categories",
    "abstention_metrics_required",
    "formal_feedback_disqualifiers",
    "repair_disqualifiers",
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
    "shipped_",
)
REQUIRED_AXES = {"solving", "verifying", "abstaining", "repairing"}
REQUIRED_EXPERIMENTS = {"exp3084", "exp3085", "exp3086", "exp3087", "exp3088", "exp3089"}
REQUIRED_GLOBAL_DISQUALIFIERS = {
    "label_leakage",
    "solver_only_parity_without_lift",
    "tautological_repair",
    "syntax_only_success",
    "tiny_model_headline_substitution",
}

JSON_SOURCES = (
    ("exp3057", EXP3057_REL_PATH, "negative_verifier_gain_evidence"),
    ("exp3070", EXP3070_REL_PATH, "first_token_abstention_evidence"),
    ("exp3071", EXP3071_REL_PATH, "formal_feedback_mcs_evidence"),
    ("exp3080", EXP3080_REL_PATH, "capstone_v287_blocker_evidence"),
)
TEXT_SOURCES = (
    ("agents", AGENTS_REL_PATH, "repo_startup_instructions"),
    ("codex", CODEX_REL_PATH, "repo_workflow_instructions"),
    ("claude", CLAUDE_REL_PATH, "verifier_authenticity_instructions"),
    ("research_references", RESEARCH_REFERENCES_REL_PATH, "verifier_hardness_references"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON authority artifact and fail closed on missing evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_text(path: Path) -> str:
    """Read a workflow or research-reference file without inventing evidence."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def sha256_file(path: Path) -> str | None:
    """Checksum source files so later matrix rows can trace the protocol inputs."""

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
    """REQ-REPORT-3083: build the failure-aware verifier recovery protocol."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    json_payloads = {
        experiment_id: read_json_object(root_path / rel_path)
        for experiment_id, rel_path, _role in JSON_SOURCES
    }
    text_payloads = {
        source_id: read_text(root_path / rel_path) for source_id, rel_path, _role in TEXT_SOURCES
    }
    source_artifacts = _source_artifacts(root_path, json_payloads, text_payloads)
    metrics = _prior_metrics(json_payloads)
    prior_failure_modes = _prior_failure_modes(metrics)
    perturbation_categories = _perturbation_categories()
    abstention_metrics = _abstention_metrics_required()
    metric_contracts = _experiment_metric_contracts()
    global_disqualifiers = _global_disqualifiers()
    formal_disqualifiers = _formal_feedback_disqualifiers()
    repair_disqualifiers = _repair_disqualifiers()
    ready = _ready(
        source_artifacts=source_artifacts,
        prior_failure_modes=prior_failure_modes,
        perturbation_categories=perturbation_categories,
        metric_contracts=metric_contracts,
        global_disqualifiers=global_disqualifiers,
        formal_disqualifiers=formal_disqualifiers,
        repair_disqualifiers=repair_disqualifiers,
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "verifier_hardness_protocol_ready": ready,
        "prior_metrics": metrics,
        "research_reference_comparisons": _research_reference_comparisons(
            text_payloads["research_references"]
        ),
        "prior_failure_modes": prior_failure_modes,
        "perturbation_categories": perturbation_categories,
        "abstention_metrics_required": abstention_metrics,
        "experiment_metric_contracts": metric_contracts,
        "global_disqualifiers": global_disqualifiers,
        "formal_feedback_disqualifiers": formal_disqualifiers,
        "repair_disqualifiers": repair_disqualifiers,
        "source_artifacts": source_artifacts,
        "source_checksums": {row["path"]: row.get("sha256") for row in source_artifacts},
        "inference_substrate": _inference_substrate(),
        "no_live_model_inference": True,
        "no_live_llm_inference": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_live_repair_rerun": True,
        "no_historical_artifact_rewrite": True,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
        "duration_s": _duration(start, now_s),
        "blocked_reasons": _blocked_reasons(source_artifacts),
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
    """Build and persist the Exp 3083 terminal artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject artifacts that could be mistaken for a successful verifier rerun."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("no_live_llm_inference") is not True:
        raise ValueError("inference_substrate.no_live_llm_inference must be true")
    axes = {
        str(row.get("primary_axis"))
        for row in _as_mapping_rows(artifact.get("perturbation_categories"))
    }
    if axes != REQUIRED_AXES:
        raise ValueError(
            "perturbation categories must cover solving/verifying/abstaining/repairing"
        )
    if set(_as_mapping(artifact.get("experiment_metric_contracts"))) != REQUIRED_EXPERIMENTS:
        raise ValueError("Exp 3084-3089 metric contracts must all be present")
    global_ids = {
        str(row.get("id")) for row in _as_mapping_rows(artifact.get("global_disqualifiers"))
    }
    formal_ids = {
        str(row.get("id"))
        for row in _as_mapping_rows(artifact.get("formal_feedback_disqualifiers"))
    }
    repair_ids = {
        str(row.get("id")) for row in _as_mapping_rows(artifact.get("repair_disqualifiers"))
    }
    if not REQUIRED_GLOBAL_DISQUALIFIERS <= global_ids:
        raise ValueError(
            "disqualifiers must cover leakage, parity, tautology, syntax, tiny-model cases"
        )
    if not formal_ids or not repair_ids:
        raise ValueError("formal-feedback and repair disqualifiers must not be empty")
    verdict = str(artifact.get("honest_verdict", ""))
    if artifact.get("verifier_hardness_protocol_ready") is True:
        if not verdict.startswith(SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must start with a terminal success prefix")
        return
    if not verdict.startswith("blocked_missing_source:"):
        raise ValueError("honest_verdict must disclose blocked_missing_source")


def _source_artifacts(
    root: Path,
    json_payloads: Mapping[str, Mapping[str, Any]],
    text_payloads: Mapping[str, str],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source_id, rel_path, role in JSON_SOURCES:
        path = root / rel_path
        rows.append(
            {
                "source_id": source_id,
                "path": rel_path.as_posix(),
                "role": role,
                "kind": "json",
                "required": True,
                "present": path.is_file(),
                "readable": bool(json_payloads[source_id]),
                "sha256": sha256_file(path),
            }
        )
    for source_id, rel_path, role in TEXT_SOURCES:
        path = root / rel_path
        rows.append(
            {
                "source_id": source_id,
                "path": rel_path.as_posix(),
                "role": role,
                "kind": "text",
                "required": True,
                "present": path.is_file(),
                "readable": bool(text_payloads[source_id].strip()),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _prior_metrics(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp3057 = payloads["exp3057"]
    exp3070 = payloads["exp3070"]
    exp3071 = payloads["exp3071"]
    exp3080 = payloads["exp3080"]
    lift = _as_mapping(exp3071.get("guidance_vs_solver_only"))
    return {
        "exp3057": {
            "verifier_gain_delta": _float(exp3057.get("verifier_gain_delta")),
            "false_negative_rate": _float(exp3057.get("false_negative_rate")),
            "false_positive_rate": _float(exp3057.get("false_positive_rate")),
            "one_shot_solver_accuracy": _float(exp3057.get("one_shot_solver_accuracy")),
            "verifier_selected_accuracy": _float(exp3057.get("verifier_selected_accuracy")),
            "exact_solver_agreement": _float(exp3057.get("exact_solver_agreement")),
            "flagged_adversarial": bool(exp3057.get("flagged_adversarial")),
            "corrigendum_kinds": _corrigendum_kinds(exp3057),
        },
        "exp3070": {
            "first_token_auc": _float(exp3070.get("first_token_auc")),
            "abstention_precision": _float(exp3070.get("abstention_precision")),
            "abstention_coverage": _float(exp3070.get("abstention_coverage")),
            "rejection_recall": _float(exp3070.get("rejection_recall")),
            "accepted_count": _int(exp3070.get("accepted_count")),
            "rejected_count": _int(exp3070.get("rejected_count")),
            "abstained_count": _int(exp3070.get("abstained_count")),
            "false_negative_rate": _float(exp3070.get("false_negative_rate")),
            "false_positive_rate": _float(exp3070.get("false_positive_rate")),
            "verifier_gain_delta_with_abstention": _float(
                exp3070.get("verifier_gain_delta_with_abstention")
            ),
            "flagged_adversarial": bool(exp3070.get("flagged_adversarial")),
            "corrigendum_kinds": _corrigendum_kinds(exp3070),
        },
        "exp3071": {
            "mcs_feedback_ready": bool(exp3071.get("mcs_feedback_ready")),
            "guided_success_count": _int(exp3071.get("guided_success_count")),
            "solver_only_success_count": _int(exp3071.get("solver_only_success_count")),
            "guided_minus_solver_only_success_count": _int(
                lift.get("guided_minus_solver_only_success_count")
            ),
            "invalid_llm_proposal_count": _int(exp3071.get("invalid_llm_proposal_count")),
            "mcs_count": _int(exp3071.get("mcs_count")),
            "fixture_count": _int(exp3071.get("fixture_count")),
            "formal_fallback_preserved": bool(exp3071.get("formal_fallback_preserved")),
            "flagged_adversarial": bool(exp3071.get("flagged_adversarial")),
            "corrigendum_kinds": _corrigendum_kinds(exp3071),
        },
        "exp3080": {
            "capstone_ready": bool(exp3080.get("capstone_ready")),
            "paper_ready": bool(exp3080.get("paper_ready")),
            "publication_blocker_count": _int(exp3080.get("publication_blocker_count")),
            "verifier_gain_status": str(exp3080.get("verifier_gain_status") or ""),
            "repair_claim_status": str(exp3080.get("repair_claim_status") or ""),
            "fr11_self_learning_status": str(exp3080.get("fr11_self_learning_status") or ""),
            "next_milestone_recommendation": str(
                exp3080.get("next_milestone_recommendation") or ""
            ),
        },
    }


def _prior_failure_modes(metrics: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    exp3057 = metrics["exp3057"]
    exp3070 = metrics["exp3070"]
    exp3071 = metrics["exp3071"]
    exp3080 = metrics["exp3080"]
    return [
        {
            "id": "negative_verifier_gain",
            "source_artifact": EXP3057_REL_PATH.as_posix(),
            "metric": "verifier_gain_delta",
            "observed_value": exp3057["verifier_gain_delta"],
            "principle": "new experiments must address known negative verifier-gain results",
            "present": exp3057["verifier_gain_delta"] <= 0.0,
        },
        {
            "id": "exact_good_false_negative",
            "source_artifact": EXP3057_REL_PATH.as_posix(),
            "metric": "false_negative_rate",
            "observed_value": exp3057["false_negative_rate"],
            "principle": "acceptance and rejection must be measured separately",
            "present": exp3057["false_negative_rate"] > 0.0,
        },
        {
            "id": "low_abstention_precision",
            "source_artifact": EXP3070_REL_PATH.as_posix(),
            "metric": "abstention_precision",
            "observed_value": exp3070["abstention_precision"],
            "principle": "abstain is not success unless it catches risky rows",
            "present": exp3070["abstention_precision"] < 0.8,
        },
        {
            "id": "tautological_abstention_metrics",
            "source_artifact": EXP3070_REL_PATH.as_posix(),
            "metric": "corrigendum_kinds",
            "observed_value": exp3070["corrigendum_kinds"],
            "principle": "identical metric values need independent row-level denominators",
            "present": "TAUTOLOGY" in exp3070["corrigendum_kinds"],
        },
        {
            "id": "formal_feedback_solver_only_parity",
            "source_artifact": EXP3071_REL_PATH.as_posix(),
            "metric": "guided_minus_solver_only_success_count",
            "observed_value": exp3071["guided_minus_solver_only_success_count"],
            "principle": "formal feedback must show lift beyond fallback enumeration",
            "present": exp3071["guided_minus_solver_only_success_count"] <= 0,
        },
        {
            "id": "provenance_contamination",
            "source_artifact": "results/experiment_3057_3070_3071_prior_artifacts",
            "metric": "corrigendum_kinds",
            "observed_value": sorted(
                set(exp3057["corrigendum_kinds"])
                | set(exp3070["corrigendum_kinds"])
                | set(exp3071["corrigendum_kinds"])
            ),
            "principle": "live-inference claims need credible duration and substrate provenance",
            "present": "DURATION_TOO_SHORT"
            in (
                set(exp3057["corrigendum_kinds"])
                | set(exp3070["corrigendum_kinds"])
                | set(exp3071["corrigendum_kinds"])
            ),
        },
        {
            "id": "capstone_publication_blockers",
            "source_artifact": EXP3080_REL_PATH.as_posix(),
            "metric": "publication_blocker_count",
            "observed_value": exp3080["publication_blocker_count"],
            "principle": "matrix/capstone blockers stay blocked until rerun gates clear",
            "present": exp3080["publication_blocker_count"] > 0,
        },
    ]


def _research_reference_comparisons(research_references: str) -> list[JsonDict]:
    has_trace = bool(research_references.strip())
    return [
        {
            "id": "verifier_hardness",
            "finding": "Verification can be harder than solving and can be insensitive to perturbations.",
            "local_consequence": "Exp 3084-3085 must include label-preserving and label-flipping perturbations.",
            "reference_trace_present": has_trace,
        },
        {
            "id": "i_calm_abstention",
            "finding": "Confidence-aware abstention moves coverage and reliability together.",
            "local_consequence": "Exp 3086 must report coverage, precision, and reliability by decision bucket.",
            "reference_trace_present": has_trace,
        },
        {
            "id": "task_abstention",
            "finding": "Code-task abstention can use execution consistency rather than oracle leakage.",
            "local_consequence": "Exp 3087 must separate execution-consistency abstention from correctness labels.",
            "reference_trace_present": has_trace,
        },
        {
            "id": "self_verification_asymmetry",
            "finding": "Improving generation does not automatically improve self-verification.",
            "local_consequence": "FR-11 rows must measure verifier improvement separately from solver improvement.",
            "reference_trace_present": has_trace,
        },
        {
            "id": "formal_feedback",
            "finding": "Formal feedback needs structural anchors and vacuity guards.",
            "local_consequence": "Exp 3088 must prove guided lift over solver-only and reject syntax-only success.",
            "reference_trace_present": has_trace,
        },
    ]


def _perturbation_categories() -> list[JsonDict]:
    return [
        {
            "name": "label_preserving_exact_fixture_variants",
            "primary_axis": "solving",
            "separates_from": ["verifying", "abstaining", "repairing"],
            "required_measurement": "solver_accuracy_delta_under_prompt_or_format_changes",
            "non_vacuous_gate": "exact label unchanged and solver-only baseline reported",
        },
        {
            "name": "candidate_label_flip_verification_variants",
            "primary_axis": "verifying",
            "separates_from": ["solving", "abstaining", "repairing"],
            "required_measurement": "accept_reject_confusion_matrix_against_exact_labels",
            "non_vacuous_gate": "correct and incorrect candidates share prompts without answer leakage",
        },
        {
            "name": "confidence_gray_zone_abstention_variants",
            "primary_axis": "abstaining",
            "separates_from": ["solving", "verifying", "repairing"],
            "required_measurement": "accept_reject_abstain_rates_by_exact_label_and_confidence_bin",
            "non_vacuous_gate": "abstention precision and coverage use independent denominators",
        },
        {
            "name": "localized_formal_feedback_repair_variants",
            "primary_axis": "repairing",
            "separates_from": ["solving", "verifying", "abstaining"],
            "required_measurement": "guided_minus_solver_only_success_count_with_exact_validation",
            "non_vacuous_gate": "feedback localizes a failing constraint and final repair passes semantic checks",
        },
    ]


def _abstention_metrics_required() -> list[JsonDict]:
    return [
        {
            "name": "acceptance_precision",
            "formula": "exact_correct_accepted / accepted_count",
            "decision": "accept",
            "non_vacuous_requirement": "accepted_count > 0 and exact labels hidden from prompts",
        },
        {
            "name": "acceptance_coverage",
            "formula": "accepted_count / total_count",
            "decision": "accept",
            "non_vacuous_requirement": "reported with abstention and rejection counts",
        },
        {
            "name": "rejection_recall",
            "formula": "exact_incorrect_rejected / exact_incorrect_count",
            "decision": "reject",
            "non_vacuous_requirement": "requires at least one exact-incorrect row",
        },
        {
            "name": "rejection_precision",
            "formula": "exact_incorrect_rejected / rejected_count",
            "decision": "reject",
            "non_vacuous_requirement": "rejected_count > 0 and false rejects listed",
        },
        {
            "name": "abstention_precision",
            "formula": "risky_or_unverifiable_abstained / abstained_count",
            "decision": "abstain",
            "non_vacuous_requirement": "abstained_count > 0 with exact-label distribution reported",
        },
        {
            "name": "abstention_coverage",
            "formula": "abstained_count / total_count",
            "decision": "abstain",
            "non_vacuous_requirement": "must not equal gain metrics by construction",
        },
        {
            "name": "false_accept_rate",
            "formula": "exact_incorrect_accepted / exact_incorrect_count",
            "decision": "accept",
            "non_vacuous_requirement": "zero false accepts is required before repair promotion",
        },
        {
            "name": "false_reject_rate",
            "formula": "exact_correct_rejected / exact_correct_count",
            "decision": "reject",
            "non_vacuous_requirement": "tracks Exp 3057-style exact-good false negatives",
        },
    ]


def _experiment_metric_contracts() -> dict[str, JsonDict]:
    metric_groups = ["acceptance", "rejection", "abstention", "formal_feedback"]
    contracts = {
        "exp3084": "exact_fixture_perturbation_bank",
        "exp3085": "rubric_conditioned_verifier_rejection",
        "exp3086": "i_calm_first_token_abstention_retry",
        "exp3087": "task_abstention_execution_consistency",
        "exp3088": "formal_feedback_repair_lift",
        "exp3089": "matrix_v22_recovery_gate",
    }
    return {
        exp_id: {
            "purpose": purpose,
            "metric_groups": metric_groups,
            "required_metrics": [
                "acceptance_precision",
                "rejection_recall",
                "abstention_precision",
                "abstention_coverage",
                "false_accept_rate",
                "false_reject_rate",
                "guided_minus_solver_only_success_count",
                "feedback_localization_accuracy",
            ],
            "exact_label_provenance_required": True,
            "solver_only_baseline_required": True,
            "row_level_decision_counts_required": True,
            "formal_feedback_lift_required": exp_id in {"exp3088", "exp3089"},
            "blocks_promotion_when": [
                "any accepted row lacks exact validation",
                "accept/reject/abstain denominators are missing",
                "solver-only parity is reported as feedback lift",
                "label leakage or tiny-model headline substitution is detected",
            ],
        }
        for exp_id, purpose in contracts.items()
    }


def _global_disqualifiers() -> list[JsonDict]:
    return [
        {
            "id": "label_leakage",
            "condition": "exact labels, answers, expected fixes, or split identity appear in prompts or candidate IDs",
            "blocks": "all exp3084-exp3089 promotion",
        },
        {
            "id": "solver_only_parity_without_lift",
            "condition": "guided or verifier-selected success equals solver-only fallback with no positive delta",
            "blocks": "formal-feedback and repair promotion",
        },
        {
            "id": "tautological_repair",
            "condition": "repair copies fixture answer, exact solver assignment, or metric definition",
            "blocks": "repair and formal-feedback promotion",
        },
        {
            "id": "syntax_only_success",
            "condition": "parse/schema/format passes but semantic tests or exact validators are absent or fail",
            "blocks": "repair promotion",
        },
        {
            "id": "tiny_model_headline_substitution",
            "condition": "headline model identity is replaced by a smaller or different model without explicit boundary",
            "blocks": "headline verifier, abstention, and repair claims",
        },
    ]


def _formal_feedback_disqualifiers() -> list[JsonDict]:
    return [
        {
            "id": "solver_only_parity_without_lift",
            "condition": "guided_minus_solver_only_success_count <= 0",
            "reason": "Formal feedback must add causal value beyond exact fallback.",
        },
        {
            "id": "feedback_without_localized_counterexample",
            "condition": "feedback lacks failing constraint IDs, MCS rows, or counterexample fields",
            "reason": "Binary pass/fail feedback is not localized repair evidence.",
        },
        {
            "id": "vacuous_formal_spec",
            "condition": "formal specification can pass while functional behavior is wrong",
            "reason": "Formal feedback needs a functional or semantic vacuity guard.",
        },
        {
            "id": "label_leakage",
            "condition": "feedback exposes the exact label or final answer before candidate generation",
            "reason": "Feedback is not evidence if it reveals the answer.",
        },
    ]


def _repair_disqualifiers() -> list[JsonDict]:
    return [
        {
            "id": "tautological_repair",
            "condition": "repair is derived from a known answer, expected output, or solver assignment",
            "reason": "Repair must generalize from localized failure, not copy labels.",
        },
        {
            "id": "syntax_only_success",
            "condition": "only syntax, parser, JSON, or schema status improves",
            "reason": "Semantic validation is the repair authority.",
        },
        {
            "id": "solver_fallback_counted_as_model_repair",
            "condition": "deterministic fallback output is counted as LLM repair success",
            "reason": "Fallback and generated repair substrates must stay separate.",
        },
        {
            "id": "tiny_model_headline_substitution",
            "condition": "repair uses non-headline tiny model output as headline SOTA evidence",
            "reason": "Model identity substitutions need explicit bounded labels.",
        },
        {
            "id": "intent_drift",
            "condition": "repaired candidate passes verifier by changing the task intent",
            "reason": "Repair must preserve fixture intent and behavioral tests.",
        },
    ]


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "specified_protocol_sources",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_solver": False,
        "executes_repair": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
    }


def _ready(
    *,
    source_artifacts: list[Mapping[str, Any]],
    prior_failure_modes: list[Mapping[str, Any]],
    perturbation_categories: list[Mapping[str, Any]],
    metric_contracts: Mapping[str, Any],
    global_disqualifiers: list[Mapping[str, Any]],
    formal_disqualifiers: list[Mapping[str, Any]],
    repair_disqualifiers: list[Mapping[str, Any]],
) -> bool:
    axes = {str(row.get("primary_axis")) for row in perturbation_categories}
    failure_ids = {str(row.get("id")) for row in prior_failure_modes if row.get("present") is True}
    disqualifier_ids = (
        {str(row.get("id")) for row in global_disqualifiers}
        | {str(row.get("id")) for row in formal_disqualifiers}
        | {str(row.get("id")) for row in repair_disqualifiers}
    )
    return (
        all(row.get("present") is True and row.get("readable") is True for row in source_artifacts)
        and len(failure_ids) >= 7
        and axes == REQUIRED_AXES
        and set(metric_contracts) == REQUIRED_EXPERIMENTS
        and REQUIRED_GLOBAL_DISQUALIFIERS <= disqualifier_ids
    )


def _blocked_reasons(source_artifacts: list[Mapping[str, Any]]) -> list[str]:
    return [
        f"missing_or_unreadable:{row['path']}"
        for row in source_artifacts
        if row.get("present") is not True or row.get("readable") is not True
    ]


def _honest_verdict(ready: bool, source_artifacts: list[Mapping[str, Any]]) -> str:
    if ready:
        return "complete: verifier_hardness_protocol_ready=true; exp3084_exp3089_gates_declared"
    return "blocked_missing_source: verifier_hardness_protocol_ready=false; " + "; ".join(
        _blocked_reasons(source_artifacts)
    )


def _corrigendum_kinds(payload: Mapping[str, Any]) -> list[str]:
    rows = payload.get("corrigendum_pending")
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("kind"))
        for row in rows
        if isinstance(row, Mapping) and row.get("kind") is not None
    ]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        if isinstance(value, bool):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
