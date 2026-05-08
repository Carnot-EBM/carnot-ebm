"""Build the Exp 1572 milestone .120 retrospective artifact.

Spec: REQ-REPORT-063, SCENARIO-REPORT-063.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
RUN_DATE = "20260508"
MILESTONE = "2026.05.120"
NEXT_MILESTONE = "2026.05.121"
EXPERIMENT = "1572_milestone_120_retro"
SCHEMA = "milestone_120_retro_v1"
CRITERIA_TOTAL = 14

DEFAULT_OUT_PATH = REPO_ROOT / "results" / "experiment_1572_milestone_120_retro.json"

MET = "MET"
NOT_MET = "NOT_MET"
BLOCKED = "BLOCKED"
MISSING = "MISSING"

SOURCE_FILES = {
    "exp1560": "experiment_1560_119_completion_archive_120_activation.json",
    "exp1561": "experiment_1561_kinetic_defense_zero_coupling_test.json",
    "exp1562": "experiment_1562_brain_linear_ar_k_sweep_extended.json",
    "exp1563": "experiment_1563_specann_rejection_architecture_record.json",
    "exp1564": "experiment_1564_thrml_vendored_block_gibbs_replacement.json",
    "exp1565": "experiment_1565_soft_gibbs_residual_implementation.json",
    "exp1566": "experiment_1566_candidate_warm_start_vs_cold_start_benchmark.json",
    "exp1567": "experiment_1567_rho_of_C_measurement_k6_ensemble.json",
    "exp1568": "experiment_1568_fr11_v14_retained_mode_collapse_audit.json",
    "exp1569": "experiment_1569_paper_v6_section_3_sampler_draft.json",
    "exp1570": "experiment_1570_soft_gibbs_coverage_bound_empirical_verification.json",
    "exp1571": "experiment_1571_step_wise_baseline_AR_REINFORCE.json",
    "exp1573": "experiment_1573_extropic_z1_readiness_packet_thrml_alignment_update.json",
}

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "next_milestone",
    "criteria_results",
    "criteria_met",
    "criteria_total",
    "criteria_met_fraction",
    "criteria_score_pct",
    "paper_v6_section_3_drafted",
    "all_4_carnot_contributions_validated",
    "rho_C_curve_published_ready",
    "terminal_verdicts",
    "notable_successes",
    "failures_or_partials",
    "bottlenecks_identified",
    "carry_forward_gates_121",
    "retro_complete",
    "honest_verdict",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-063: leave an auditable seed before source-artifact reads."""

    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "in_progress",
        "milestone": MILESTONE,
        "next_milestone": NEXT_MILESTONE,
        "criteria_results": {},
        "criteria_met": 0,
        "criteria_total": CRITERIA_TOTAL,
        "criteria_met_fraction": f"0/{CRITERIA_TOTAL}",
        "criteria_score_pct": 0.0,
        "paper_v6_section_3_drafted": False,
        "all_4_carnot_contributions_validated": False,
        "rho_C_curve_published_ready": False,
        "terminal_verdicts": {},
        "notable_successes": [],
        "failures_or_partials": [],
        "bottlenecks_identified": [],
        "carry_forward_gates_121": [],
        "retro_complete": False,
        "honest_verdict": "complete: in_progress_milestone_120_retro_seeded",
    }
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.append(exp_id)
        else:
            sources[exp_id] = payload
    return sources, missing


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status") or "").lower()


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict") or "")


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "complete"


def _is_blocked(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "blocked" or _verdict(payload).startswith("blocked")


def _number(payload: Mapping[str, Any], field: str) -> float | None:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _source_path(exp_id: str, field: str | None = None) -> str:
    path = f"results/{SOURCE_FILES[exp_id]}"
    return f"{path}:{field}" if field else path


def _criterion(
    *,
    key: str,
    tier: str,
    exp_id: str,
    target: str,
    fields: tuple[str, ...],
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    passed: bool,
    reason: str,
    caveats: tuple[str, ...] = (),
) -> dict[str, Any]:
    if exp_id in missing_source_ids or exp_id not in sources:
        return {
            "criterion": key,
            "tier": tier,
            "experiment_id": exp_id,
            "status": MISSING,
            "met": False,
            "target": target,
            "evidence_paths": [_source_path(exp_id)],
            "source_values": {"status": "missing", "honest_verdict": "missing_artifact"},
            "reason": f"{exp_id} source artifact is missing.",
            "caveats": [],
        }

    payload = sources[exp_id]
    status = MET if passed else BLOCKED if _is_blocked(payload) else NOT_MET
    source_values = {field: payload.get(field) for field in fields}
    source_values.update({"status": payload.get("status"), "honest_verdict": _verdict(payload)})
    return {
        "criterion": key,
        "tier": tier,
        "experiment_id": exp_id,
        "status": status,
        "met": status == MET,
        "target": target,
        "evidence_paths": [_source_path(exp_id, field) for field in fields],
        "source_values": source_values,
        "reason": "criterion satisfied" if status == MET else reason,
        "caveats": list(caveats),
    }


def _has_ci(payload: Mapping[str, Any], prefix: str) -> bool:
    estimate = _number(payload, f"{prefix}_estimate")
    lower = _number(payload, f"{prefix}_ci_lower")
    upper = _number(payload, f"{prefix}_ci_upper")
    return estimate is not None and lower is not None and upper is not None and lower <= estimate <= upper


def _activation_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1560", {})
    fields = (
        "activation_manifest_complete",
        "thrml_scaling_sweep_lineage_retired",
        "research_roadmap_yaml_modified",
        "scripts_research_conductor_modified",
    )
    passed = (
        _is_complete(payload)
        and payload.get("activation_manifest_complete") is True
        and payload.get("thrml_scaling_sweep_lineage_retired") is True
        and payload.get("research_roadmap_yaml_modified") is False
        and payload.get("scripts_research_conductor_modified") is False
    )
    return _criterion(
        key="activation",
        tier="activation",
        exp_id="exp1560",
        target=".120 activation manifest complete with protected files unchanged",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="activation manifest, retirement, or protected-file gate was not satisfied",
    )


def _kinetic_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1561", {})
    fields = (
        "kinetic_defense_in_depth_validated",
        "thrml_security_parity_with_single_site_gibbs",
        "p_n_at_k100_mh",
        "p_n_at_k100_single_site_gibbs",
        "p_n_at_k100_thrml_block_gibbs",
    )
    passed = (
        _is_complete(payload)
        and payload.get("kinetic_defense_in_depth_validated") is True
        and payload.get("thrml_security_parity_with_single_site_gibbs") is True
    )
    return _criterion(
        key="kinetic_defense",
        tier="Tier 1",
        exp_id="exp1561",
        target="THRML block-Gibbs preserves the kinetic defense-in-depth gate",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason=str(payload.get("falsification_note") or "kinetic defense gate was not validated"),
    )


def _brain_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1562", {})
    ratio = _number(payload, "factorized_vs_ar_ratio_at_k15") or 0.0
    best_kl = _number(payload, "best_parameterization_kl_at_k15")
    fields = (
        "brain_linear_ar_rescue_validated",
        "factorized_vs_ar_ratio_at_k15",
        "best_parameterization_kl_at_k15",
        "phase_3_recommendation",
    )
    passed = (
        _is_complete(payload)
        and payload.get("brain_linear_ar_rescue_validated") is True
        and ratio >= 10.0
        and best_kl is not None
        and best_kl <= 0.1
    )
    return _criterion(
        key="brain_linear_ar_rescue",
        tier="Tier 1",
        exp_id="exp1562",
        target="BRAIN+Linear-AR rescue validates >=10x widening at k=15 with KL <= 0.1",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="BRAIN+Linear-AR rescue gate was falsified or incomplete",
    )


def _specann_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1563", {})
    fields = ("architecture_record_updated", "spec_requirements_added", "exclusion_manifest_updated")
    passed = (
        _is_complete(payload)
        and payload.get("architecture_record_updated") is True
        and (payload.get("spec_requirements_added") or 0) >= 3
        and payload.get("exclusion_manifest_updated") is True
    )
    return _criterion(
        key="specann_rejection_record",
        tier="Tier 1",
        exp_id="exp1563",
        target="SpecAnn rejection is documented in architecture, spec, and exclusion manifest",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="SpecAnn rejection record was not fully documented",
    )


def _thrml_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1564", {})
    fields = (
        "thrml_vendoring_complete",
        "kl_to_thrml_after_vendoring",
        "candidate_warm_start_implemented",
        "regression_tests_passed",
        "focused_regression_tests_passed",
        "applicable_e2e_passed",
        "mirror_repo_url",
    )
    focused_or_full_tests = (
        payload.get("regression_tests_passed") is True
        or payload.get("focused_regression_tests_passed") is True
    )
    passed = (
        _is_complete(payload)
        and payload.get("thrml_vendoring_complete") is True
        and _number(payload, "kl_to_thrml_after_vendoring") == 0.0
        and payload.get("candidate_warm_start_implemented") is True
        and focused_or_full_tests
        and payload.get("applicable_e2e_passed") is True
        and bool(payload.get("mirror_repo_url"))
    )
    caveats = (
        ("full tests/python run did not complete in exp1564 source artifact",)
        if payload.get("regression_tests_passed") is False
        else ()
    )
    return _criterion(
        key="thrml_vendoring",
        tier="Tier 1",
        exp_id="exp1564",
        target="THRML is vendored with KL=0 parity, candidate warm-start, tests, E2E, and mirror URL",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="THRML vendoring or candidate warm-start gate was incomplete",
        caveats=caveats,
    )


def _soft_residual_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1565", {})
    fields = (
        "soft_gibbs_residual_implemented",
        "hard_brs_acceptance_rate",
        "soft_brs_decay_confirmed",
        "min_violation_state_found",
        "z_beta_curve",
    )
    passed = (
        _is_complete(payload)
        and payload.get("soft_gibbs_residual_implemented") is True
        and _number(payload, "hard_brs_acceptance_rate") == 0.0
        and payload.get("soft_brs_decay_confirmed") is True
        and payload.get("min_violation_state_found") is True
        and isinstance(payload.get("z_beta_curve"), list)
        and bool(payload.get("z_beta_curve"))
    )
    return _criterion(
        key="soft_gibbs_residual",
        tier="Tier 1",
        exp_id="exp1565",
        target="Soft-Gibbs Residual implemented and Hard-BRS empty-intersection failure shown",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="Soft-Gibbs Residual implementation gate was incomplete",
    )


def _candidate_warm_start_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1566", {})
    fields = (
        "candidate_warm_start_validated",
        "cold_start_accuracy_drop_percent_at_k100",
        "cached_state_worse_than_cold_start",
        "recommended_deployment_policy",
    )
    passed = (
        _is_complete(payload)
        and payload.get("candidate_warm_start_validated") is True
        and (_number(payload, "cold_start_accuracy_drop_percent_at_k100") or 0.0) >= 50.0
        and payload.get("cached_state_worse_than_cold_start") is True
        and payload.get("recommended_deployment_policy") == "candidate_warm_start"
    )
    return _criterion(
        key="candidate_warm_start",
        tier="Tier 2",
        exp_id="exp1566",
        target="candidate warm-start beats cold start and cached-state at K=100",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="candidate warm-start benchmark gate was incomplete",
    )


def _rho_criterion(sources: Mapping[str, Mapping[str, Any]], missing: set[str]) -> dict[str, Any]:
    payload = sources.get("exp1567", {})
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    s_r_star = float(metadata.get("s_r_star", 0.72))
    srs_accuracy = _number(payload, "srs_accepted_accuracy_at_C_above_C_inv")
    fields = (
        "rho_C_curve_fitted",
        "rho_C_r_squared",
        "C_star_estimate",
        "C_star_ci_lower",
        "C_star_ci_upper",
        "C_inv_estimate",
        "C_inv_ci_lower",
        "C_inv_ci_upper",
        "inversion_empirically_confirmed",
        "srs_accepted_accuracy_at_C_above_C_inv",
    )
    passed = (
        _is_complete(payload)
        and payload.get("rho_C_curve_fitted") is True
        and (_number(payload, "rho_C_r_squared") or 0.0) >= 0.9
        and _has_ci(payload, "C_star")
        and _has_ci(payload, "C_inv")
        and payload.get("inversion_empirically_confirmed") is True
        and srs_accuracy is not None
        and srs_accuracy < s_r_star
    )
    return _criterion(
        key="rho_C_curve",
        tier="Tier 2",
        exp_id="exp1567",
        target="rho(C) curve has R^2 >= 0.9, C*/C_inv confidence intervals, and inversion confirmation",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="rho(C) publication-readiness gate was incomplete",
    )


def _fr11_criterion(sources: Mapping[str, Mapping[str, Any]], missing: set[str]) -> dict[str, Any]:
    payload = sources.get("exp1568", {})
    fields = (
        "mode_collapse_audit_complete",
        "retained_policies_audited_count",
        "retained_policy_target_count",
        "retained_policy_target_met",
        "mode_collapse_confirmed_count",
        "reversal_recommended_count",
    )
    passed = (
        _is_complete(payload)
        and payload.get("mode_collapse_audit_complete") is True
        and (payload.get("retained_policies_audited_count") or 0) > 0
        and (payload.get("mode_collapse_confirmed_count") or 0) > 0
        and (payload.get("reversal_recommended_count") or 0) > 0
    )
    caveats = (
        ("retained policy target count was not met in exp1568 source artifact",)
        if payload.get("retained_policy_target_met") is False
        else ()
    )
    return _criterion(
        key="fr11_v14_retention_audit",
        tier="Tier 2",
        exp_id="exp1568",
        target="FR-11 v14 retained-policy audit completes and flags contaminated retentions",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="FR-11 retained-policy audit gate was incomplete",
        caveats=caveats,
    )


def _paper_draft_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1569", {})
    fields = (
        "section_3_drafted",
        "word_count",
        "subsections_completed",
        "all_4_carnot_contributions_present",
        "all_3_ruleouts_documented",
        "blocked_at_layer",
    )
    passed = (
        _is_complete(payload)
        and payload.get("section_3_drafted") is True
        and (payload.get("subsections_completed") or 0) >= 8
        and payload.get("all_4_carnot_contributions_present") is True
        and payload.get("all_3_ruleouts_documented") is True
    )
    return _criterion(
        key="paper_v6_section_3_draft",
        tier="Tier 2",
        exp_id="exp1569",
        target="paper-v6 Section 3 sampler draft is complete with eight subsections",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="paper-v6 Section 3 draft was blocked or incomplete",
    )


def _coverage_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1570", {})
    fields = (
        "alpha_i_per_verifier",
        "z_beta_jensen_bound",
        "z_beta_empirical",
        "jensen_bound_holds_for_all_beta",
        "optimal_beta_for_deployment",
    )
    passed = (
        _is_complete(payload)
        and isinstance(payload.get("alpha_i_per_verifier"), list)
        and len(payload.get("alpha_i_per_verifier", [])) == 6
        and isinstance(payload.get("z_beta_jensen_bound"), list)
        and bool(payload.get("z_beta_jensen_bound"))
        and isinstance(payload.get("z_beta_empirical"), list)
        and bool(payload.get("z_beta_empirical"))
        and payload.get("jensen_bound_holds_for_all_beta") is True
        and _number(payload, "optimal_beta_for_deployment") is not None
    )
    return _criterion(
        key="soft_gibbs_coverage_bound",
        tier="Tier 2",
        exp_id="exp1570",
        target="Soft-Gibbs Jensen coverage bound holds for all tested beta values",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="Soft-Gibbs coverage bound gate was incomplete",
    )


def _stepwise_criterion(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1571", {})
    fields = (
        "step_wise_baseline_implemented",
        "gradient_variance_reduction_factor",
        "convergence_rate_matches_theorem_2",
    )
    passed = (
        _is_complete(payload)
        and payload.get("step_wise_baseline_implemented") is True
        and (_number(payload, "gradient_variance_reduction_factor") or 0.0) >= 10.0
        and payload.get("convergence_rate_matches_theorem_2") is True
    )
    return _criterion(
        key="step_wise_ar_reinforce_baseline",
        tier="Tier 3",
        exp_id="exp1571",
        target="step-wise baseline reduces AR-REINFORCE variance by >=10x and preserves convergence",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="step-wise AR-REINFORCE baseline gate was incomplete",
    )


def _z1_criterion(sources: Mapping[str, Mapping[str, Any]], missing: set[str]) -> dict[str, Any]:
    payload = sources.get("exp1573", {})
    fields = (
        "z1_readiness_packet_updated",
        "thrml_vendor_mirroring_verified",
        "kv260_open_fpga_tracks_preserved",
        "kinetic_defense_z1_transfer_question_filed",
        "sampler_backend_protocol_documented",
        "blocked_at_layer",
    )
    passed = (
        _is_complete(payload)
        and payload.get("z1_readiness_packet_updated") is True
        and payload.get("thrml_vendor_mirroring_verified") is True
        and payload.get("kv260_open_fpga_tracks_preserved") is True
        and payload.get("kinetic_defense_z1_transfer_question_filed") is True
        and payload.get("sampler_backend_protocol_documented") is True
    )
    return _criterion(
        key="extropic_z1_readiness",
        tier="Tier 3",
        exp_id="exp1573",
        target="Extropic Z1 readiness packet is updated with THRML alignment and portability checks",
        fields=fields,
        sources=sources,
        missing_source_ids=missing,
        passed=passed,
        reason="Extropic Z1 readiness update was blocked or incomplete",
    )


def _retro_criterion() -> dict[str, Any]:
    return {
        "criterion": "retrospective",
        "tier": "Tier 3",
        "experiment_id": "exp1572",
        "status": MET,
        "met": True,
        "target": "Exp1572 records .120 criteria, verdicts, and .121 carry-forward gates",
        "evidence_paths": ["results/experiment_1572_milestone_120_retro.json"],
        "source_values": {"status": "complete", "honest_verdict": "self"},
        "reason": "criterion satisfied",
        "caveats": [],
    }


def evaluate_criteria(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: list[str] | set[str]
) -> dict[str, dict[str, Any]]:
    """REQ-REPORT-063: score every .120 criterion from source fields."""

    missing = set(missing_source_ids)
    criteria = {
        "activation": _activation_criterion(sources, missing),
        "kinetic_defense": _kinetic_criterion(sources, missing),
        "brain_linear_ar_rescue": _brain_criterion(sources, missing),
        "specann_rejection_record": _specann_criterion(sources, missing),
        "thrml_vendoring": _thrml_criterion(sources, missing),
        "soft_gibbs_residual": _soft_residual_criterion(sources, missing),
        "candidate_warm_start": _candidate_warm_start_criterion(sources, missing),
        "rho_C_curve": _rho_criterion(sources, missing),
        "fr11_v14_retention_audit": _fr11_criterion(sources, missing),
        "paper_v6_section_3_draft": _paper_draft_criterion(sources, missing),
        "soft_gibbs_coverage_bound": _coverage_criterion(sources, missing),
        "step_wise_ar_reinforce_baseline": _stepwise_criterion(sources, missing),
        "extropic_z1_readiness": _z1_criterion(sources, missing),
        "retrospective": _retro_criterion(),
    }
    return criteria


def _terminal_verdicts(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str], self_verdict: str
) -> dict[str, str]:
    verdicts = {
        exp_id: "MISSING"
        if exp_id in missing_source_ids or exp_id not in sources
        else _verdict(sources[exp_id]) or "NO_VERDICT"
        for exp_id in SOURCE_FILES
    }
    verdicts["exp1572"] = self_verdict
    return verdicts


def _slowest_experiments(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    durations = []
    for exp_id, payload in sources.items():
        duration = _number(payload, "duration_s")
        if duration is not None:
            durations.append({"experiment_id": exp_id, "duration_min": round(duration / 60.0, 2)})
    durations.sort(key=lambda item: (-item["duration_min"], item["experiment_id"]))
    return [{"rank": index + 1, **item} for index, item in enumerate(durations[:5])]


def _contribution_validation(criteria: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    return {
        "rho_C_curve": bool(criteria["rho_C_curve"]["met"]),
        "soft_gibbs_residual": bool(criteria["soft_gibbs_residual"]["met"]),
        "kinetic_defense_in_depth": bool(criteria["kinetic_defense"]["met"]),
        "brain_linear_ar_rescue": bool(criteria["brain_linear_ar_rescue"]["met"]),
    }


def _carry_forward_gates_121() -> list[dict[str, str]]:
    return [
        {
            "gate": "paper_v6_section_3_finalization_after_exp1569_draft",
            "reason": "exp1569 blocked at conductor prior-failure pre-gate; finalization waits for resumed draft.",
            "source": "exp1569",
        },
        {
            "gate": "specann_rejection_record_verification",
            "reason": "exp1563 wrote the rejection record; .121 should verify the record remains in roadmap and exclusion checks.",
            "source": "exp1563",
        },
        {
            "gate": "phase5_pcd_divergence_audit",
            "reason": "Deep Think DT-MCMC-K1 left Phase 5 PCD divergence as the next empirical sampler-risk check.",
            "source": "docs/research-notes/iclr26-deep-think-responses.md",
        },
        {
            "gate": "soft_gibbs_residual_production_scale_n128",
            "reason": "exp1565 and exp1570 are prototype/calibration results; production n=128 remains open.",
            "source": "exp1565+exp1570",
        },
        {
            "gate": "mcmc_layer_free_phase5_architecture",
            "reason": "SpecAnn, BRAIN expressivity, and kinetic-security findings require a Phase 5 architecture without MCMC Layers as an inference dependency.",
            "source": "exp1561+exp1562+exp1563",
        },
    ]


def _additional_carry_forwards() -> list[dict[str, str]]:
    return [
        {
            "gate": "extropic_z1_readiness_packet_resumed_after_exp1573",
            "reason": "exp1573 blocked at conductor prior-failure pre-gate and needs a corrected .121 carry-forward record.",
            "source": "exp1573",
        },
        {
            "gate": "brain_reinforce_training_dynamics_at_k15",
            "reason": "exp1562 falsified expressivity widening but did not test BRAIN REINFORCE training dynamics.",
            "source": "exp1562",
        },
        {
            "gate": "fr11_v15_lambda_grpo_retention_reversal",
            "reason": "exp1568 flagged one retained v14 policy for reversal under the lambda-GRPO fix path.",
            "source": "exp1568",
        },
    ]


def _notable_successes(criteria: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        f"{item['experiment_id']}:{name}"
        for name, item in criteria.items()
        if item["status"] == MET and name != "retrospective"
    ]


def _failures_or_partials(criteria: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "criterion": name,
            "experiment_id": item["experiment_id"],
            "status": item["status"],
            "reason": item["reason"],
            "caveats": item["caveats"],
        }
        for name, item in criteria.items()
        if item["status"] != MET or item["caveats"]
    ]


def _bottlenecks_identified() -> list[str]:
    return [
        "planner prior_failures placeholders blocked exp1569 and exp1573 before launch",
        "full tests/python instability remained a caveat on exp1564 even though focused sampler tests passed",
        "kinetic-security and BRAIN+Linear-AR acceptance gates were empirically falsified rather than validated",
        "Soft-Gibbs Residual still needs production-scale n=128 validation before broad paper claims",
    ]


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: list[str] | set[str]
) -> dict[str, Any]:
    criteria = evaluate_criteria(sources, missing_source_ids)
    criteria_met = sum(1 for item in criteria.values() if item["met"])
    criteria_total = len(criteria)
    contribution_validation = _contribution_validation(criteria)
    all_contributions_validated = all(contribution_validation.values())
    rho_ready = contribution_validation["rho_C_curve"]
    paper_drafted = bool(criteria["paper_v6_section_3_draft"]["met"])
    honest_verdict = (
        f"complete: milestone_120_{criteria_met}_of_{criteria_total}_criteria_met_"
        "paper_v6_exp1569_and_z1_exp1573_carried_to_121"
    )
    missing = set(missing_source_ids)
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "milestone": MILESTONE,
        "next_milestone": NEXT_MILESTONE,
        "criteria_results": criteria,
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
        "criteria_met_fraction": f"{criteria_met}/{criteria_total}",
        "criteria_score_pct": round(criteria_met / criteria_total * 100.0, 2),
        "paper_v6_section_3_drafted": paper_drafted,
        "all_4_carnot_contributions_validated": all_contributions_validated,
        "rho_C_curve_published_ready": rho_ready,
        "contribution_validation": contribution_validation,
        "terminal_verdicts": _terminal_verdicts(sources, missing, honest_verdict),
        "notable_successes": _notable_successes(criteria),
        "failures_or_partials": _failures_or_partials(criteria),
        "bottlenecks_identified": _bottlenecks_identified(),
        "carry_forward_gates_121": _carry_forward_gates_121(),
        "additional_carry_forwards_121": _additional_carry_forwards(),
        "slowest_experiments": _slowest_experiments(sources),
        "wall_time_minutes": round(
            sum(_number(payload, "duration_s") or 0.0 for payload in sources.values()) / 60.0,
            2,
        ),
        "wall_time_source": "source_artifact_duration_s_fields_only",
        "missing_source_ids": sorted(missing),
        "retro_complete": True,
        "required_artifact_fields_present": sorted(REQUIRED_ARTIFACT_FIELDS),
        "honest_verdict": honest_verdict,
    }


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-063: write bootstrap and terminal retrospective JSON."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing = _load_sources(root_path / "results")
    artifact = build_artifact(sources=sources, missing_source_ids=missing)
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
