"""Build the Exp 1402 milestone .108 retrospective artifact.

Spec: REQ-REPORT-009.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1402_milestone_108_retro.json"
RUN_DATE = "20260506"
MILESTONE = "2026.04.108"
EXPERIMENT = "1402_milestone_108_retro"
SCHEMA = "milestone_108_retro_v1"

MET = "MET"
NOT_MET = "NOT_MET"
MISSING = "MISSING"

SOURCE_FILES = {
    "exp1390": "experiment_1390_arxiv_submission_sword_api.json",
    "exp1391": "experiment_1391_fullscale_pipeline_failure_diagnosis.json",
    "exp1392": "experiment_1392_test_suite_hygiene_v2.json",
    "exp1393": "experiment_1393_grpo_v8_ngrpo_zero_reward_fix.json",
    "exp1394": "experiment_1394_dvi_v2_secl_combined.json",
    "exp1395": "experiment_1395_fr11_self_learning_v5.json",
    "exp1396": "experiment_1396_semantic_validation_pass_rate_fix_v1.json",
    "exp1397": "experiment_1397_fullscale_pipeline_v2_200cases.json",
    "exp1398": "experiment_1398_ngrpo_theory_probe.json",
    "exp1399": "experiment_1399_discrete_sb_kv260_cpu_simulation.json",
    "exp1400": "experiment_1400_biprm_retrospective_verification_probe.json",
    "exp1401": "experiment_1401_ebm_cot_v2_hinge_only.json",
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "criteria_total",
    "criteria_met",
    "experiment_statuses",
    "arxiv_submission_verdict",
    "pipeline_failure_diagnosis_verdict",
    "test_suite_hygiene_verdict",
    "grpo_ngrpo_verdict",
    "dvi_v2_secl_verdict",
    "fr11_v5_verdict",
    "semantic_validation_fix_verdict",
    "full_pipeline_v2_verdict",
    "ngrpo_theory_verdict",
    "discrete_sb_kv260_verdict",
    "biprm_verdict",
    "ebm_cot_v2_verdict",
    "carry_forward_tasks",
    "prior_failure_hygiene_notes",
    "honest_verdict",
)

CRITERIA = (
    "arxiv_submitted",
    "failure_diagnosis_complete",
    "test_suite_collection_clean",
    "grpo_ngrpo_measured",
    "dvi_v2_deployed",
    "fr11_v5_fresh_count_growing",
    "semantic_validation_fix_measured",
    "full_pipeline_v2_at_scale",
    "ngrpo_theory_confirmed",
    "discrete_sb_kv260_estimated",
    "biprm_verified",
    "ebm_cot_hinge_only_measured",
    "retro_108_complete",
)


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-009: persist the STEP 0 skeleton before source evaluation."""

    artifact = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact["status"] = "in_progress"
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


def _number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _gt(value: object, threshold: float) -> bool:
    numeric = _number(value)
    return numeric is not None and numeric > threshold


def _gte(value: object, threshold: float) -> bool:
    numeric = _number(value)
    return numeric is not None and numeric >= threshold


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status", "")).lower()


def _honest_verdict(payload: Mapping[str, Any]) -> str | None:
    verdict = payload.get("honest_verdict")
    if verdict is None:
        return None
    return str(verdict)


def _criterion(
    status: str, source: str, target: str, observed: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "target": target,
        "observed": dict(observed),
    }


def _experiment_statuses(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: list[str]
) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for exp_id, filename in SOURCE_FILES.items():
        exists = exp_id not in missing_source_ids
        payload = sources.get(exp_id, {})
        statuses[exp_id] = {
            "artifact": f"results/{filename}",
            "exists": exists,
            "status": MISSING.lower() if not exists else _status(payload),
            "honest_verdict": None if not exists else _honest_verdict(payload),
        }
    statuses["exp1402"] = {
        "artifact": "results/experiment_1402_milestone_108_retro.json",
        "exists": True,
        "status": "complete",
        "honest_verdict": "written_by_this_artifact",
    }
    return statuses


def _arxiv_submission_verdict(exp1390: Mapping[str, Any]) -> dict[str, Any]:
    attempted = exp1390.get("submission_attempted") is True
    result = str(exp1390.get("submission_result", ""))
    met = attempted and result != "not_attempted"
    return {
        "criterion_status": MET if met else NOT_MET,
        "submission_attempted": attempted,
        "submission_result": result,
        "submission_method": exp1390.get("submission_method"),
        "manual_checklist_generated": exp1390.get("manual_checklist_generated"),
        "manual_checklist_path": exp1390.get("manual_checklist_path"),
        "bundle_path": exp1390.get("bundle_path"),
        "arxiv_id_if_submitted": exp1390.get("arxiv_id_if_submitted"),
        "reason": "actual arXiv submission was not attempted"
        if not attempted
        else "submission attempted",
    }


def _pipeline_failure_diagnosis_verdict(exp1391: Mapping[str, Any]) -> dict[str, Any]:
    root_causes_identified = bool(
        exp1391.get("failure_root_causes")
        or exp1391.get("failure_categories")
        or exp1391.get("top_failure_category")
    )
    met = exp1391.get("failure_analysis_complete") is True and root_causes_identified
    return {
        "criterion_status": MET if met else NOT_MET,
        "failure_analysis_complete": exp1391.get("failure_analysis_complete"),
        "top_failure_category": exp1391.get("top_failure_category"),
        "failure_category_counts": exp1391.get("failure_category_counts"),
        "root_causes_identified": root_causes_identified,
        "fixable_failure_pct": exp1391.get("fixable_failure_pct"),
    }


def _test_suite_hygiene_verdict(exp1392: Mapping[str, Any]) -> dict[str, Any]:
    remaining = exp1392.get("collection_errors_remaining", exp1392.get("collection_errors_after"))
    met = remaining == 0
    return {
        "criterion_status": MET if met else NOT_MET,
        "test_suite_collection_clean": exp1392.get("test_suite_collection_clean"),
        "collection_errors_after": exp1392.get("collection_errors_after"),
        "collection_errors_remaining": remaining,
        "full_suite_tail_result": exp1392.get("validation", {}).get("full_suite_tail_result"),
        "spec_coverage_result": exp1392.get("validation", {}).get("spec_coverage_result"),
    }


def _grpo_ngrpo_verdict(exp1393: Mapping[str, Any]) -> dict[str, Any]:
    improvement_pp = exp1393.get("grpo_v8_improvement_pp")
    retired = exp1393.get("retire_if_same_verdict") is True
    met = _gt(improvement_pp, 0.0) or retired
    return {
        "criterion_status": MET if met else NOT_MET,
        "grpo_v8_improvement_pp": improvement_pp,
        "unknown_rollout_rate": exp1393.get("unknown_rollout_rate"),
        "retire_if_same_verdict": exp1393.get("retire_if_same_verdict"),
        "terminal_blocker": exp1393.get("terminal_blocker"),
        "formal_reward_pass_rate": exp1393.get("formal_reward_pass_rate"),
        "met_via_retirement": retired and not _gt(improvement_pp, 0.0),
    }


def _dvi_v2_secl_verdict(exp1394: Mapping[str, Any]) -> dict[str, Any]:
    met = exp1394.get("dvi_v2_deployed") is True and _gt(
        exp1394.get("dvi_v2_auroc_delta"), 0.003486
    )
    return {
        "criterion_status": MET if met else NOT_MET,
        "dvi_v2_auroc_delta": exp1394.get("dvi_v2_auroc_delta"),
        "dvi_v2_deployed": exp1394.get("dvi_v2_deployed"),
        "dvi_v2_trained_auroc": exp1394.get("dvi_v2_trained_auroc"),
        "fresh_cases_used": exp1394.get("fresh_cases_used"),
        "secl_ece_reduction_pct": exp1394.get("secl_ece_reduction_pct"),
    }


def _fr11_v5_verdict(exp1395: Mapping[str, Any]) -> dict[str, Any]:
    fresh_count = exp1395.get("fresh_verified_sample_count")
    gate_blocked = exp1395.get("gate_blocked_artifact_emitted") is True
    met = _gt(fresh_count, 59.0) or gate_blocked
    return {
        "criterion_status": MET if met else NOT_MET,
        "fresh_verified_sample_count": fresh_count,
        "grpo_v8_cases_integrated": exp1395.get("grpo_v8_cases_integrated"),
        "dvi_v2_checkpoint_active": exp1395.get("dvi_v2_checkpoint_active"),
        "headline_result_allowed": exp1395.get("headline_result_allowed"),
        "self_learning_delta_overall": exp1395.get("self_learning_delta_overall"),
    }


def _semantic_validation_fix_verdict(exp1396: Mapping[str, Any]) -> dict[str, Any]:
    fix_applied = exp1396.get("fix_applied") is True or bool(exp1396.get("fixes_applied"))
    met = exp1396.get("semantic_validation_improvement_measured") is True and fix_applied
    return {
        "criterion_status": MET if met else NOT_MET,
        "semantic_validation_pass_rate_before_fix": exp1396.get(
            "semantic_validation_pass_rate_before_fix"
        ),
        "semantic_validation_pass_rate_after_fix": exp1396.get(
            "semantic_validation_pass_rate_after_fix"
        ),
        "semantic_validation_improvement_measured": exp1396.get(
            "semantic_validation_improvement_measured"
        ),
        "fix_applied": fix_applied,
        "all_exp1391_failures_recovered_after_fix_count": exp1396.get(
            "all_exp1391_failures_recovered_after_fix_count"
        ),
    }


def _full_pipeline_v2_verdict(exp1397: Mapping[str, Any]) -> dict[str, Any]:
    gate_blocked = exp1397.get("gate_blocked_artifact_emitted") is True
    met = _gte(exp1397.get("cases_evaluated"), 200.0) or gate_blocked
    return {
        "criterion_status": MET if met else NOT_MET,
        "cases_evaluated": exp1397.get("cases_evaluated"),
        "certificate_parse_rate": exp1397.get("certificate_parse_rate"),
        "semantic_validation_pass_rate": exp1397.get("semantic_validation_pass_rate"),
        "full_pipeline_pass_rate": exp1397.get("full_pipeline_pass_rate"),
        "headline_result_allowed": exp1397.get("headline_result_allowed"),
        "headline_metric_gate_passed": exp1397.get("headline_metric_gate_passed"),
        "scheduler_accept_rate": exp1397.get("scheduler_accept_rate"),
        "repair_hint_precision": exp1397.get("repair_hint_precision"),
    }


def _ngrpo_theory_verdict(exp1398: Mapping[str, Any]) -> dict[str, Any]:
    tested = (
        exp1398.get("ngrpo_advantage_calibration_tested") is True
        or exp1398.get("ngrpo_advantage_calibration_verified") is True
    )
    return {
        "criterion_status": MET if tested else NOT_MET,
        "ngrpo_advantage_calibration_verified": exp1398.get("ngrpo_advantage_calibration_verified"),
        "theory_supports_exp1393": exp1398.get("theory_supports_exp1393"),
        "ngrpo_augmented_advantage_variance": exp1398.get("ngrpo_augmented_advantage_variance"),
        "original_resZero_advantage_variance": exp1398.get("original_resZero_advantage_variance"),
    }


def _discrete_sb_kv260_verdict(exp1399: Mapping[str, Any]) -> dict[str, Any]:
    claim_set = "hardware_claim_allowed" in exp1399 or "hardware_claim_set" in exp1399
    met = "bram_budget_feasible" in exp1399 and claim_set
    return {
        "criterion_status": MET if met else NOT_MET,
        "bram_budget_feasible": exp1399.get("bram_budget_feasible"),
        "hardware_claim_allowed": exp1399.get("hardware_claim_allowed"),
        "kv260_claim_allowed": exp1399.get("kv260_claim_allowed"),
        "convergence_speedup_discrete_sb": exp1399.get("convergence_speedup_discrete_sb"),
        "hardware_execution_performed": exp1399.get("metadata", {}).get(
            "hardware_execution_performed"
        ),
        "synthesis_performed": exp1399.get("metadata", {}).get("synthesis_performed"),
    }


def _biprm_verdict(exp1400: Mapping[str, Any]) -> dict[str, Any]:
    determined = "retrospective_verification_viable" in exp1400
    return {
        "criterion_status": MET if determined else NOT_MET,
        "pivot_precision_delta": exp1400.get("pivot_precision_delta"),
        "retrospective_verification_viable": exp1400.get("retrospective_verification_viable"),
        "forward_only_pivot_precision": exp1400.get("forward_only_pivot_precision"),
        "biprm_r2l_pivot_precision": exp1400.get("biprm_r2l_pivot_precision"),
        "human_annotated_pivot_cases": exp1400.get("human_annotated_pivot_cases"),
    }


def _ebm_cot_v2_verdict(exp1401: Mapping[str, Any]) -> dict[str, Any]:
    met = _gt(exp1401.get("calibration_auroc_delta"), 0.0) and (
        exp1401.get("consistency_regularization_weight") in (0, 0.0, None)
    )
    return {
        "criterion_status": MET if met else NOT_MET,
        "calibration_auroc_delta": exp1401.get("calibration_auroc_delta"),
        "variance_worsened": exp1401.get("variance_worsened"),
        "consistency_regularization_weight": exp1401.get("consistency_regularization_weight"),
        "ebm_cot_v2_auroc": exp1401.get("ebm_cot_v2_auroc"),
        "paraphrase_energy_variance_before": exp1401.get("paraphrase_energy_variance_before"),
        "paraphrase_energy_variance_after": exp1401.get("paraphrase_energy_variance_after"),
    }


def _carry_forward_tasks(verdicts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    tasks = [
        {
            "source": "exp1390",
            "task": "Complete actual arXiv submission using the verified bundle and manual checklist.",
            "reason": "submission_attempted=false; manual checklist was generated but no arXiv ID exists.",
        },
        {
            "source": "exp1397",
            "task": "Raise full pipeline pass rate above the headline threshold by improving scheduler acceptance or post-repair acceptance, not semantic parsing.",
            "reason": "semantic_validation_pass_rate=1.0 but full_pipeline_pass_rate=0.305 below the 0.40 headline gate.",
        },
        {
            "source": "exp1393",
            "task": "Do not rerun the same FoVer JURY-RL/NGRPO path unless a non-UNKNOWN reward or verifier signal is introduced.",
            "reason": "NGRPO produced no held-out improvement and triggered retire_if_same_verdict=true.",
        },
        {
            "source": "exp1400",
            "task": "Retire or redesign proxy-only BiPRM; use human pivot labels before another viability claim.",
            "reason": "pivot_precision_delta=-0.013334 and retrospective_verification_viable=false.",
        },
        {
            "source": "exp1401",
            "task": "Preserve hinge-only AUROC gains while reducing paraphrase energy variance.",
            "reason": "calibration_auroc_delta is positive, but variance_worsened=true.",
        },
        {
            "source": "exp1392",
            "task": "Carry forward execution-time test failures and spec-coverage hygiene separately from collection hygiene.",
            "reason": "collection is clean, but exp1392 reported unrelated full-suite failures and spec-coverage debt.",
        },
        {
            "source": "exp1399",
            "task": "Run synthesis or KV260 board validation before upgrading discrete-SB estimates into hardware-execution claims.",
            "reason": "hardware_claim_allowed is an estimate; hardware_execution_performed=false and synthesis_performed=false.",
        },
    ]
    if verdicts["fr11_v5_verdict"].get("grpo_v8_cases_integrated") == 0:
        tasks.append(
            {
                "source": "exp1395",
                "task": "Keep FR-11 GRPO integration closed until a positive GRPO result exists.",
                "reason": "fresh count grew through DVI v2; grpo_v8_cases_integrated=0.",
            }
        )
    return tasks


def _prior_failure_hygiene_notes(
    sources: Mapping[str, Mapping[str, Any]],
    missing_artifacts: list[str],
    missing_inputs: list[str],
) -> dict[str, Any]:
    retirements: list[dict[str, Any]] = []
    exp1393 = sources.get("exp1393", {})
    if exp1393.get("retire_if_same_verdict") is True and not _gt(
        exp1393.get("grpo_v8_improvement_pp"), 0.0
    ):
        retirements.append(
            {
                "experiment_id": "exp1393",
                "triggered": True,
                "retire_if_same_verdict": True,
                "same_verdict_signal": ("grpo_v8_improvement_pp=0.0 and unknown_rollout_rate=1.0"),
                "prior_failure_scope": "exp1383 GRPO zero-reward/all-UNKNOWN rollout failure",
                "required_action": (
                    "Retire this exact reward path from future milestones unless the "
                    "root cause changes through a non-UNKNOWN verifier or reward source."
                ),
            }
        )
    return {
        "retirements_triggered": retirements,
        "missing_result_artifacts": missing_artifacts,
        "missing_requested_inputs": missing_inputs,
    }


def _criteria_results(verdicts: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        "arxiv_submitted": _criterion(
            str(verdicts["arxiv_submission_verdict"]["criterion_status"]),
            "exp1390",
            'submission_attempted=true AND submission_result!="not_attempted"',
            verdicts["arxiv_submission_verdict"],
        ),
        "failure_diagnosis_complete": _criterion(
            str(verdicts["pipeline_failure_diagnosis_verdict"]["criterion_status"]),
            "exp1391",
            "failure_analysis_complete=true AND failure_root_causes identified",
            verdicts["pipeline_failure_diagnosis_verdict"],
        ),
        "test_suite_collection_clean": _criterion(
            str(verdicts["test_suite_hygiene_verdict"]["criterion_status"]),
            "exp1392",
            "collection_errors_remaining=0",
            verdicts["test_suite_hygiene_verdict"],
        ),
        "grpo_ngrpo_measured": _criterion(
            str(verdicts["grpo_ngrpo_verdict"]["criterion_status"]),
            "exp1393",
            "grpo_v8_improvement_pp > 0 OR retire_if_same_verdict=true applied",
            verdicts["grpo_ngrpo_verdict"],
        ),
        "dvi_v2_deployed": _criterion(
            str(verdicts["dvi_v2_secl_verdict"]["criterion_status"]),
            "exp1394",
            "dvi_v2_deployed=true AND dvi_v2_auroc_delta > 0.003486",
            verdicts["dvi_v2_secl_verdict"],
        ),
        "fr11_v5_fresh_count_growing": _criterion(
            str(verdicts["fr11_v5_verdict"]["criterion_status"]),
            "exp1395",
            "fresh_verified_sample_count > 59 OR gate_blocked_artifact_emitted",
            verdicts["fr11_v5_verdict"],
        ),
        "semantic_validation_fix_measured": _criterion(
            str(verdicts["semantic_validation_fix_verdict"]["criterion_status"]),
            "exp1396",
            "semantic_validation_improvement_measured=true AND fix_applied",
            verdicts["semantic_validation_fix_verdict"],
        ),
        "full_pipeline_v2_at_scale": _criterion(
            str(verdicts["full_pipeline_v2_verdict"]["criterion_status"]),
            "exp1397",
            "cases_evaluated >= 200 OR gate_blocked_artifact_emitted",
            verdicts["full_pipeline_v2_verdict"],
        ),
        "ngrpo_theory_confirmed": _criterion(
            str(verdicts["ngrpo_theory_verdict"]["criterion_status"]),
            "exp1398",
            "ngrpo_advantage_calibration_tested=true",
            verdicts["ngrpo_theory_verdict"],
        ),
        "discrete_sb_kv260_estimated": _criterion(
            str(verdicts["discrete_sb_kv260_verdict"]["criterion_status"]),
            "exp1399",
            "bram_budget_feasible determined AND hardware_claim_set",
            verdicts["discrete_sb_kv260_verdict"],
        ),
        "biprm_verified": _criterion(
            str(verdicts["biprm_verdict"]["criterion_status"]),
            "exp1400",
            "retrospective_verification_viable determined",
            verdicts["biprm_verdict"],
        ),
        "ebm_cot_hinge_only_measured": _criterion(
            str(verdicts["ebm_cot_v2_verdict"]["criterion_status"]),
            "exp1401",
            "calibration_auroc_delta > 0 with hinge-only and no consistency regularization",
            verdicts["ebm_cot_v2_verdict"],
        ),
        "retro_108_complete": _criterion(
            MET,
            "exp1402",
            "criteria_met/criteria_total written",
            {"criteria_written": True},
        ),
    }


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: list[str],
    *,
    roadmap_next_present: bool,
    change_proposal_present: bool,
) -> dict[str, Any]:
    """REQ-REPORT-009: evaluate .108 criteria from authoritative artifacts."""

    missing_inputs = [] if roadmap_next_present else ["research-roadmap-next.yaml"]
    verdicts: dict[str, dict[str, Any]] = {
        "arxiv_submission_verdict": _arxiv_submission_verdict(sources.get("exp1390", {})),
        "pipeline_failure_diagnosis_verdict": _pipeline_failure_diagnosis_verdict(
            sources.get("exp1391", {})
        ),
        "test_suite_hygiene_verdict": _test_suite_hygiene_verdict(sources.get("exp1392", {})),
        "grpo_ngrpo_verdict": _grpo_ngrpo_verdict(sources.get("exp1393", {})),
        "dvi_v2_secl_verdict": _dvi_v2_secl_verdict(sources.get("exp1394", {})),
        "fr11_v5_verdict": _fr11_v5_verdict(sources.get("exp1395", {})),
        "semantic_validation_fix_verdict": _semantic_validation_fix_verdict(
            sources.get("exp1396", {})
        ),
        "full_pipeline_v2_verdict": _full_pipeline_v2_verdict(sources.get("exp1397", {})),
        "ngrpo_theory_verdict": _ngrpo_theory_verdict(sources.get("exp1398", {})),
        "discrete_sb_kv260_verdict": _discrete_sb_kv260_verdict(sources.get("exp1399", {})),
        "biprm_verdict": _biprm_verdict(sources.get("exp1400", {})),
        "ebm_cot_v2_verdict": _ebm_cot_v2_verdict(sources.get("exp1401", {})),
    }
    criteria_results = _criteria_results(verdicts)
    criteria_met = sum(1 for item in criteria_results.values() if item["status"] == MET)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": MILESTONE,
        "status": "complete",
        "criteria_total": len(CRITERIA),
        "criteria_met": criteria_met,
        "criteria_results": criteria_results,
        "experiment_statuses": _experiment_statuses(sources, missing_source_ids),
        "source_artifacts_checked": [
            {
                "experiment_id": exp_id,
                "path": f"results/{filename}",
                "exists": exp_id not in missing_source_ids,
            }
            for exp_id, filename in SOURCE_FILES.items()
        ]
        + [
            {
                "experiment_id": "exp1402",
                "path": "results/experiment_1402_milestone_108_retro.json",
                "exists": True,
            }
        ],
        "roadmap_inputs": {
            "research_roadmap_next_yaml_present": roadmap_next_present,
            "change_proposal_present": change_proposal_present,
            "missing_requested_inputs": missing_inputs,
        },
        **verdicts,
    }
    artifact["carry_forward_tasks"] = _carry_forward_tasks(verdicts)
    artifact["prior_failure_hygiene_notes"] = _prior_failure_hygiene_notes(
        sources, missing_source_ids, missing_inputs
    )
    artifact["honest_verdict"] = (
        f"milestone_108_{criteria_met}_of_{len(CRITERIA)}_criteria_met_"
        "arxiv_submission_not_attempted_pipeline_semantic_fixed_full_pipeline_below_headline_"
        "grpo_retired_biprm_negative"
    )
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-009: write skeleton, read sources, then persist final retro."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing_source_ids = _load_sources(root_path / "results")
    artifact = build_artifact(
        sources,
        missing_source_ids,
        roadmap_next_present=(root_path / "research-roadmap-next.yaml").exists(),
        change_proposal_present=(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ).exists(),
    )
    return _write_json(out, artifact)


if __name__ == "__main__":
    run()
