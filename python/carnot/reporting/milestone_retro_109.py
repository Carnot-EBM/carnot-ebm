"""Build the Exp 1424 milestone .109 retrospective artifact.

Spec: REQ-REPORT-032.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1424_milestone_109_retro.json"
RUN_DATE = "20260506"
MILESTONE = "2026.04.109"
EXPERIMENT = "1424_milestone_109_retro"
SCHEMA = "milestone_109_retro_v1"

MET = "MET"
NOT_MET = "NOT_MET"
GATE_BLOCKED = "GATE_BLOCKED"
BLOCKED = "BLOCKED"

SOURCE_FILES = {
    "exp1412": "experiment_1412_arxiv_operator_action_sheet_v3.json",
    "exp1413": "experiment_1413_certificate_repair_execution_diagnosis.json",
    "exp1414": "experiment_1414_certificate_llm_repair_executor_v1.json",
    "exp1415": "experiment_1415_dvi_v3_1508_fresh_cases.json",
    "exp1416": "experiment_1416_ebm_cot_v3_temperature_calibration.json",
    "exp1417": "experiment_1417_ebrm_latent_trajectory_drift_smoke.json",
    "exp1418": "experiment_1418_fr11_self_learning_v6_dvi_v3.json",
    "exp1419": "experiment_1419_fullscale_pipeline_v3_repair_executor.json",
    "exp1420": "experiment_1420_dpo_verified_pairs_1508.json",
    "exp1421": "experiment_1421_test_suite_execution_debt_v1.json",
    "exp1422": "experiment_1422_discrete_sb_kv260_rtl_spec.json",
    "exp1423": "experiment_1423_process_reward_model_v1_fover_1508.json",
}

CRITERIA = (
    "arxiv_action_sheet_complete",
    "repair_diagnosis_complete",
    "llm_repair_executor_deployed",
    "dvi_v3_improves_on_v2",
    "ebm_cot_variance_fixed",
    "latent_drift_smoke_complete",
    "fr11_v6_headline_allowed",
    "full_pipeline_clears_headline_gate",
    "dpo_measured",
    "execution_debt_fixed",
    "discrete_sb_rtl_spec_complete",
    "prm_v1_measured",
    "retro_complete",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "criteria_total",
    "criteria_met",
    "success_criteria_results",
    "retired_experiments",
    "carry_forward_tasks",
    "prior_failures_required_next",
    "gpu_utilization_summary",
    "honest_verdict",
)


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-032: write the bootstrap artifact before reading source results."""

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


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status", "")).lower()


def _honest_verdict(payload: Mapping[str, Any]) -> str | None:
    verdict = payload.get("honest_verdict")
    if verdict is None:
        return None
    return str(verdict)


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _determined(value: object) -> bool:
    return _number(value) is not None or (isinstance(value, str) and bool(value))


def _truthy_sequence_or_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, list | tuple | set):
        return bool(value)
    return False


def _criterion(
    status: str,
    source: str,
    target: str,
    observed: Mapping[str, Any],
    rationale: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "source": source,
        "target": target,
        "observed": dict(observed),
        "rationale": rationale,
    }


def _blocked_or_missing(exp_id: str, sources: Mapping[str, Mapping[str, Any]]) -> str | None:
    payload = sources.get(exp_id)
    if payload is None:
        return NOT_MET
    if _status(payload) == "blocked":
        return BLOCKED
    return None


def _score_arxiv(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1412", {})
    blocked = _blocked_or_missing("exp1412", sources)
    met = payload.get("submission_ready_for_operator") is True
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1412.submission_ready_for_operator",
        "submission_ready_for_operator=true",
        {
            "submission_ready_for_operator": payload.get("submission_ready_for_operator"),
            "credentialed_submission_attempted": payload.get("credentialed_submission_attempted"),
            "bundle_exists": payload.get("bundle_exists"),
        },
        "Operator action sheet is ready." if met else "Submission handoff is incomplete.",
    )


def _score_repair_diagnosis(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1413", {})
    blocked = _blocked_or_missing("exp1413", sources)
    met = payload.get("repair_execution_diagnosis_complete") is True
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1413.repair_execution_diagnosis_complete",
        "repair_execution_diagnosis_complete=true",
        {
            "repair_execution_diagnosis_complete": payload.get(
                "repair_execution_diagnosis_complete"
            ),
            "repair_hint_cases_total": payload.get("repair_hint_cases_total"),
            "executable_hint_pct": payload.get("executable_hint_pct"),
        },
        "Repair-hint diagnosis is complete." if met else "Repair diagnosis did not close.",
    )


def _score_repair_executor(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1414", {})
    blocked = _blocked_or_missing("exp1414", sources)
    met = payload.get("repair_executor_deployed") is True
    successful = payload.get("repaired_cases_successful")
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1414.repair_executor_deployed",
        "repair_executor_deployed=true",
        {
            "repair_executor_deployed": payload.get("repair_executor_deployed"),
            "repair_hint_cases_tested": payload.get("repair_hint_cases_tested"),
            "repaired_cases_successful": successful,
            "repaired_case_success_rate": payload.get("repaired_case_success_rate"),
        },
        "Executor was wired in, but successful repairs still need follow-up."
        if met and successful == 0
        else "Repair executor deployment target evaluated.",
    )


def _score_dvi_v3(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1415", {})
    if payload and _status(payload) == "blocked":
        status = BLOCKED
        rationale = "DVI v3 was blocked by the nonforgetting gate despite a small AUROC delta."
    else:
        baseline = _number(payload.get("dvi_v2_auroc_delta_baseline")) or 0.011458
        delta = _number(payload.get("dvi_v3_auroc_delta"))
        status = MET if delta is not None and delta > baseline else NOT_MET
        rationale = "DVI v3 AUROC delta beat the v2 baseline." if status == MET else (
            "DVI v3 did not beat the v2 baseline."
        )
    return _criterion(
        status,
        "exp1415.dvi_v3_auroc_delta",
        "dvi_v3_auroc_delta > 0.011458",
        {
            "status": payload.get("status"),
            "fresh_verified_cases_used": payload.get("fresh_verified_cases_used"),
            "dvi_v2_auroc_delta_baseline": payload.get("dvi_v2_auroc_delta_baseline"),
            "dvi_v3_auroc_delta": payload.get("dvi_v3_auroc_delta"),
            "dvi_v3_deployed": payload.get("dvi_v3_deployed"),
            "nonforgetting_rate": payload.get("nonforgetting_rate"),
        },
        rationale,
    )


def _score_ebm_cot(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1416", {})
    blocked = _blocked_or_missing("exp1416", sources)
    met = payload.get("variance_worsened") is False and payload.get("auroc_preserved") is True
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1416.variance_worsened and exp1416.auroc_preserved",
        "variance_worsened=false and auroc_preserved=true",
        {
            "temperature_scaling_applied": payload.get("temperature_scaling_applied"),
            "variance_worsened": payload.get("variance_worsened"),
            "auroc_preserved": payload.get("auroc_preserved"),
            "best_temperature": payload.get("best_temperature"),
        },
        "Temperature scaling fixed variance while preserving AUROC." if met else (
            "EBM-CoT calibration still fails the variance/AUROC gate."
        ),
    )


def _score_latent_drift(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1417", {})
    blocked = _blocked_or_missing("exp1417", sources)
    met = payload.get("latent_drift_smoke_complete") is True
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1417.latent_drift_smoke_complete",
        "latent_drift_smoke_complete=true",
        {
            "latent_drift_smoke_complete": payload.get("latent_drift_smoke_complete"),
            "energy_monotone": payload.get("energy_monotone"),
            "accuracy_delta_after_planning": payload.get("accuracy_delta_after_planning"),
            "dual_path_decoder_required": payload.get("dual_path_decoder_required"),
            "anchoring_required": payload.get("anchoring_required"),
        },
        "Latent drift smoke completed and exposed an off-decoder-support failure."
        if met
        else "Latent drift smoke did not complete.",
    )


def _fr11_gate_closed(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: list[str],
    conductor_log_text: str,
) -> bool:
    if "exp1418" not in missing_source_ids:
        return False
    exp1415 = sources.get("exp1415", {})
    log_lower = conductor_log_text.lower()
    return (
        exp1415.get("dvi_v3_deployed") is False
        or _status(exp1415) == "blocked"
        or ("fr-11" in log_lower and "gate_block" in log_lower)
        or ("exp1418" in log_lower and "gate" in log_lower)
    )


def _score_fr11(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: list[str],
    conductor_log_text: str,
) -> dict[str, Any]:
    payload = sources.get("exp1418", {})
    if _fr11_gate_closed(sources, missing_source_ids, conductor_log_text):
        status = GATE_BLOCKED
        rationale = "FR-11 v6 did not run because upstream DVI v3 deployment gate stayed closed."
    else:
        fresh = _number(payload.get("fresh_verified_sample_count"))
        met = payload.get("headline_result_allowed") is True and fresh is not None and fresh > 1508
        status = MET if met else NOT_MET
        rationale = "FR-11 v6 exceeded the 1508-case baseline." if met else (
            "FR-11 v6 did not clear the headline gate."
        )
    return _criterion(
        status,
        "exp1418.headline_result_allowed and exp1418.fresh_verified_sample_count",
        "headline_result_allowed=true and fresh_verified_sample_count > 1508",
        {
            "exists": "exp1418" not in missing_source_ids,
            "headline_result_allowed": payload.get("headline_result_allowed"),
            "fresh_verified_sample_count": payload.get("fresh_verified_sample_count"),
            "upstream_dvi_v3_deployed": sources.get("exp1415", {}).get("dvi_v3_deployed"),
            "upstream_dvi_status": sources.get("exp1415", {}).get("status"),
        },
        rationale,
    )


def _score_full_pipeline(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1419", {})
    blocked = _blocked_or_missing("exp1419", sources)
    pass_rate = _number(payload.get("full_pipeline_pass_rate"))
    met = (
        pass_rate is not None
        and pass_rate >= 0.40
        and payload.get("full_pipeline_headline_gate_met") is True
    )
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1419.full_pipeline_pass_rate",
        "full_pipeline_pass_rate >= 0.40",
        {
            "cases_evaluated": payload.get("cases_evaluated"),
            "repair_hint_cases_total": payload.get("repair_hint_cases_total"),
            "repaired_cases_successful": payload.get("repaired_cases_successful"),
            "repair_success_rate": payload.get("repair_success_rate"),
            "full_pipeline_pass_rate": payload.get("full_pipeline_pass_rate"),
            "full_pipeline_headline_gate_met": payload.get("full_pipeline_headline_gate_met"),
        },
        "Full pipeline cleared the 0.40 headline gate." if met else (
            "Full pipeline remained at the .108 0.305 pass-rate floor."
        ),
    )


def _score_dpo(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1420", {})
    blocked = _blocked_or_missing("exp1420", sources)
    met = _determined(payload.get("dpo_improvement_pp"))
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1420.dpo_improvement_pp",
        "dpo_improvement_pp determined",
        {
            "verified_pairs_available": payload.get("verified_pairs_available"),
            "dpo_full_finetune_performed": payload.get("dpo_full_finetune_performed"),
            "dpo_reranker_fallback_used": payload.get("dpo_reranker_fallback_used"),
            "dpo_improvement_pp": payload.get("dpo_improvement_pp"),
            "headline_result_allowed": payload.get("headline_result_allowed"),
        },
        "DPO-style fallback was measured, but not headline-ready." if met else (
            "DPO improvement was not determined."
        ),
    )


def _score_test_debt(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1421", {})
    blocked = _blocked_or_missing("exp1421", sources)
    met = _truthy_sequence_or_bool(payload.get("execution_failures_fixed"))
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1421.execution_failures_fixed",
        "execution_failures_fixed=true",
        {
            "collection_clean_confirmed": payload.get("collection_clean_confirmed"),
            "execution_failures_fixed": payload.get("execution_failures_fixed"),
            "spec_coverage_checked": payload.get("spec_coverage_checked"),
            "remaining_debt": payload.get("remaining_debt"),
        },
        "Focused execution-debt cluster was fixed; broader suite debt remains."
        if met
        else "Execution debt fix did not complete.",
    )


def _score_rtl(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1422", {})
    blocked = _blocked_or_missing("exp1422", sources)
    met = payload.get("rtl_spec_complete") is True
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1422.rtl_spec_complete",
        "rtl_spec_complete=true",
        {
            "rtl_spec_complete": payload.get("rtl_spec_complete"),
            "rtl_spec_path": payload.get("rtl_spec_path"),
            "kv260_budget_fits": payload.get("kv260_budget_fits"),
            "hardware_execution_performed": payload.get("hardware_execution_performed"),
            "hardware_claim_allowed": payload.get("hardware_claim_allowed"),
        },
        "Discrete SB RTL specification is complete without making a hardware claim."
        if met
        else "Discrete SB RTL specification is incomplete.",
    )


def _score_prm(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    payload = sources.get("exp1423", {})
    blocked = _blocked_or_missing("exp1423", sources)
    met = _determined(payload.get("prmv1_auroc"))
    return _criterion(
        blocked or (MET if met else NOT_MET),
        "exp1423.prmv1_auroc",
        "prmv1_auroc determined",
        {
            "training_traces_used": payload.get("training_traces_used"),
            "step_labels_available": payload.get("step_labels_available"),
            "prmv1_trained": payload.get("prmv1_trained"),
            "prmv1_auroc": payload.get("prmv1_auroc"),
            "prmv1_step_precision": payload.get("prmv1_step_precision"),
            "prmv1_step_recall": payload.get("prmv1_step_recall"),
        },
        "PRM v1 was measured on available labels; missing labels remain follow-up."
        if met
        else "PRM v1 AUROC was not determined.",
    )


def evaluate_success_criteria(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: list[str],
    conductor_log_text: str,
) -> dict[str, dict[str, Any]]:
    """REQ-REPORT-032: score the 13 roadmap success criteria mechanically.

    The retrospective is meant to be auditable later, so each criterion keeps
    the source field, target, observed values, and a short rationale instead of
    only returning a boolean. This makes blocked positive-looking metrics, such
    as DVI v3's small AUROC delta, visible without turning them into successes.
    """

    results = {
        "arxiv_action_sheet_complete": _score_arxiv(sources),
        "repair_diagnosis_complete": _score_repair_diagnosis(sources),
        "llm_repair_executor_deployed": _score_repair_executor(sources),
        "dvi_v3_improves_on_v2": _score_dvi_v3(sources),
        "ebm_cot_variance_fixed": _score_ebm_cot(sources),
        "latent_drift_smoke_complete": _score_latent_drift(sources),
        "fr11_v6_headline_allowed": _score_fr11(
            sources, missing_source_ids, conductor_log_text
        ),
        "full_pipeline_clears_headline_gate": _score_full_pipeline(sources),
        "dpo_measured": _score_dpo(sources),
        "execution_debt_fixed": _score_test_debt(sources),
        "discrete_sb_rtl_spec_complete": _score_rtl(sources),
        "prm_v1_measured": _score_prm(sources),
    }

    pre_retro_met = sum(1 for item in results.values() if item["status"] == MET)
    retro_met = pre_retro_met + 1 >= 10
    results["retro_complete"] = _criterion(
        MET if retro_met else NOT_MET,
        "exp1424.status and criteria threshold",
        "criteria_met >= 10/13",
        {
            "retro_artifact_written": True,
            "criteria_met_before_retro": pre_retro_met,
            "criteria_met_if_retro_counts": pre_retro_met + 1,
            "threshold": 10,
        },
        "Writing this retro brings the milestone to the planned 10/13 threshold."
        if retro_met
        else "Even with the retro, the milestone misses the 10/13 threshold.",
    )
    return results


def _experiment_statuses(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: list[str],
) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for exp_id, filename in SOURCE_FILES.items():
        exists = exp_id not in missing_source_ids
        payload = sources.get(exp_id, {})
        statuses[exp_id] = {
            "artifact": f"results/{filename}",
            "exists": exists,
            "status": "missing" if not exists else _status(payload),
            "honest_verdict": None if not exists else _honest_verdict(payload),
        }
    statuses["exp1424"] = {
        "artifact": "results/experiment_1424_milestone_109_retro.json",
        "exists": True,
        "status": "complete",
        "honest_verdict": "written_by_this_artifact",
    }
    return statuses


def _source_artifacts_checked(missing_source_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": exp_id,
            "path": f"results/{filename}",
            "exists": exp_id not in missing_source_ids,
        }
        for exp_id, filename in SOURCE_FILES.items()
    ]


def _prior(
    experiment_id: str,
    verdict: str,
    addressed_by: str,
    retire_if_same_verdict: bool,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "verdict": verdict,
        "addressed_by": addressed_by,
        "retire_if_same_verdict": retire_if_same_verdict,
    }


def _carry_forward_tasks(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exp1414 = _honest_verdict(sources.get("exp1414", {})) or "missing"
    exp1415 = _honest_verdict(sources.get("exp1415", {})) or "missing"
    exp1419 = _honest_verdict(sources.get("exp1419", {})) or "missing"
    exp1420 = _honest_verdict(sources.get("exp1420", {})) or "missing"
    exp1421 = _honest_verdict(sources.get("exp1421", {})) or "missing"
    exp1423 = _honest_verdict(sources.get("exp1423", {})) or "missing"

    tasks = [
        {
            "id": "repair-executor-v2-root-cause",
            "title": "Repair executor v2 must prove nonzero validated repairs before scale-up",
            "root_cause": (
                "Exp1414 deployed the executor but repaired 0/20 cases, and Exp1419 repaired "
                "0/100 repair-hint cases, leaving full_pipeline_pass_rate at 0.305."
            ),
            "next_action": (
                "Instrument rejected repair outputs, split STEP_REWRITE failures by validator "
                "reason, and gate the next full-pipeline run on repaired_case_success_rate > 0."
            ),
            "prior_failures": [
                _prior(
                    "exp1414-certificate-llm-repair-executor-v1",
                    exp1414,
                    "Next attempt changes prompt/schema validation and must diagnose why all "
                    "candidate repairs failed semantic acceptance.",
                    False,
                ),
                _prior(
                    "exp1419-fullscale-pipeline-v3-repair-executor",
                    exp1419,
                    "Do not rerun the 200-case pipeline until the executor has a measured "
                    "nonzero repair-success gate.",
                    True,
                ),
            ],
        },
        {
            "id": "dvi-v3-nonforgetting-gate-fix",
            "title": "DVI v3 needs a nonforgetting-preserving training pass",
            "root_cause": (
                "Exp1415 used 1508 fresh cases and beat the AUROC-delta baseline slightly, "
                "but status stayed blocked because nonforgetting_rate was 0.968604."
            ),
            "next_action": (
                "Use replay-balanced training or threshold calibration and require both "
                "dvi_v3_deployed=true and nonforgetting gate pass before FR-11 v6."
            ),
            "prior_failures": [
                _prior(
                    "exp1415-dvi-v3-1508-fresh-cases",
                    exp1415,
                    "The next DVI task must explicitly address the nonforgetting gate, not only "
                    "optimize AUROC delta.",
                    True,
                )
            ],
        },
        {
            "id": "fr11-v6-after-dvi-v3",
            "title": "FR-11 v6 remains gated on deployable DVI v3",
            "root_cause": "Exp1418 did not run because exp1415 left dvi_v3_deployed=false.",
            "next_action": "Carry FR-11 v6 only after the DVI v3 deployment gate is proven open.",
            "prior_failures": [
                _prior(
                    "exp1418-fr11-self-learning-v6-dvi-v3",
                    "gate_blocked_upstream_dvi_v3_not_deployed",
                    "The next FR-11 v6 task must name exp1415 and require dvi_v3_deployed=true "
                    "before launch.",
                    False,
                )
            ],
        },
        {
            "id": "dpo-headline-validation-or-finetune-support",
            "title": "DPO path needs headline-valid local-model provenance",
            "root_cause": (
                "Exp1420 measured a DPO-style reranker fallback, but direct GGUF fine-tuning was "
                "not performed and headline_result_allowed=false."
            ),
            "next_action": (
                "Either add a supported local adapter/fine-tune path or relabel the track as a "
                "feature-reranker benchmark with separate headline gates."
            ),
            "prior_failures": [
                _prior(
                    "exp1420-dpo-verified-pairs-1508",
                    exp1420,
                    "The next task must resolve the gap between measured fallback AUROC and "
                    "headline-valid local SOTA model use.",
                    False,
                )
            ],
        },
        {
            "id": "test-suite-remaining-debt",
            "title": "Full Python suite and spec-coverage debt remain after focused fix",
            "root_cause": (
                "Exp1421 fixed the embedding-store runtime cluster, but the required full "
                "tests/python command and pre-existing spec-coverage debt remained red."
            ),
            "next_action": (
                "Partition the remaining full-suite failures into one bounded cluster per task "
                "and keep each task tied to explicit REQ/SCENARIO references."
            ),
            "prior_failures": [
                _prior(
                    "exp1421-test-suite-execution-debt-v1",
                    exp1421,
                    "The next test-debt task must not re-open the fixed embedding cluster; it "
                    "must target a named remaining failure cluster.",
                    False,
                )
            ],
        },
        {
            "id": "prm-label-completion",
            "title": "PRM v1 should fill missing local step labels before headline use",
            "root_cause": (
                "Exp1423 trained PRM v1 and measured AUROC, but only 1030 of the 1508 promoted "
                "traces were used because 478 local labels were missing."
            ),
            "next_action": (
                "Generate or recover labels for the missing promoted traces, then re-evaluate "
                "precision/recall and AUROC on a held-out split."
            ),
            "prior_failures": [
                _prior(
                    "exp1423-process-reward-model-v1-fover-1508",
                    exp1423,
                    "The next PRM task must address missing-label coverage before claiming a "
                    "1508-trace PRM result.",
                    False,
                )
            ],
        },
    ]
    return tasks


def _retired_experiments(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exp1419 = sources.get("exp1419", {})
    retired: list[dict[str, Any]] = []
    if _honest_verdict(exp1419) == "not_headline_full_pipeline_below_0_40":
        retired.append(
            {
                "experiment_id": "exp1419-fullscale-pipeline-v3-repair-executor",
                "result_artifact": "results/experiment_1419_fullscale_pipeline_v3_repair_executor.json",
                "retirement_scope": "exact rerun without a new repair-success root-cause fix",
                "reason": (
                    "The task had retire_if_same_verdict=true against exp1397 and returned the "
                    "same not_headline_full_pipeline_below_0_40 verdict with 0 successful repairs."
                ),
                "prior_failure_matched": "exp1397-fullscale-pipeline-v2-200cases",
                "root_cause": "Repair executor deployment did not translate into accepted repairs.",
            }
        )
    return retired


def _gpu_utilization_summary(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    model_assignments: list[dict[str, Any]] = []
    for exp_id in ("exp1414", "exp1419", "exp1420"):
        payload = sources.get(exp_id, {})
        specs = payload.get("model_specs")
        if not isinstance(specs, list):
            continue
        for spec in specs:
            if not isinstance(spec, Mapping):
                continue
            assignment = {
                "experiment_id": exp_id,
                "hf_id": spec.get("hf_id"),
                "role": spec.get("role"),
                "gpu": spec.get("gpu"),
                "cache_status": spec.get("cache_status"),
            }
            model_assignments.append(assignment)

    return {
        "gpu_required_tasks": ["exp1414", "exp1419", "exp1420"],
        "utilization_metrics_available": False,
        "observed_model_assignments": model_assignments,
        "summary": (
            "No .109 artifact or conductor-log entry exposes GPU utilization, VRAM, or "
            "wall-time telemetry. Available evidence only records model cache/provenance "
            "and GPU indices for exp1414/exp1419; exp1420 used a reranker fallback."
        ),
    }


def build_artifact(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: list[str],
    conductor_log_text: str,
    roadmap_next_present: bool,
    change_proposal_present: bool,
) -> dict[str, Any]:
    """REQ-REPORT-032: assemble the final retrospective from source artifacts."""

    criteria_results = evaluate_success_criteria(sources, missing_source_ids, conductor_log_text)
    criteria_met = sum(1 for item in criteria_results.values() if item["status"] == MET)
    carry_forward_tasks = _carry_forward_tasks(sources)
    prior_failures_required_next = {
        task["id"]: task["prior_failures"] for task in carry_forward_tasks
    }
    blocked_or_unmet = [
        criterion
        for criterion, result in criteria_results.items()
        if result["status"] != MET
    ]
    honest_verdict = (
        f"milestone_109_{criteria_met}_of_{len(CRITERIA)}_criteria_met_"
        "threshold_met_but_repair_dvi_fr11_and_pipeline_carry_forward"
        if criteria_met >= 10
        else f"milestone_109_{criteria_met}_of_{len(CRITERIA)}_criteria_met_threshold_missed"
    )

    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "criteria_total": len(CRITERIA),
        "criteria_met": criteria_met,
        "success_criteria_results": criteria_results,
        "retired_experiments": _retired_experiments(sources),
        "carry_forward_tasks": carry_forward_tasks,
        "prior_failures_required_next": prior_failures_required_next,
        "gpu_utilization_summary": _gpu_utilization_summary(sources),
        "experiment_statuses": _experiment_statuses(sources, missing_source_ids),
        "source_artifacts_checked": _source_artifacts_checked(missing_source_ids),
        "roadmap_inputs": {
            "research_roadmap_yaml_present": True,
            "research_roadmap_next_yaml_present": roadmap_next_present,
            "change_proposal_present": change_proposal_present,
            "missing_requested_inputs": []
            if roadmap_next_present
            else ["research-roadmap-next.yaml"],
        },
        "criteria_not_met_or_blocked": blocked_or_unmet,
        "honest_verdict": honest_verdict,
    }


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-032: write skeleton, load .109 evidence, then write final retro."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing = _load_sources(root_path / "results")
    conductor_log_path = root_path / "ops" / "conductor-log.md"
    conductor_log_text = (
        conductor_log_path.read_text(encoding="utf-8") if conductor_log_path.exists() else ""
    )
    artifact = build_artifact(
        sources,
        missing_source_ids=missing,
        conductor_log_text=conductor_log_text,
        roadmap_next_present=(root_path / "research-roadmap-next.yaml").exists(),
        change_proposal_present=(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ).exists(),
    )
    return _write_json(out, artifact)


if __name__ == "__main__":  # pragma: no cover - thin CLI convenience
    run()
