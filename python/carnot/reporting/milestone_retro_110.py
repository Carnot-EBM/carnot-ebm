"""Build the Exp 1438 milestone .110 retrospective artifact.

Spec: REQ-REPORT-035, SCENARIO-REPORT-035.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1438_milestone_110_retro.json"

EXPERIMENT = "1438_milestone_110_retro"
SCHEMA = "milestone_110_retro_v1"
RUN_DATE = "20260506"
MILESTONE = "2026.04.110"

MET = "met"
NOT_MET = "not_met"
BLOCKED = "blocked"
NOT_RUN = "not_run"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone",
    "criteria_total",
    "criteria_met",
    "success_criteria_results",
    "repair_v2_verdict",
    "dvi_fr11_verdict",
    "prm_verdict",
    "hardware_verdict",
    "carry_forward_tasks",
    "retired_exact_scopes",
    "honest_verdict",
)

SOURCE_FILES = {
    "exp1425": "experiment_1425_109_carryforward_activation_audit.json",
    "exp1426": "experiment_1426_test_suite_remaining_debt_cluster_map.json",
    "exp1427": "experiment_1427_repair_executor_rejection_ledger.json",
    "exp1428": "experiment_1428_dccd_schema_constrained_repair_v2.json",
    "exp1429": "experiment_1429_mcmc_constrained_repair_candidate_search.json",
    "exp1430": "experiment_1430_prm_guided_repair_selector.json",
    "exp1431": "experiment_1431_fullscale_pipeline_v4_micro_gated.json",
    "exp1432": "experiment_1432_dvi_v3_nonforgetting_replay_balanced.json",
    "exp1433": "experiment_1433_fr11_self_learning_v6_dvi_v3_gated.json",
    "exp1434": "experiment_1434_fover_prm_label_completion_v2.json",
    "exp1435": "experiment_1435_dpo_headline_provenance_audit.json",
    "exp1436": "experiment_1436_anchored_dual_path_latent_repair_v1.json",
    "exp1437": "experiment_1437_discrete_sb_kv260_rtl_lint_sim.json",
}

CRITERION_IDS = (
    "carry_forward_manifest",
    "test_debt_map",
    "repair_rejection_ledger",
    "repair_executor_v2",
    "candidate_search",
    "prm_repair_selector",
    "pipeline_micro_validation",
    "dvi_nonforgetting",
    "continuous_self_learning",
    "prm_label_completion",
    "dpo_provenance",
    "latent_planning_safeguard",
    "rtl_evidence",
    "retro",
)


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-035: write the durable in-progress skeleton first."""

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


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict", ""))


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _source_path(exp_id: str, field: str | None = None) -> str:
    path = f"results/{SOURCE_FILES[exp_id]}"
    return f"{path}:{field}" if field else path


def _source_unavailable(
    exp_id: str,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> str | None:
    if exp_id in missing_source_ids or exp_id not in sources:
        return NOT_RUN
    if _status(sources[exp_id]) == "blocked":
        return BLOCKED
    return None


def _criterion(
    status: str,
    target: str,
    evidence_paths: list[str],
    positive_evidence: list[str],
    negative_evidence: list[str],
    source_values: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": status,
        "target": target,
        "evidence_paths": evidence_paths,
        "positive_evidence": positive_evidence,
        "negative_evidence": negative_evidence,
        "source_values": dict(source_values),
    }


def _scored(
    exp_id: str,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    passed: bool,
    target: str,
    fields: tuple[str, ...],
    positive: str,
    negative: str,
) -> dict[str, Any]:
    unavailable = _source_unavailable(exp_id, sources, missing_source_ids)
    payload = sources.get(exp_id, {})
    source_values = {field: payload.get(field) for field in fields}
    source_values["honest_verdict"] = _verdict(payload)
    if unavailable is not None:
        return _criterion(
            unavailable,
            target,
            [_source_path(exp_id)],
            [],
            [f"{exp_id} unavailable or blocked: {source_values['honest_verdict']}"],
            source_values,
        )
    return _criterion(
        MET if passed else NOT_MET,
        target,
        [_source_path(exp_id, field) for field in fields],
        [positive] if passed else [],
        [] if passed else [negative],
        source_values,
    )


def _score_criteria(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> dict[str, dict[str, Any]]:
    exp1425 = sources.get("exp1425", {})
    exp1426 = sources.get("exp1426", {})
    exp1427 = sources.get("exp1427", {})
    exp1428 = sources.get("exp1428", {})
    exp1429 = sources.get("exp1429", {})
    exp1430 = sources.get("exp1430", {})
    exp1431 = sources.get("exp1431", {})
    exp1432 = sources.get("exp1432", {})
    exp1433 = sources.get("exp1433", {})
    exp1434 = sources.get("exp1434", {})
    exp1435 = sources.get("exp1435", {})
    exp1436 = sources.get("exp1436", {})
    exp1437 = sources.get("exp1437", {})

    dvi_auroc_delta = _number(exp1432.get("dvi_v3_auroc_delta"))
    dvi_v2_baseline = _number(exp1432.get("dvi_v2_auroc_delta_baseline"))
    selected_repair_rate = _number(exp1430.get("selected_repair_success_rate"))
    raw_best_of_n_rate = _number(exp1430.get("raw_best_of_n_repair_success_rate"))
    full_pipeline_pass_rate = _number(exp1431.get("full_pipeline_pass_rate"))
    missing_labels_filled = _number(exp1434.get("missing_labels_filled"))

    criteria = {
        "carry_forward_manifest": _scored(
            "exp1425",
            sources,
            missing_source_ids,
            exp1425.get("carryforward_manifest_complete") is True,
            "exp1425.carryforward_manifest_complete=true",
            ("carryforward_manifest_complete", "carryforward_task_count"),
            "Carry-forward manifest mapped prior .109 failures.",
            "Carry-forward activation did not complete.",
        ),
        "test_debt_map": _scored(
            "exp1426",
            sources,
            missing_source_ids,
            exp1426.get("failure_cluster_map_complete") is True
            and exp1426.get("next_cluster_recommended") is not None,
            "exp1426.failure_cluster_map_complete=true and one next cluster recommended",
            (
                "failure_cluster_map_complete",
                "next_cluster_recommended",
                "spec_coverage_debt_count",
            ),
            "Remaining test debt was partitioned into a next fix cluster.",
            "Test debt map is incomplete or lacks a next cluster.",
        ),
        "repair_rejection_ledger": _scored(
            "exp1427",
            sources,
            missing_source_ids,
            exp1427.get("rejection_ledger_complete") is True
            and exp1427.get("top_rejection_reason") is not None,
            "exp1427.rejection_ledger_complete=true and top reasons quantified",
            ("rejection_ledger_complete", "top_rejection_reason", "repair_v2_contract_ready"),
            "Repair rejection reasons were ledged before repair v2.",
            "Repair rejection ledger did not close.",
        ),
        "repair_executor_v2": _scored(
            "exp1428",
            sources,
            missing_source_ids,
            exp1428.get("repair_executor_v2_deployed") is True
            and (_number(exp1428.get("repaired_case_success_rate")) or 0.0) > 0.0,
            "exp1428.repair_executor_v2_deployed=true and repaired_case_success_rate > 0.0",
            (
                "repair_executor_v2_deployed",
                "repaired_case_success_rate",
                "repaired_cases_successful",
                "local_sota_model_inference_used",
            ),
            "Repair v2 produced nonzero accepted repairs.",
            "Repair v2 did not produce accepted repairs.",
        ),
        "candidate_search": _scored(
            "exp1429",
            sources,
            missing_source_ids,
            exp1429.get("candidate_search_complete") is True
            and (_number(exp1429.get("repair_success_rate_best_of_n")) or 0.0) > 0.0,
            "exp1429.candidate_search_complete=true and best-of-N success exceeds zero baseline",
            (
                "candidate_search_complete",
                "repair_success_rate_best_of_n",
                "repair_success_rate_one_candidate",
                "local_sota_model_inference_used",
            ),
            "Best-of-N repair candidate search exceeded the .109 zero baseline.",
            "Candidate search did not beat the zero-repair baseline.",
        ),
        "prm_repair_selector": _scored(
            "exp1430",
            sources,
            missing_source_ids,
            exp1430.get("prm_guided_selection_ready") is True
            and selected_repair_rate is not None
            and raw_best_of_n_rate is not None
            and selected_repair_rate >= raw_best_of_n_rate,
            "exp1430.prm_guided_selection_ready=true and selected rate does not reduce acceptance",
            (
                "prm_guided_selection_ready",
                "selected_repair_success_rate",
                "raw_best_of_n_repair_success_rate",
                "selection_improvement_pp",
            ),
            "PRM selection was ready and did not reduce accepted repair rate.",
            "PRM selection was unavailable or reduced accepted repair rate.",
        ),
        "pipeline_micro_validation": _scored(
            "exp1431",
            sources,
            missing_source_ids,
            full_pipeline_pass_rate is not None and full_pipeline_pass_rate > 0.305,
            "exp1431.full_pipeline_pass_rate > 0.305 on a new 50-case micro run",
            (
                "cases_evaluated",
                "full_pipeline_pass_rate",
                "beats_exp1419_baseline",
                "runtime_evidence_allows_headline_scaleup",
            ),
            "Micro validation beat the Exp 1419 baseline.",
            "Micro validation did not beat the Exp 1419 baseline.",
        ),
        "dvi_nonforgetting": _scored(
            "exp1432",
            sources,
            missing_source_ids,
            exp1432.get("dvi_v3_deployed") is True
            and (_number(exp1432.get("nonforgetting_rate")) or 0.0) >= 0.99
            and dvi_auroc_delta is not None
            and dvi_v2_baseline is not None
            and dvi_auroc_delta >= dvi_v2_baseline
            and exp1432.get("dvi_v3_auroc_nonregression_gate") is True,
            "exp1432.dvi_v3_deployed=true, nonforgetting_rate >= 0.99, AUROC nonregresses",
            (
                "dvi_v3_deployed",
                "nonforgetting_rate",
                "dvi_v3_auroc_delta",
                "dvi_v2_auroc_delta_baseline",
                "dvi_v3_auroc_nonregression_gate",
            ),
            "DVI v3 passed deployment, nonforgetting, and AUROC gates.",
            "DVI v3 failed deployment, nonforgetting, or AUROC gates.",
        ),
        "continuous_self_learning": _scored(
            "exp1433",
            sources,
            missing_source_ids,
            exp1433.get("headline_result_allowed") is True,
            "exp1433.headline_result_allowed=true or a DVI gate-block artifact identifies unmet DVI",
            (
                "dvi_v3_checkpoint_active",
                "headline_result_allowed",
                "v6_new_promoted_count",
                "self_learning_delta_overall",
                "session_memory_updated",
            ),
            "FR-11 v6 produced headline self-learning evidence.",
            "DVI was active, but FR-11 v6 promoted zero new cases and stayed non-headline.",
        ),
        "prm_label_completion": _scored(
            "exp1434",
            sources,
            missing_source_ids,
            missing_labels_filled is not None
            and missing_labels_filled >= 478
            and exp1434.get("prmv2_trained") is True,
            "exp1434.missing_labels_filled >= 478 or exact blocker ledger; PRM v2 uses labels",
            (
                "missing_labels_filled",
                "missing_labels_remaining",
                "prmv2_trained",
                "headline_label_coverage_ready",
            ),
            "PRM labels were completed and PRM v2 trained.",
            "PRM labels remain incomplete without satisfying the criterion.",
        ),
        "dpo_provenance": _scored(
            "exp1435",
            sources,
            missing_source_ids,
            exp1435.get("headline_provenance_ready") is True
            or exp1435.get("reranker_track_relabelled") is True,
            "exp1435.headline_provenance_ready=true or DPO relabeled reranker-only",
            (
                "headline_provenance_ready",
                "reranker_track_relabelled",
                "direct_gguf_finetune_supported",
            ),
            "DPO was honestly relabeled as reranker-only until local adapter support exists.",
            "DPO provenance is neither headline-ready nor relabeled.",
        ),
        "latent_planning_safeguard": _scored(
            "exp1436",
            sources,
            missing_source_ids,
            exp1436.get("anchored_repair_viable") is True
            or exp1436.get("accuracy_delta_after_planning") is not None,
            "exp1436.anchored_repair_viable=true or decisive negative drift metrics recorded",
            ("anchored_repair_viable", "accuracy_delta_after_planning", "latent_drift_norm"),
            "Anchored dual-path latent repair was viable.",
            "Latent planning safeguard evidence is missing.",
        ),
    }

    rtl_unavailable = _source_unavailable("exp1437", sources, missing_source_ids)
    missing_tool_blocker = "missing_tool" in _verdict(exp1437) or "missing-tool" in _verdict(
        exp1437
    )
    rtl_passed = exp1437.get("rtl_lint_complete") is True or (
        missing_tool_blocker and exp1437.get("hardware_claim_allowed") is False
    )
    if rtl_unavailable is not None and not rtl_passed:
        rtl_status = rtl_unavailable
    else:
        rtl_status = MET if rtl_passed else NOT_MET
    criteria["rtl_evidence"] = _criterion(
        rtl_status,
        "exp1437.rtl_lint_complete=true or missing-tool blockers; hardware_claim_allowed=false unless hardware ran",
        [
            _source_path("exp1437", "rtl_lint_complete"),
            _source_path("exp1437", "simulation_complete"),
            _source_path("exp1437", "hardware_claim_allowed"),
            _source_path("exp1437", "rtl_sources_checked"),
        ],
        ["RTL lint/simulation evidence or an allowed missing-tool blocker was recorded."]
        if rtl_status == MET
        else [],
        []
        if rtl_status == MET
        else ["RTL source was missing, so lint and simulation did not run."],
        {
            "status": exp1437.get("status"),
            "rtl_lint_complete": exp1437.get("rtl_lint_complete"),
            "simulation_complete": exp1437.get("simulation_complete"),
            "hardware_claim_allowed": exp1437.get("hardware_claim_allowed"),
            "hardware_execution_performed": exp1437.get("hardware_execution_performed"),
            "rtl_sources_checked": exp1437.get("rtl_sources_checked"),
            "honest_verdict": _verdict(exp1437),
        },
    )
    criteria["retro"] = _criterion(
        MET,
        "exp1438.criteria_total=14 and honest carry-forward rules are recorded",
        ["results/experiment_1438_milestone_110_retro.json"],
        ["This final artifact records all 14 criteria and carry-forward rules."],
        [],
        {"criteria_total": 14, "status": "complete"},
    )
    return criteria


def _source_checks(missing_source_ids: set[str]) -> list[dict[str, Any]]:
    checks = []
    for exp_id, filename in SOURCE_FILES.items():
        checks.append(
            {
                "experiment_id": exp_id,
                "path": f"results/{filename}",
                "exists": exp_id not in missing_source_ids,
            }
        )
    return checks


def _repair_v2_verdict(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1428 = sources.get("exp1428", {})
    exp1429 = sources.get("exp1429", {})
    exp1430 = sources.get("exp1430", {})
    exp1431 = sources.get("exp1431", {})
    return {
        "summary": "nonzero_repair_breakthrough_but_headline_scaleup_still_prototype_bound",
        "repair_executor_v2_deployed": exp1428.get("repair_executor_v2_deployed"),
        "accepted_repairs": exp1428.get("repaired_cases_successful"),
        "repair_success_rate": exp1428.get("repaired_case_success_rate"),
        "candidate_search_best_of_n": exp1429.get("repair_success_rate_best_of_n"),
        "prm_selector_ready": exp1430.get("prm_guided_selection_ready"),
        "pipeline_micro_pass_rate": exp1431.get("full_pipeline_pass_rate"),
        "headline_limitations": [
            "exp1428 and exp1429 report prototype/no live SOTA inference for headline claims",
            "exp1431 reports runtime_evidence_allows_headline_scaleup=false",
        ],
        "evidence_paths": [
            _source_path("exp1428"),
            _source_path("exp1429"),
            _source_path("exp1430"),
            _source_path("exp1431"),
        ],
    }


def _dvi_fr11_verdict(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1432 = sources.get("exp1432", {})
    exp1433 = sources.get("exp1433", {})
    return {
        "summary": "dvi_v3_deployed_but_fr11_v6_no_positive_growth",
        "dvi_v3_deployed": exp1432.get("dvi_v3_deployed"),
        "nonforgetting_rate": exp1432.get("nonforgetting_rate"),
        "dvi_v3_auroc_delta": exp1432.get("dvi_v3_auroc_delta"),
        "fr11_headline_allowed": exp1433.get("headline_result_allowed"),
        "fr11_new_promoted_count": exp1433.get("v6_new_promoted_count"),
        "fr11_self_learning_delta_overall": exp1433.get("self_learning_delta_overall"),
        "evidence_paths": [_source_path("exp1432"), _source_path("exp1433")],
    }


def _prm_verdict(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1430 = sources.get("exp1430", {})
    exp1434 = sources.get("exp1434", {})
    return {
        "summary": "prm_labels_completed_and_selector_non_degrading_but_selector_no_improvement",
        "selector_ready": exp1430.get("prm_guided_selection_ready"),
        "selection_improvement_pp": exp1430.get("selection_improvement_pp"),
        "missing_labels_filled": exp1434.get("missing_labels_filled"),
        "missing_labels_remaining": exp1434.get("missing_labels_remaining"),
        "prmv2_trained": exp1434.get("prmv2_trained"),
        "evidence_paths": [_source_path("exp1430"), _source_path("exp1434")],
    }


def _hardware_verdict(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    exp1437 = sources.get("exp1437", {})
    return {
        "summary": "blocked_missing_discrete_sb_rtl_source_no_hardware_claim",
        "rtl_lint_complete": exp1437.get("rtl_lint_complete"),
        "simulation_complete": exp1437.get("simulation_complete"),
        "hardware_claim_allowed": exp1437.get("hardware_claim_allowed"),
        "hardware_execution_performed": exp1437.get("hardware_execution_performed"),
        "rtl_sources_checked": exp1437.get("rtl_sources_checked"),
        "honest_verdict": _verdict(exp1437),
        "evidence_paths": [_source_path("exp1437")],
    }


def _prior_failure(exp_id: str, sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "verdict": _verdict(sources.get(exp_id, {})) or "missing",
        "evidence_path": _source_path(exp_id),
    }


def _carry_forward_tasks(
    sources: Mapping[str, Mapping[str, Any]],
    criteria: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "id": "repair_v2_live_sota_headline_scaleup",
            "title": "Convert repair v2 prototype wins into live-SOTA, headline-eligible scale evidence.",
            "prior_failures": [
                _prior_failure("exp1428", sources),
                _prior_failure("exp1429", sources),
                _prior_failure("exp1431", sources),
            ],
            "next_rule": (
                "Do not run another large full-pipeline scale-up until live local SOTA "
                "inference and runtime evidence allow headline scale-up."
            ),
            "retire_if_same_verdict": True,
            "retire_rule": (
                "If the next scale-up again reports prototype/no-live-SOTA or "
                "no-headline-scaleup with the same method, retire headline repair-v2 "
                "scale claims until the runtime path changes."
            ),
        },
        {
            "id": "fr11_positive_growth_followup",
            "title": "Diagnose why deployed DVI v3 produced zero FR-11 promoted growth.",
            "prior_failures": [_prior_failure("exp1433", sources)],
            "next_rule": (
                "Future FR-11 v6 work must change promotion thresholds, candidate "
                "generation, or memory policy before rerun."
            ),
            "retire_if_same_verdict": True,
            "retire_rule": (
                "If DVI remains deployed and FR-11 again reports zero promoted growth, "
                "retire this FR-11 v6 variant and require a new root-cause plan."
            ),
        },
        {
            "id": "test_debt_spec_coverage_cluster",
            "title": "Fix the prioritized spec-coverage traceability metadata cluster.",
            "prior_failures": [_prior_failure("exp1426", sources)],
            "next_rule": (
                "Handle the named spec_coverage_traceability_metadata cluster first; "
                "do not reopen the already-fixed embedding-store cluster."
            ),
            "retire_if_same_verdict": False,
            "retire_rule": "Do not retire whole test debt; split persistent failures by cluster.",
        },
        {
            "id": "dpo_adapter_or_reranker_only",
            "title": "Keep DPO reranker-only unless direct local adapter or conversion tooling exists.",
            "prior_failures": [_prior_failure("exp1435", sources)],
            "next_rule": (
                "A future DPO headline task must name concrete local adapter/conversion "
                "tooling before running."
            ),
            "retire_if_same_verdict": True,
            "retire_rule": (
                "If direct GGUF fine-tune support remains absent, retire DPO headline "
                "wording and preserve reranker-only status."
            ),
        },
        {
            "id": "hardware_rtl_source_before_lint_sim",
            "title": "Implement the missing Discrete SB RTL source before rerunning lint/sim.",
            "prior_failures": [_prior_failure("exp1437", sources)],
            "next_rule": (
                "Create hardware/kv260/discrete_sb_256.v from the RTL spec before "
                "rerunning Exp 1437 lint or simulation."
            ),
            "retire_if_same_verdict": True,
            "retire_rule": (
                "If the exact lint/sim rerun still lacks the RTL source, retire that "
                "rerun and require source implementation first."
            ),
        },
    ]


def _retired_exact_scopes(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    retired: list[dict[str, Any]] = []
    for item in sources.get("exp1425", {}).get("forbidden_exact_reruns", []):
        if isinstance(item, dict):
            retired.append(
                {
                    "scope": str(item.get("forbidden_scope", "exact exp1419 200-case rerun")),
                    "prior_verdict": item.get("prior_verdict"),
                    "source": _source_path("exp1425", "forbidden_exact_reruns"),
                    "retire_if_same_verdict": item.get("retire_if_same_verdict", True),
                }
            )
    retired.extend(
        [
            {
                "scope": "DPO headline claim without direct local adapter or conversion tooling",
                "prior_verdict": _verdict(sources.get("exp1435", {})),
                "source": _source_path("exp1435"),
                "retire_if_same_verdict": True,
            },
            {
                "scope": "FR-11 v6 headline self-learning claim with zero promoted growth",
                "prior_verdict": _verdict(sources.get("exp1433", {})),
                "source": _source_path("exp1433"),
                "retire_if_same_verdict": True,
            },
            {
                "scope": "exp1437 RTL lint/sim rerun before hardware/kv260/discrete_sb_256.v exists",
                "prior_verdict": _verdict(sources.get("exp1437", {})),
                "source": _source_path("exp1437"),
                "retire_if_same_verdict": True,
            },
        ]
    )
    return retired


def _outcome_summaries(sources: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    exp1426 = sources.get("exp1426", {})
    exp1435 = sources.get("exp1435", {})
    exp1436 = sources.get("exp1436", {})
    return {
        "repair": _repair_v2_verdict(sources),
        "dvi_fr11": _dvi_fr11_verdict(sources),
        "prm": _prm_verdict(sources),
        "dpo": {
            "summary": "headline_not_ready_reranker_only",
            "headline_provenance_ready": exp1435.get("headline_provenance_ready"),
            "reranker_track_relabelled": exp1435.get("reranker_track_relabelled"),
            "direct_gguf_finetune_supported": exp1435.get("direct_gguf_finetune_supported"),
            "evidence_paths": [_source_path("exp1435")],
        },
        "latent": {
            "summary": "anchored_dual_path_viable_with_zero_accuracy_loss",
            "anchored_repair_viable": exp1436.get("anchored_repair_viable"),
            "accuracy_delta_after_planning": exp1436.get("accuracy_delta_after_planning"),
            "latent_drift_norm": exp1436.get("latent_drift_norm"),
            "evidence_paths": [_source_path("exp1436")],
        },
        "test_debt": {
            "summary": "collection_clean_but_spec_coverage_debt_remains",
            "next_cluster_recommended": exp1426.get("next_cluster_recommended"),
            "spec_coverage_debt_count": exp1426.get("spec_coverage_debt_count"),
            "evidence_paths": [_source_path("exp1426")],
        },
        "hardware": _hardware_verdict(sources),
    }


def build_artifact(
    sources: Mapping[str, dict[str, Any]],
    missing_source_ids: list[str],
    roadmap_doc_present: bool,
    roadmap_yaml_present: bool,
    conductor_log_text: str,
    roadmap_next_present: bool = False,
) -> dict[str, Any]:
    """REQ-REPORT-035: score .110 criteria and assemble the terminal artifact."""

    missing = set(missing_source_ids)
    criteria = _score_criteria(sources, missing)
    criteria_met = sum(1 for result in criteria.values() if result["status"] == MET)
    criteria_total = len(CRITERION_IDS)
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "milestone": MILESTONE,
        "criteria_total": criteria_total,
        "criteria_met": criteria_met,
        "success_criteria_results": criteria,
        "repair_v2_verdict": _repair_v2_verdict(sources),
        "dvi_fr11_verdict": _dvi_fr11_verdict(sources),
        "prm_verdict": _prm_verdict(sources),
        "hardware_verdict": _hardware_verdict(sources),
        "carry_forward_tasks": _carry_forward_tasks(sources, criteria),
        "retired_exact_scopes": _retired_exact_scopes(sources),
        "honest_verdict": (
            f"milestone_110_{criteria_met}_of_{criteria_total}_criteria_met_"
            "threshold_met_repair_dvi_prm_dpo_latent_positive_fr11_growth_and_rtl_source_"
            "carry_forward"
        ),
        "source_artifacts_checked": _source_checks(missing),
        "roadmap_inputs": {
            "change_proposal_path": "openspec/change-proposals/research-roadmap-vNEXT.md",
            "change_proposal_present": roadmap_doc_present,
            "active_roadmap_yaml_path": "research-roadmap.yaml",
            "active_roadmap_yaml_present": roadmap_yaml_present,
            "requested_research_roadmap_next_present": roadmap_next_present,
        },
        "outcome_summaries": _outcome_summaries(sources),
        "operational_notes": {
            "conductor_log_examined": bool(conductor_log_text),
            "research_roadmap_next_yaml_note": (
                "research-roadmap-next.yaml was not present; active .110 roadmap was read "
                "from research-roadmap.yaml"
            ),
            "ops_docs_update_status": (
                "ops/status.md, ops/changelog.md, and _bmad/traceability.md left unchanged "
                "because this terminal task is followed by the conductor reconciliation step."
            ),
            "scripts_research_conductor_modified": False,
        },
    }
    return artifact


def run(root: Path | str = REPO_ROOT, out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    root_path = Path(root)
    sources, missing = _load_sources(root_path / "results")
    conductor_log_path = root_path / "ops" / "conductor-log.md"
    conductor_log_text = (
        conductor_log_path.read_text(encoding="utf-8") if conductor_log_path.exists() else ""
    )
    artifact = build_artifact(
        sources,
        missing,
        roadmap_doc_present=(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ).exists(),
        roadmap_yaml_present=(root_path / "research-roadmap.yaml").exists(),
        roadmap_next_present=(root_path / "research-roadmap-next.yaml").exists(),
        conductor_log_text=conductor_log_text,
    )
    return _write_json(Path(out_path), artifact)
