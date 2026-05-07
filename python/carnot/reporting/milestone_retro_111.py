"""Build the Exp 1452 milestone .111 retrospective artifact.

Spec: REQ-REPORT-038, SCENARIO-REPORT-038.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1452_milestone_111_retro.json"

EXPERIMENT = "1452_milestone_111_retro"
SCHEMA = "milestone_111_retro_v1"
RUN_DATE = "20260507"
MILESTONE = "2026.04.111"

MET = "met"
UNMET = "unmet"
GATE_BLOCKED_WITH_EVIDENCE = "gate_blocked_with_evidence"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone",
    "criteria_total",
    "criteria_met",
    "successful_tasks",
    "blocked_tasks",
    "retired_variants",
    "carry_forward_tracks",
    "ops_docs_updated",
    "honest_verdict",
)

SOURCE_FILES = {
    "exp1439": "experiment_1439_110_carryforward_activation_manifest.json",
    "exp1440": "experiment_1440_spec_coverage_traceability_metadata_fix.json",
    "exp1441": "experiment_1441_discrete_sb_rtl_source_implementation.json",
    "exp1442": "experiment_1442_live_sota_repair_runtime_preflight.json",
    "exp1443": "experiment_1443_live_sota_dccd_semctrl_repair_v3.json",
    "exp1444": "experiment_1444_arm_carnot_energy_repair_reranker.json",
    "exp1445": "experiment_1445_full_pipeline_v5_100case_prescale.json",
    "exp1446": "experiment_1446_fr11_zero_growth_root_cause_diagnosis.json",
    "exp1447": "experiment_1447_fr11_v7_memory_policy_growth.json",
    "exp1448": "experiment_1448_prm_v3_online_process_reward_agent.json",
    "exp1449": "experiment_1449_ltlzinc_temporal_continual_learning_adapter.json",
    "exp1450": "experiment_1450_ebt_nrgpt_local_microprototype_audit.json",
    "exp1451": "experiment_1451_discrete_sb_rtl_lint_sim_rerun.json",
}

CRITERION_SOURCE = {
    "carry_forward_manifest": "exp1439",
    "spec_coverage_cluster": "exp1440",
    "discrete_sb_rtl_source": "exp1441",
    "live_sota_runtime": "exp1442",
    "live_repair_v3": "exp1443",
    "energy_reranker": "exp1444",
    "pipeline_pre_scale": "exp1445",
    "fr11_diagnosis": "exp1446",
    "continuous_self_learning": "exp1447",
    "prm_process_agent": "exp1448",
    "ltlzinc_adapter": "exp1449",
    "ebt_nrgpt_micro_baseline": "exp1450",
    "rtl_lint_sim": "exp1451",
    "retro": "exp1452",
}


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-038: write the durable in-progress skeleton first.

    The conductor has had prior retro tasks exhaust their turn budget before a
    terminal file was written. This skeleton makes partial progress auditable
    even if later evidence scoring is interrupted.
    """

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


def _source_missing_criterion(exp_id: str, target: str) -> dict[str, Any]:
    return _criterion(
        UNMET,
        target,
        [_source_path(exp_id)],
        [],
        [f"{exp_id} source artifact is missing."],
        {"status": "missing", "honest_verdict": "missing_artifact"},
    )


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
    if exp_id in missing_source_ids or exp_id not in sources:
        return _source_missing_criterion(exp_id, target)
    payload = sources[exp_id]
    source_values = {field: payload.get(field) for field in fields}
    source_values["status"] = payload.get("status")
    source_values["honest_verdict"] = _verdict(payload)
    return _criterion(
        MET if passed else UNMET,
        target,
        [_source_path(exp_id, field) for field in fields],
        [positive] if passed else [],
        [] if passed else [negative],
        source_values,
    )


def _gate_blocked(
    exp_id: str,
    target: str,
    evidence_paths: list[str],
    reason: str,
    source_values: Mapping[str, Any],
) -> dict[str, Any]:
    values = dict(source_values)
    values.setdefault("status", "missing" if exp_id not in SOURCE_FILES else values.get("status"))
    return _criterion(
        GATE_BLOCKED_WITH_EVIDENCE,
        target,
        evidence_paths,
        [],
        [reason],
        values,
    )


def _score_live_runtime(exp1442: Mapping[str, Any]) -> dict[str, Any]:
    target = "exp1442.local_sota_runtime_ready=true; otherwise precise cache/GPU blockers"
    ready = exp1442.get("local_sota_runtime_ready") is True
    if ready and exp1442.get("live_sota_model_inference_used") is True:
        return _criterion(
            MET,
            target,
            [_source_path("exp1442", "local_sota_runtime_ready")],
            ["Live local SOTA runtime completed mandated model inference."],
            [],
            {
                "status": exp1442.get("status"),
                "local_sota_runtime_ready": exp1442.get("local_sota_runtime_ready"),
                "live_sota_model_inference_used": exp1442.get("live_sota_model_inference_used"),
                "honest_verdict": _verdict(exp1442),
            },
        )
    return _gate_blocked(
        "exp1442",
        target,
        [
            _source_path("exp1442", "local_sota_runtime_ready"),
            _source_path("exp1442", "models_missing_from_cache"),
            _source_path("exp1442", "blockers"),
        ],
        "Live SOTA runtime preflight recorded exact blockers and no completed live inference.",
        {
            "status": exp1442.get("status"),
            "local_sota_runtime_ready": exp1442.get("local_sota_runtime_ready"),
            "live_sota_model_inference_used": exp1442.get("live_sota_model_inference_used"),
            "models_found_in_cache": exp1442.get("models_found_in_cache"),
            "models_missing_from_cache": exp1442.get("models_missing_from_cache"),
            "blockers": exp1442.get("blockers"),
            "honest_verdict": _verdict(exp1442),
        },
    )


def _score_repair_v3(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> dict[str, Any]:
    target = "exp1443.live_sota_inference_used=true and live_repair_success_rate > 0.0"
    exp1443 = sources.get("exp1443", {})
    success_rate = _number(exp1443.get("live_repair_success_rate"))
    if "exp1443" not in missing_source_ids and exp1443:
        passed = exp1443.get("live_sota_inference_used") is True and (success_rate or 0.0) > 0.0
        return _scored(
            "exp1443",
            sources,
            missing_source_ids,
            passed,
            target,
            (
                "live_sota_inference_used",
                "live_repair_success_rate",
                "live_repair_candidate_pool_ready",
            ),
            "Live SOTA repair v3 produced nonzero repairs.",
            "Repair v3 ran but did not produce nonzero live SOTA repair evidence.",
        )
    exp1442 = sources.get("exp1442", {})
    return _gate_blocked(
        "exp1443",
        target,
        [_source_path("exp1442"), _source_path("exp1443")],
        "Exp1443 artifact is missing because the exp1442 live-SOTA runtime gate failed.",
        {
            "status": "missing",
            "honest_verdict": "missing_artifact_gate_blocked_by_exp1442",
            "upstream_exp1442_verdict": _verdict(exp1442),
            "upstream_exp1442_local_sota_runtime_ready": exp1442.get("local_sota_runtime_ready"),
        },
    )


def _score_energy_reranker(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> dict[str, Any]:
    target = "exp1444.energy_reranker_ready=true and false-acceptance rate does not increase"
    exp1444 = sources.get("exp1444", {})
    if "exp1444" in missing_source_ids or not exp1444:
        return _source_missing_criterion("exp1444", target)
    ready = exp1444.get("energy_reranker_ready") is True
    false_acceptance_delta = _number(exp1444.get("false_acceptance_rate_delta"))
    passed = ready and false_acceptance_delta is not None and false_acceptance_delta <= 0.0
    if _status(exp1444) == "blocked" or _verdict(exp1444) == "blocked_gate_check_failed":
        return _gate_blocked(
            "exp1444",
            target,
            [_source_path("exp1444", "gate_check_summary")],
            "Energy reranker did not run because the upstream repair-v3 candidate-pool gate failed.",
            {
                "status": exp1444.get("status"),
                "gate_check_summary": exp1444.get("gate_check_summary"),
                "gates_evaluated": exp1444.get("gates_evaluated"),
                "honest_verdict": _verdict(exp1444),
            },
        )
    return _scored(
        "exp1444",
        sources,
        missing_source_ids,
        passed,
        target,
        ("energy_reranker_ready", "false_acceptance_rate_delta"),
        "Energy reranker was ready and did not increase false acceptance.",
        "Energy reranker ran without satisfying readiness or false-acceptance gates.",
    )


def _score_pipeline_pre_scale(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> dict[str, Any]:
    target = "exp1445.full_pipeline_pass_rate > 0.62 or an honest blocker prevents scale-up"
    exp1445 = sources.get("exp1445", {})
    if "exp1445" not in missing_source_ids and exp1445:
        rate = _number(exp1445.get("full_pipeline_pass_rate"))
        if rate is not None and rate > 0.62:
            return _criterion(
                MET,
                target,
                [_source_path("exp1445", "full_pipeline_pass_rate")],
                ["Full pipeline v5 exceeded the .110 0.62 pre-scale baseline."],
                [],
                {"full_pipeline_pass_rate": rate, "honest_verdict": _verdict(exp1445)},
            )
        return _gate_blocked(
            "exp1445",
            target,
            [_source_path("exp1445")],
            "Full pipeline pre-scale recorded an honest scale-up blocker.",
            {"status": exp1445.get("status"), "honest_verdict": _verdict(exp1445)},
        )
    return _gate_blocked(
        "exp1445",
        target,
        [_source_path("exp1443"), _source_path("exp1444"), _source_path("exp1445")],
        "Exp1445 artifact is missing because repair-v3 and reranker gates did not pass.",
        {
            "status": "missing",
            "honest_verdict": "missing_artifact_gate_blocked_by_exp1443_exp1444",
            "upstream_exp1443_present": "exp1443" in sources,
            "upstream_exp1444_verdict": _verdict(sources.get("exp1444", {})),
        },
    )


def _score_prm_process_agent(exp1448: Mapping[str, Any]) -> dict[str, Any]:
    target = (
        "exp1448.pra_selector_ready=true and selection_improvement_pp > 0 "
        "or decisive no-improvement evidence"
    )
    improvement = _number(exp1448.get("selection_improvement_pp"))
    decisive_no_improvement = (
        exp1448.get("pra_selector_ready") is True
        and improvement == 0.0
        and (_number(exp1448.get("step_scores_generated")) or 0.0) > 0.0
        and (_number(exp1448.get("cases_evaluated")) or 0.0) > 0.0
        and exp1448.get("regression_against_prm_v1") is False
        and "no_headline_improvement" in _verdict(exp1448)
    )
    passed = exp1448.get("pra_selector_ready") is True and (
        (improvement is not None and improvement > 0.0) or decisive_no_improvement
    )
    return _criterion(
        MET if passed else UNMET,
        target,
        [
            _source_path("exp1448", "pra_selector_ready"),
            _source_path("exp1448", "selection_improvement_pp"),
            _source_path("exp1448", "step_scores_generated"),
        ],
        ["PRM v3 produced decisive no-improvement evidence without regressing PRM v1."]
        if decisive_no_improvement
        else (["PRM v3 improved selection over the prior selector."] if passed else []),
        [] if passed else ["PRM process agent did not satisfy readiness or evidence gates."],
        {
            "status": exp1448.get("status"),
            "pra_selector_ready": exp1448.get("pra_selector_ready"),
            "selection_improvement_pp": exp1448.get("selection_improvement_pp"),
            "step_scores_generated": exp1448.get("step_scores_generated"),
            "cases_evaluated": exp1448.get("cases_evaluated"),
            "regression_against_prm_v1": exp1448.get("regression_against_prm_v1"),
            "honest_verdict": _verdict(exp1448),
        },
    )


def _score_criteria(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
) -> dict[str, dict[str, Any]]:
    exp1439 = sources.get("exp1439", {})
    exp1440 = sources.get("exp1440", {})
    exp1441 = sources.get("exp1441", {})
    exp1442 = sources.get("exp1442", {})
    exp1446 = sources.get("exp1446", {})
    exp1447 = sources.get("exp1447", {})
    exp1448 = sources.get("exp1448", {})
    exp1449 = sources.get("exp1449", {})
    exp1450 = sources.get("exp1450", {})
    exp1451 = sources.get("exp1451", {})
    temporal_cases = _number(
        exp1449.get("temporal_constraint_cases_generated", exp1449.get("temporal_cases_generated"))
    )

    return {
        "carry_forward_manifest": _scored(
            "exp1439",
            sources,
            missing_source_ids,
            exp1439.get("carryforward_manifest_complete") is True,
            "exp1439.carryforward_manifest_complete=true and .110 carry-forwards mapped",
            ("carryforward_manifest_complete", "carryforward_task_count"),
            ".110 carry-forward tracks were mapped into .111 tasks.",
            "Carry-forward manifest did not complete.",
        ),
        "spec_coverage_cluster": _scored(
            "exp1440",
            sources,
            missing_source_ids,
            exp1440.get("spec_coverage_metadata_cluster_fixed") is True,
            "exp1440.spec_coverage_metadata_cluster_fixed=true or exact residual blockers exist",
            (
                "spec_coverage_metadata_cluster_fixed",
                "initial_spec_coverage_debt_count",
                "final_spec_coverage_debt_count",
                "residual_blockers",
            ),
            "Spec-coverage metadata cluster was fixed to zero checker debt.",
            "Spec-coverage metadata cluster remains unresolved.",
        ),
        "discrete_sb_rtl_source": _scored(
            "exp1441",
            sources,
            missing_source_ids,
            exp1441.get("rtl_source_created") is True and exp1441.get("testbench_created") is True,
            "exp1441.rtl_source_created=true and a testbench exists",
            ("rtl_source_created", "rtl_source_path", "testbench_created", "testbench_path"),
            "Discrete SB RTL source and testbench were created.",
            "Discrete SB RTL source or testbench remains missing.",
        ),
        "live_sota_runtime": _score_live_runtime(exp1442),
        "live_repair_v3": _score_repair_v3(sources, missing_source_ids),
        "energy_reranker": _score_energy_reranker(sources, missing_source_ids),
        "pipeline_pre_scale": _score_pipeline_pre_scale(sources, missing_source_ids),
        "fr11_diagnosis": _scored(
            "exp1446",
            sources,
            missing_source_ids,
            exp1446.get("fr11_zero_growth_root_cause_identified") is True,
            "exp1446.fr11_zero_growth_root_cause_identified=true",
            ("fr11_zero_growth_root_cause_identified", "fr11_zero_growth_root_cause"),
            "FR-11 zero-growth root cause was identified before rerun.",
            "FR-11 zero-growth root cause remains unidentified.",
        ),
        "continuous_self_learning": _scored(
            "exp1447",
            sources,
            missing_source_ids,
            (_number(exp1447.get("self_learning_delta_overall")) or 0.0) > 0.0
            and exp1447.get("nonforgetting_preserved") is True,
            "exp1447.self_learning_delta_overall > 0 with nonforgetting preserved",
            ("self_learning_delta_overall", "nonforgetting_preserved", "nonforgetting_rate"),
            "FR-11 v7 produced positive verified growth while preserving nonforgetting.",
            "FR-11 v7 did not produce positive nonforgetting growth.",
        ),
        "prm_process_agent": _score_prm_process_agent(exp1448),
        "ltlzinc_adapter": _scored(
            "exp1449",
            sources,
            missing_source_ids,
            exp1449.get("ltlzinc_adapter_ready") is True and (temporal_cases or 0.0) >= 20.0,
            "exp1449.ltlzinc_adapter_ready=true and at least 20 temporal cases generated",
            ("ltlzinc_adapter_ready", "temporal_cases_generated", "accepted_case_count"),
            "LTLZinc adapter generated at least 20 temporal constraint cases.",
            "LTLZinc adapter did not satisfy readiness or case-count gates.",
        ),
        "ebt_nrgpt_micro_baseline": _scored(
            "exp1450",
            sources,
            missing_source_ids,
            exp1450.get("energy_convergence_probe_complete") is True
            and exp1450.get("scale_recommendation") is not None,
            "exp1450.energy_convergence_probe_complete=true with scale/no-scale recommendation",
            ("energy_convergence_probe_complete", "scale_recommendation"),
            "EBT/NRGPT micro baseline recorded a no-scale recommendation.",
            "EBT/NRGPT micro baseline did not complete.",
        ),
        "rtl_lint_sim": _scored(
            "exp1451",
            sources,
            missing_source_ids,
            exp1451.get("rtl_lint_complete") is True
            and exp1451.get("simulation_complete") is True
            and exp1451.get("hardware_claim_allowed") is False,
            "exp1451.rtl_lint_complete=true or precise missing-tool blockers; no hardware claim",
            (
                "rtl_lint_complete",
                "simulation_complete",
                "hardware_claim_allowed",
                "hardware_execution_performed",
            ),
            "Discrete SB RTL lint and simulation completed without a hardware execution claim.",
            "RTL lint/simulation did not complete or made an unsupported hardware claim.",
        ),
        "retro": _criterion(
            MET,
            "exp1452.criteria_total=14 and honest carry-forward rules are recorded",
            ["results/experiment_1452_milestone_111_retro.json"],
            ["This final artifact records all 14 criteria and .112 carry-forward rules."],
            [],
            {"criteria_total": 14, "status": "complete"},
        ),
    }


def _source_checks(missing_source_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": exp_id,
            "path": f"results/{filename}",
            "exists": exp_id not in missing_source_ids,
        }
        for exp_id, filename in SOURCE_FILES.items()
    ]


def _missing_artifacts(missing_source_ids: set[str]) -> list[dict[str, str]]:
    return [
        {"experiment_id": exp_id, "path": f"results/{SOURCE_FILES[exp_id]}"}
        for exp_id in SOURCE_FILES
        if exp_id in missing_source_ids
    ]


def _prior_failure(
    exp_id: str,
    sources: Mapping[str, Mapping[str, Any]],
    fallback_verdict: str = "missing_artifact",
) -> dict[str, Any]:
    payload = sources.get(exp_id, {})
    return {
        "experiment_id": exp_id,
        "verdict": _verdict(payload) or fallback_verdict,
        "evidence_path": _source_path(exp_id),
    }


def _successful_tasks(
    sources: Mapping[str, Mapping[str, Any]],
    criteria: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    successes: list[dict[str, Any]] = []
    for criterion_id, result in criteria.items():
        if result["status"] != MET:
            continue
        source_id = CRITERION_SOURCE[criterion_id]
        successes.append(
            {
                "criterion": criterion_id,
                "experiment_id": source_id,
                "honest_verdict": "retro_complete"
                if source_id == "exp1452"
                else _verdict(sources.get(source_id, {})),
            }
        )
    return successes


def _blocked_tasks(
    sources: Mapping[str, Mapping[str, Any]],
    criteria: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    for criterion_id, result in criteria.items():
        if result["status"] != GATE_BLOCKED_WITH_EVIDENCE:
            continue
        source_id = CRITERION_SOURCE[criterion_id]
        blocked.append(
            {
                "criterion": criterion_id,
                "experiment_id": source_id,
                "honest_verdict": _verdict(sources.get(source_id, {}))
                or result["source_values"].get("honest_verdict"),
                "blocker": result["negative_evidence"][0],
            }
        )
    return blocked


def _retired_variants(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exp1439 = sources.get("exp1439", {})
    exp1442 = sources.get("exp1442", {})
    exp1448 = sources.get("exp1448", {})
    retired = [
        {
            "scope": "exact no-live-SOTA runtime repair scale path without fixing llama.cpp/CUDA/cache blockers",
            "prior_verdict": _verdict(exp1442),
            "source": _source_path("exp1442"),
            "retire_if_same_verdict": True,
        },
        {
            "scope": "exact repair v3 or 100-case pre-scale rerun while exp1442.local_sota_runtime_ready=false",
            "prior_verdict": "missing_artifact_gate_blocked_by_exp1442_exp1444",
            "source": "results/experiment_1443_live_sota_dccd_semctrl_repair_v3.json + "
            "results/experiment_1445_full_pipeline_v5_100case_prescale.json",
            "retire_if_same_verdict": True,
        },
        {
            "scope": "PRM v3 no-improvement prototype selector on saturated best-of-N candidate pool",
            "prior_verdict": _verdict(exp1448),
            "source": _source_path("exp1448"),
            "retire_if_same_verdict": True,
        },
    ]
    for item in exp1439.get("forbidden_exact_reruns", []):
        if isinstance(item, Mapping):
            retired.append(
                {
                    "scope": str(item.get("forbidden_scope", "")),
                    "prior_verdict": ", ".join(str(v) for v in item.get("prior_verdicts", [])),
                    "source": _source_path("exp1439", "forbidden_exact_reruns"),
                    "retire_if_same_verdict": item.get("retire_if_same_verdict", True),
                }
            )
    return retired


def _carry_forward_tracks(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": "live_sota_runtime_repair_gate",
            "title": "Fix live local SOTA GGUF runtime before any repair v3 or scale-up rerun.",
            "prior_failures": [_prior_failure("exp1442", sources)],
            "next_rule": (
                "Do not launch repair v3, energy reranking, or 100-case scale-up until "
                "a mandated local SOTA GGUF model completes live inference."
            ),
            "retire_if_same_verdict": True,
        },
        {
            "id": "repair_v3_and_prescale_gated_missing",
            "title": "Treat missing exp1443/exp1445 artifacts as gate-blocked, not successful.",
            "prior_failures": [
                _prior_failure("exp1443", sources, "missing_artifact_gate_blocked_by_exp1442"),
                _prior_failure(
                    "exp1445", sources, "missing_artifact_gate_blocked_by_exp1443_exp1444"
                ),
            ],
            "next_rule": (
                "A .112 repair scale task must name the live-runtime fix and cannot reuse "
                "the same gate-blocked path as success evidence."
            ),
            "retire_if_same_verdict": True,
        },
        {
            "id": "prm_process_agent_no_improvement",
            "title": "Retire saturated PRM selector pools unless the candidate pool changes.",
            "prior_failures": [_prior_failure("exp1448", sources)],
            "next_rule": (
                "Future PRM process-agent work must unsaturate the candidate pool or target "
                "false-acceptance reduction instead of repeating the same selection gate."
            ),
            "retire_if_same_verdict": True,
        },
    ]


def build_artifact(
    sources: Mapping[str, dict[str, Any]],
    missing_source_ids: list[str],
    roadmap_doc_present: bool,
    roadmap_yaml_present: bool,
    roadmap_next_present: bool,
) -> dict[str, Any]:
    """REQ-REPORT-038: score .111 criteria and assemble the terminal artifact.

    The milestone had several useful negative results. Keeping gate blocks out
    of `criteria_met` prevents the next planner from mistaking blocked evidence
    for a completed live-SOTA repair result.
    """

    missing = set(missing_source_ids)
    criteria = _score_criteria(sources, missing)
    criteria_met = sum(1 for result in criteria.values() if result["status"] == MET)
    criteria_total = len(CRITERION_SOURCE)
    successful_tasks = _successful_tasks(sources, criteria)
    blocked_tasks = _blocked_tasks(sources, criteria)
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "milestone": MILESTONE,
        "criteria_total": criteria_total,
        "criteria_met": criteria_met,
        "success_criteria_results": criteria,
        "successful_tasks": successful_tasks,
        "blocked_tasks": blocked_tasks,
        "lessons_learned": [
            (
                ".111 made real progress on non-runtime tracks: carry-forward discipline, "
                "spec coverage, Discrete SB source/lint/sim, FR-11 positive growth, "
                "LTLZinc cases, and EBT smoke evidence all closed their local gates."
            ),
            (
                "The live-SOTA repair chain did not reach headline evidence because "
                "exp1442 proved the local GGUF runtime was not ready; exp1443 and "
                "exp1445 are missing gated artifacts, and exp1444 correctly stopped on "
                "the failed upstream candidate-pool gate."
            ),
            (
                "FR-11 zero growth was not a DVI nonforgetting failure; policy and "
                "memory changes in exp1447 produced positive verified growth."
            ),
            (
                "PRM v3 produced decisive no-improvement evidence on a saturated "
                "candidate pool, so .112 should change the pool or target false-"
                "acceptance reduction instead of repeating the selector."
            ),
        ],
        "retired_variants": _retired_variants(sources),
        "carry_forward_tracks": _carry_forward_tracks(sources),
        "ops_docs_updated": False,
        "ops_docs_update_note": (
            "ops/status.md and ops/changelog.md intentionally left unchanged because the "
            "terminal stop rule delegates docs/status reconciliation to the conductor's "
            "separate Haiku step."
        ),
        "honest_verdict": (
            f"milestone_111_{criteria_met}_of_{criteria_total}_criteria_met_threshold_not_met_"
            "live_sota_runtime_gate_blocked_repair_scale_carry_forward"
        ),
        "missing_artifacts": _missing_artifacts(missing),
        "source_artifacts_checked": _source_checks(missing),
        "roadmap_inputs": {
            "change_proposal_path": "openspec/change-proposals/research-roadmap-vNEXT.md",
            "change_proposal_present": roadmap_doc_present,
            "active_roadmap_yaml_path": "research-roadmap.yaml",
            "active_roadmap_yaml_present": roadmap_yaml_present,
            "requested_research_roadmap_next_path": "research-roadmap-next.yaml",
            "requested_research_roadmap_next_present": roadmap_next_present,
        },
        "operational_notes": {
            "research_roadmap_next_yaml_missing": not roadmap_next_present,
            "scripts_research_conductor_modified": False,
            "research_roadmap_yaml_modified": False,
            "score_rule": "criteria_met counts only status=met; gate_blocked_with_evidence is not success.",
        },
    }


def run(root: Path | str = REPO_ROOT, out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    root_path = Path(root)
    write_in_progress_artifact(out_path)
    sources, missing = _load_sources(root_path / "results")
    artifact = build_artifact(
        sources,
        missing,
        roadmap_doc_present=(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ).exists(),
        roadmap_yaml_present=(root_path / "research-roadmap.yaml").exists(),
        roadmap_next_present=(root_path / "research-roadmap-next.yaml").exists(),
    )
    return _write_json(Path(out_path), artifact)
