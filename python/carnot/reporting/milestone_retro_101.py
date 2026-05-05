"""Build the Exp 1308 milestone .101 retrospective artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1308_milestone_retro_101.json"

EXPERIMENT = "1308_milestone_retro_101"
SCHEMA = "milestone_retro_v6"
RUN_DATE = "20260505"
MILESTONE = "2026.04.101"

MET = "MET"
BLOCKED = "BLOCKED"
GATED = "GATED"
MISSING = "MISSING"
FAILED = "FAILED"

SOURCE_FILES = {
    1296: "experiment_1296_prior_failures_activation_audit.json",
    1297: "experiment_1297_sota_gguf_cache_provenance_preflight_v2.json",
    1298: "experiment_1298_sota_answer_stability_falcon_audit.json",
    1299: "experiment_1299_triggered_certificate_extraction_v3.json",
    1300: "experiment_1300_semantic_routing_v2.json",
    1301: "experiment_1301_safe_prefix_cactus_acceptance_v3.json",
    1302: "experiment_1302_skill_graph_promotion_demotion_v2.json",
    1303: "experiment_1303_querybandits_ngc_online_memory_policy.json",
    1304: "experiment_1304_grpo_vprm_v10_sota_gated.json",
    1305: "experiment_1305_hardnetpp_dsp_feasibility_stop_policy.json",
    1306: "experiment_1306_ebt_arm_ebm_cot_energy_bridge_audit_v2.json",
    1307: "experiment_1307_arxiv_v10_hold_receipt_v2.json",
}

CRITERIA: tuple[tuple[str, int | None], ...] = (
    ("prior_failures_activation_audit", 1296),
    ("sota_gguf_cache_provenance_preflight_v2", 1297),
    ("sota_answer_stability_falcon_audit", 1298),
    ("triggered_certificate_extraction_v3", 1299),
    ("semantic_routing_v2", 1300),
    ("safe_prefix_cactus_acceptance_v3", 1301),
    ("skill_graph_promotion_demotion_v2", 1302),
    ("querybandits_ngc_online_memory_policy", 1303),
    ("grpo_vprm_v10_sota_gated", 1304),
    ("hardnetpp_dsp_feasibility_stop_policy", 1305),
    ("ebt_arm_ebm_cot_energy_bridge_audit_v2", 1306),
    ("arxiv_v10_hold_receipt_v2", 1307),
    ("retro_101_complete", None),
)

CRITERION_NAMES = tuple(name for name, _exp_id in CRITERIA)


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-027: write a durable placeholder before source evaluation."""

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "milestone": MILESTONE,
            "status": "in_progress",
            "criteria_total": len(CRITERIA),
            "criteria_met": 0,
            "carry_forward_tasks": [],
            "docs_reconciled": False,
            "retro_complete": False,
            "honest_verdict": "in_progress",
        },
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status", "")).lower()


def _honest_verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict", "")).lower()


def _is_number(value: object) -> bool:
    return isinstance(value, int | float)


def _at_least(value: object, threshold: float) -> bool:
    return _is_number(value) and float(value) >= threshold


def _positive(value: object) -> bool:
    return _is_number(value) and float(value) > 0.0


def _is_conductor_dependency_gate(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "blocked" and payload.get("blocked_at_layer") == "conductor_pre_gate"


def _cached_sota_ready(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return sources.get(1297, {}).get("cached_sota_ready") is True


def _grammar_backend_available(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return sources.get(1296, {}).get("exp1283_grammar_backend_available") is True


def _answer_stability_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _at_least(sources.get(1298, {}).get("answer_stability_score"), 0.6)


def _certificate_parse_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _at_least(sources.get(1299, {}).get("certificate_parse_rate"), 0.8)


def _semantic_routing_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _at_least(sources.get(1300, {}).get("semantic_routing_coverage"), 0.5)


def _headline_cert_allowed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return sources.get(1299, {}).get("headline_result_allowed") is True


def _self_learning_improved(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _positive(sources.get(1303, {}).get("self_learning_delta_overall"))


def _gate_unmet(name: str, sources: Mapping[int, Mapping[str, Any]]) -> bool:
    if name == "sota_answer_stability_falcon_audit":
        return not _cached_sota_ready(sources)
    if name == "triggered_certificate_extraction_v3":
        return not (
            _cached_sota_ready(sources)
            and _grammar_backend_available(sources)
            and _answer_stability_passed(sources)
        )
    if name == "semantic_routing_v2":
        return not _certificate_parse_passed(sources)
    if name == "safe_prefix_cactus_acceptance_v3":
        return not (_certificate_parse_passed(sources) and _semantic_routing_passed(sources))
    if name == "grpo_vprm_v10_sota_gated":
        return not (_headline_cert_allowed(sources) and _self_learning_improved(sources))
    return False


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "complete"


def _classify_criterion(
    name: str,
    exp_id: int | None,
    sources: Mapping[int, Mapping[str, Any]],
) -> str:
    if exp_id is None:
        return MET
    if exp_id not in sources:
        return GATED if _gate_unmet(name, sources) else MISSING

    payload = sources[exp_id]
    if _is_conductor_dependency_gate(payload) and _gate_unmet(name, sources):
        return GATED
    if _status(payload) == "blocked":
        return BLOCKED

    if name == "prior_failures_activation_audit":
        return (
            MET
            if _is_complete(payload)
            and payload.get("prior_failures_coverage_ok") is True
            and payload.get("roadmap_gate_audit_passed") is True
            else FAILED
        )
    if name == "sota_gguf_cache_provenance_preflight_v2":
        exact_blocker = bool(payload.get("missing_models")) or bool(payload.get("blocked_reason"))
        return (
            MET
            if _is_complete(payload)
            and payload.get("provenance_ok") is True
            and (payload.get("cached_sota_ready") is True or exact_blocker)
            else FAILED
        )
    if name == "sota_answer_stability_falcon_audit":
        return (
            MET
            if _is_complete(payload) and _at_least(payload.get("answer_stability_score"), 0.6)
            else FAILED
        )
    if name == "triggered_certificate_extraction_v3":
        measured = _is_number(payload.get("certificate_parse_rate")) or bool(
            payload.get("precise_blocker")
        )
        return MET if _is_complete(payload) and measured else FAILED
    if name == "semantic_routing_v2":
        return (
            MET
            if _is_complete(payload) and _is_number(payload.get("semantic_routing_coverage"))
            else FAILED
        )
    if name == "safe_prefix_cactus_acceptance_v3":
        return (
            MET
            if _is_complete(payload) and _is_number(payload.get("cactus_acceptance_rate"))
            else FAILED
        )
    if name == "skill_graph_promotion_demotion_v2":
        return (
            MET
            if _is_complete(payload)
            and payload.get("memory_update_written") is True
            and _positive(payload.get("skill_graph_candidate_count"))
            and "promoted_memory_count" in payload
            and "demoted_memory_count" in payload
            and "expired_memory_count" in payload
            else FAILED
        )
    if name == "querybandits_ngc_online_memory_policy":
        return (
            MET
            if _is_complete(payload)
            and _is_number(payload.get("bandit_regret"))
            and _is_number(payload.get("accepted_violation_delta"))
            and _positive(payload.get("self_learning_delta_overall"))
            else FAILED
        )
    if name == "grpo_vprm_v10_sota_gated":
        return (
            MET if _is_complete(payload) and _is_number(payload.get("grpo_vprm_delta")) else FAILED
        )
    if name == "hardnetpp_dsp_feasibility_stop_policy":
        return (
            MET
            if _is_complete(payload) and payload.get("feasibility_stop_policy_written") is True
            else FAILED
        )
    if name == "ebt_arm_ebm_cot_energy_bridge_audit_v2":
        return (
            MET
            if _is_complete(payload) and payload.get("energy_bridge_completed") is True
            else FAILED
        )
    if name == "arxiv_v10_hold_receipt_v2":
        terminal_publication = (
            payload.get("arxiv_receipt_present") is True
            or payload.get("operator_hold_active") is True
            or bool(payload.get("blocker"))
        )
        return (
            MET
            if _is_complete(payload)
            and payload.get("credentialed_submission_attempted") is False
            and terminal_publication
            else FAILED
        )
    raise AssertionError(f"unknown criterion: {name}")  # pragma: no cover


def _build_criteria_results(sources: Mapping[int, Mapping[str, Any]]) -> dict[str, str]:
    return {name: _classify_criterion(name, exp_id, sources) for name, exp_id in CRITERIA}


def _source_artifacts_checked(sources: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": exp_id,
            "path": f"results/{filename}",
            "loaded": exp_id in sources,
            "status": sources.get(exp_id, {}).get("status"),
            "honest_verdict": sources.get(exp_id, {}).get("honest_verdict"),
        }
        for exp_id, filename in SOURCE_FILES.items()
    ]


def _gated_or_skipped_tasks(
    sources: Mapping[int, Mapping[str, Any]],
    criteria_results: Mapping[str, str],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for name, exp_id in CRITERIA:
        if exp_id is None or criteria_results[name] != GATED:
            continue
        payload = sources.get(exp_id, {})
        tasks.append(
            {
                "experiment_id": exp_id,
                "criterion": name,
                "status": payload.get("status", "missing"),
                "honest_verdict": payload.get("honest_verdict", "missing"),
                "gate_check_summary": payload.get("gate_check_summary")
                or "upstream milestone gate did not open",
            }
        )
    return tasks


def _activation_failures(
    sources: Mapping[int, Mapping[str, Any]],
    criteria_results: Mapping[str, str],
) -> list[dict[str, Any]]:
    payload = sources.get(1296, {})
    if criteria_results["prior_failures_activation_audit"] == MET:
        return []
    return [
        {
            "experiment_id": 1296,
            "criterion": "prior_failures_activation_audit",
            "honest_verdict": payload.get("honest_verdict", "missing"),
            "details": list(payload.get("activation_blockers") or []),
        }
    ]


def _scientific_negative_results(
    sources: Mapping[int, Mapping[str, Any]],
    criteria_results: Mapping[str, str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    repair = sources.get(1305, {})
    if criteria_results[
        "hardnetpp_dsp_feasibility_stop_policy"
    ] == MET and "not a learned general stop rule" in _honest_verdict(repair):
        results.append(
            {
                "experiment_id": 1305,
                "criterion": "hardnetpp_dsp_feasibility_stop_policy",
                "status": MET,
                "honest_verdict": repair.get("honest_verdict"),
                "interpretation": "terminal repair-policy artifact; DSP remains marginal and the rule is not a learned general stop policy",
            }
        )
    for name, exp_id in CRITERIA:
        if exp_id is None or exp_id == 1296 or criteria_results[name] != FAILED:
            continue
        payload = sources.get(exp_id, {})
        results.append(
            {
                "experiment_id": exp_id,
                "criterion": name,
                "status": FAILED,
                "honest_verdict": payload.get("honest_verdict", "missing"),
                "interpretation": "terminal source artifact did not satisfy its planned scientific criterion",
            }
        )
    return results


def _missing_models_text(sources: Mapping[int, Mapping[str, Any]]) -> str:
    missing_models = sources.get(1297, {}).get("missing_models") or ["required SOTA GGUF model"]
    return ", ".join(str(model) for model in missing_models)


def _carry_forward_tasks(sources: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    missing_models = _missing_models_text(sources)
    return [
        {
            "task_id": "sota_gguf_cache_readiness",
            "reason": "The SOTA certificate path stayed closed because the local mandated GGUF cache was incomplete.",
            "prior_failures": [
                {
                    "experiment_id": "exp1297-sota-gguf-cache-provenance-preflight-v2",
                    "verdict": "sota_gguf_cache_not_ready",
                    "addressed_by": f"Provision or deliberately replace the missing {missing_models} cache entry before rerunning SOTA certificate work.",
                    "retire_if_same_verdict": False,
                }
            ],
        },
        {
            "task_id": "triggered_certificate_path",
            "reason": "Answer-stability and certificate extraction did not run because cached_sota_ready was false.",
            "prior_failures": [
                {
                    "experiment_id": "exp1298-sota-answer-stability-falcon-audit",
                    "verdict": "blocked_gate_check_failed",
                    "addressed_by": "Rerun only after exp1297.cached_sota_ready == true and record answer_stability_score.",
                    "retire_if_same_verdict": False,
                },
                {
                    "experiment_id": "exp1299-triggered-certificate-extraction-v3",
                    "verdict": "missing_gated_by_sota_cache_and_answer_stability",
                    "addressed_by": "Produce certificate_parse_rate, headline_result_allowed, grammar cost, truthfulness, and FALCON repair metrics after SOTA readiness gates open.",
                    "retire_if_same_verdict": False,
                },
            ],
        },
        {
            "task_id": "semantic_routing_and_cactus_acceptance",
            "reason": "Routing and safe-prefix acceptance remained downstream of the missing certificate parse-rate gate.",
            "prior_failures": [
                {
                    "experiment_id": "exp1300-semantic-routing-v2",
                    "verdict": "blocked_gate_check_failed",
                    "addressed_by": "Run after exp1299.certificate_parse_rate >= 0.8 and write semantic_routing_coverage.",
                    "retire_if_same_verdict": False,
                },
                {
                    "experiment_id": "exp1301-safe-prefix-cactus-acceptance-v3",
                    "verdict": "missing_gated_by_certificate_parse_and_semantic_routing",
                    "addressed_by": "Run after exp1299.certificate_parse_rate >= 0.8 and exp1300.semantic_routing_coverage >= 0.5.",
                    "retire_if_same_verdict": False,
                },
            ],
        },
        {
            "task_id": "grpo_vprm_headline_gate",
            "reason": "Online self-learning improved, but GRPO/VPRM headline learning stayed gated by the absent SOTA certificate path.",
            "prior_failures": [
                {
                    "experiment_id": "exp1304-grpo-vprm-v10-sota-gated",
                    "verdict": "missing_gated_by_sota_certificate_headline_result",
                    "addressed_by": "Run only after exp1299.headline_result_allowed == true and exp1303.self_learning_delta_overall > 0.0.",
                    "retire_if_same_verdict": False,
                }
            ],
        },
        {
            "task_id": "repair_policy_generalization",
            "reason": "The repair stop policy is useful as an operator gate, but it is not yet a learned general stop rule.",
            "prior_failures": [
                {
                    "experiment_id": "exp1305-hardnetpp-dsp-feasibility-stop-policy",
                    "verdict": "dsp_feasibility_marginal_not_learned_general_stop_rule",
                    "addressed_by": "Convert the replay stop policy into a learned/generalized policy only after new non-replay evidence is available.",
                    "retire_if_same_verdict": False,
                }
            ],
        },
        {
            "task_id": "publication_hold",
            "reason": "Publication remains terminally recorded but held by operator state rather than a local arXiv receipt.",
            "prior_failures": [
                {
                    "experiment_id": "exp1307-arxiv-v10-hold-receipt-v2",
                    "verdict": "operator_hold_active_no_local_arxiv_receipt",
                    "addressed_by": "Keep publication tasks terminal by recording the operator hold or a local receipt; do not attempt credentialed submission without operator approval.",
                    "retire_if_same_verdict": False,
                }
            ],
        },
    ]


def _milestone_summary(sources: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "activation_audit_passed": sources.get(1296, {}).get("prior_failures_coverage_ok") is True,
        "cached_sota_ready": sources.get(1297, {}).get("cached_sota_ready") is True,
        "missing_sota_models": list(sources.get(1297, {}).get("missing_models") or []),
        "skill_graph_candidate_count": sources.get(1302, {}).get("skill_graph_candidate_count"),
        "self_learning_delta_overall": sources.get(1303, {}).get("self_learning_delta_overall"),
        "accepted_violation_delta": sources.get(1303, {}).get("accepted_violation_delta"),
        "stop_policy_precision": sources.get(1305, {}).get("stop_policy_precision"),
        "energy_bridge_completed": sources.get(1306, {}).get("energy_bridge_completed") is True,
        "publication_state": sources.get(1307, {}).get("publication_state", "missing"),
    }


def build_artifact(
    sources: Mapping[int, Mapping[str, Any]],
    *,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """SCENARIO-REPORT-027: synthesize .101 scoring from source artifacts."""

    criteria_results = _build_criteria_results(sources)
    criteria_met = sum(1 for status in criteria_results.values() if status == MET)
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "metadata": {
            "project_root": str(REPO_ROOT),
            "run_date": run_date,
            "source_roadmap": "research-roadmap.yaml",
            "requested_next_roadmap": "research-roadmap-next.yaml",
        },
        "milestone": MILESTONE,
        "status": "complete",
        "criteria_results": criteria_results,
        "criteria_met": criteria_met,
        "criteria_total": len(CRITERIA),
        "source_artifacts_checked": _source_artifacts_checked(sources),
        "milestone_summary": _milestone_summary(sources),
        "activation_failures": _activation_failures(sources, criteria_results),
        "gated_or_skipped_tasks": _gated_or_skipped_tasks(sources, criteria_results),
        "scientific_negative_results": _scientific_negative_results(sources, criteria_results),
        "carry_forward_tasks": _carry_forward_tasks(sources),
        "docs_reconciled": False,
        "docs_reconciliation_note": (
            "ops/status.md, ops/changelog.md, and _bmad/traceability.md were left untouched "
            "because the conductor stop rule delegates ops reconciliation to the following Haiku step."
        ),
        "findings_summary": (
            f"Milestone .101 met {criteria_met} of 13 criteria. Activation hygiene passed and "
            "self-learning, repair, bridge, publication-state, and retrospective artifacts reached terminal states; "
            "SOTA certificate work stayed gated by the missing mandated GGUF cache entry."
        ),
        "retro_complete": True,
        "honest_verdict": f"milestone_101_{criteria_met}_of_13_criteria_met",
    }


def run(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """Load .101 source artifacts, write the Exp 1308 result JSON, and return it."""

    results_path = Path(results_dir)
    target = Path(out_path)
    write_in_progress_artifact(target)
    sources = {
        exp_id: loaded
        for exp_id, filename in SOURCE_FILES.items()
        if (loaded := _load_json(results_path / filename)) is not None
    }
    return _write_json(target, build_artifact(sources))
