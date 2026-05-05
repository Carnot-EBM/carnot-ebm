"""Build the Exp 1322 milestone .102 retrospective artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1322_milestone_retro_102.json"

EXPERIMENT = "1322_milestone_retro_102"
SCHEMA = "milestone_retro_v7"
RUN_DATE = "20260505"
MILESTONE = "2026.04.102"

MET = "MET"
GATED = "GATED"
MISSING = "MISSING"
BLOCKED = "BLOCKED"
FAILED = "FAILED"

SOURCE_FILES = {
    1309: "experiment_1309_sota_gguf_pair_resolver_repair.json",
    1310: "experiment_1310_sota_gguf_llamacpp_smoke_load.json",
    1311: "experiment_1311_sota_constraintbench_satquest_answer_stability.json",
    1312: "experiment_1312_triggered_certificate_extraction_dccd_gbnf.json",
    1313: "experiment_1313_constrainprompt_nsvif_semantic_validator_mus_repair.json",
    1314: "experiment_1314_beaver_lite_cactus_safe_prefix_acceptance.json",
    1315: "experiment_1315_continuous_self_learning_cerce_nonforgetting_audit.json",
    1316: "experiment_1316_dvi_certificate_tail_online_update.json",
    1317: "experiment_1317_grpo_vprm_v11_headline_gate.json",
    1318: "experiment_1318_hardnetpp_dsp_learned_stop_policy.json",
    1319: "experiment_1319_kan_hardware_complexity_audit.json",
    1320: "experiment_1320_pbit_sampler_portability_packet.json",
    1321: "experiment_1321_publication_hold_related_work_delta_v11.json",
}

CRITERIA: tuple[tuple[str, int | None], ...] = (
    ("sota_gguf_pair_resolver_repair", 1309),
    ("sota_gguf_llamacpp_smoke_load", 1310),
    ("sota_constraintbench_satquest_answer_stability", 1311),
    ("triggered_certificate_extraction_dccd_gbnf", 1312),
    ("constrainprompt_nsvif_semantic_validator_mus_repair", 1313),
    ("beaver_lite_cactus_safe_prefix_acceptance", 1314),
    ("continuous_self_learning_cerce_nonforgetting_audit", 1315),
    ("dvi_certificate_tail_online_update", 1316),
    ("grpo_vprm_v11_headline_gate", 1317),
    ("hardnetpp_dsp_learned_stop_policy", 1318),
    ("kan_hardware_complexity_audit", 1319),
    ("pbit_sampler_portability_packet", 1320),
    ("publication_hold_related_work_delta_v11", 1321),
    ("retro_102_complete", None),
)

CRITERION_NAMES = tuple(name for name, _exp_id in CRITERIA)


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-028: write a durable placeholder before source evaluation."""

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
            "sota_runtime_recovered": False,
            "certificate_path_headline_ready": False,
            "continuous_self_learning_advanced": False,
            "repair_generalization_advanced": False,
            "hardware_claims_honest": False,
            "publication_state": "in_progress",
            "carry_forward_tasks": [],
            "retro_complete": False,
            "honest_verdict": "milestone_102_in_progress",
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


def _nonpositive(value: object) -> bool:
    return _is_number(value) and float(value) <= 0.0


def _nonempty(value: object) -> bool:
    return bool(value)


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "complete"


def _is_conductor_dependency_gate(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "blocked" and payload.get("blocked_at_layer") == "conductor_pre_gate"


def _certificate_parse_gate_open(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _at_least(sources.get(1312, {}).get("certificate_parse_rate"), 0.75)


def _validator_gate_open(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _at_least(sources.get(1313, {}).get("validator_execution_pass_rate"), 0.5)


def _nonforgetting_gate_open(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _at_least(sources.get(1315, {}).get("nonforgetting_certificate_rate"), 0.9)


def _gate_unmet(name: str, sources: Mapping[int, Mapping[str, Any]]) -> bool:
    if name == "constrainprompt_nsvif_semantic_validator_mus_repair":
        return not _certificate_parse_gate_open(sources)
    if name == "beaver_lite_cactus_safe_prefix_acceptance":
        return not (_certificate_parse_gate_open(sources) and _validator_gate_open(sources))
    if name == "dvi_certificate_tail_online_update":
        return not (_certificate_parse_gate_open(sources) and _nonforgetting_gate_open(sources))
    if name == "grpo_vprm_v11_headline_gate":
        source_1312 = sources.get(1312, {})
        source_1315 = sources.get(1315, {})
        return not (
            source_1312.get("headline_result_allowed") is True
            and _positive(source_1315.get("self_learning_delta_overall"))
            and _nonforgetting_gate_open(sources)
        )
    return False


def _criterion_met(name: str, payload: Mapping[str, Any]) -> bool:
    if name == "sota_gguf_pair_resolver_repair":
        return (
            _is_complete(payload)
            and payload.get("sota_pair_ready") is True
            and _at_least(payload.get("cached_pair_specs_count"), 2)
            and payload.get("headline_result_possible") is True
            and payload.get("focused_tests_passed") is True
        )
    if name == "sota_gguf_llamacpp_smoke_load":
        return (
            _is_complete(payload)
            and _at_least(payload.get("models_loaded"), 2)
            and payload.get("llama_cpp_import_ok") is True
            and _at_least(payload.get("model_specs_count"), 2)
            and _positive(payload.get("tokens_per_second"))
            and payload.get("headline_result_possible") is True
        )
    if name == "sota_constraintbench_satquest_answer_stability":
        return (
            _is_complete(payload)
            and _at_least(payload.get("answer_stability_score"), 0.6)
            and _is_number(payload.get("pysat_verified_rate"))
            and _is_number(payload.get("unknown_or_abstain_rate"))
            and payload.get("headline_result_allowed") is True
        )
    if name == "triggered_certificate_extraction_dccd_gbnf":
        return (
            _is_complete(payload)
            and _is_number(payload.get("certificate_parse_rate"))
            and _is_number(payload.get("certificate_truthfulness_rate"))
            and _is_number(payload.get("dccd_delta_over_grammar_only"))
            and _is_number(payload.get("repair_success_rate"))
            and _nonempty(payload.get("grammar_projection_tax_proxy"))
            and payload.get("headline_result_allowed") is True
        )
    if name == "constrainprompt_nsvif_semantic_validator_mus_repair":
        return (
            _is_complete(payload)
            and _positive(payload.get("compiled_validator_count"))
            and _at_least(payload.get("validator_execution_pass_rate"), 0.5)
            and _is_number(payload.get("semantic_violation_reduction"))
            and _is_number(payload.get("mus_repair_hint_count"))
            and "residual_drift_cases" in payload
            and _is_number(payload.get("unknown_or_abstain_rate"))
        )
    if name == "beaver_lite_cactus_safe_prefix_acceptance":
        return (
            _is_complete(payload)
            and _is_number(payload.get("low_risk_acceptance_rate"))
            and _nonpositive(payload.get("false_acceptance_rate"))
            and _is_number(payload.get("safe_prefix_repair_delta"))
            and _is_number(payload.get("full_verifier_call_reduction"))
            and _is_number(payload.get("risk_bound_proxy"))
            and payload.get("headline_result_allowed") is True
        )
    if name == "continuous_self_learning_cerce_nonforgetting_audit":
        return (
            _is_complete(payload)
            and _at_least(payload.get("nonforgetting_certificate_rate"), 0.9)
            and payload.get("memory_regression_count") == 0
            and _positive(payload.get("self_learning_delta_overall"))
            and _nonpositive(payload.get("accepted_violation_delta"))
            and _is_number(payload.get("lagrangian_violation_penalty"))
            and _positive(payload.get("promoted_memory_count"))
            and _is_number(payload.get("demoted_memory_count"))
        )
    if name == "dvi_certificate_tail_online_update":
        return (
            _is_complete(payload)
            and _is_number(payload.get("drafter_acceptance_delta"))
            and _nonpositive(payload.get("accepted_violation_delta"))
            and _positive(payload.get("online_update_count"))
            and payload.get("nonforgetting_preserved") is True
            and payload.get("lossless_acceptance_claim_allowed") is True
            and payload.get("headline_result_allowed") is True
        )
    if name == "grpo_vprm_v11_headline_gate":
        return (
            _is_complete(payload)
            and _positive(payload.get("grpo_vprm_delta"))
            and _is_number(payload.get("verifier_feedback_token_mask_delta"))
            and payload.get("nonforgetting_preserved") is True
            and _positive(payload.get("self_verification_gain"))
            and payload.get("headline_result_allowed") is True
        )
    if name == "hardnetpp_dsp_learned_stop_policy":
        return (
            _is_complete(payload)
            and payload.get("learned_stop_policy_written") is True
            and _positive((payload.get("generalization_split") or {}).get("held_out_count"))
            and _is_number(payload.get("stop_policy_precision"))
            and _is_number(payload.get("stop_policy_recall"))
            and _is_number(payload.get("hardnetpp_delta_over_replay_policy"))
            and _is_number(payload.get("dsp_feasibility_auc"))
        )
    if name == "kan_hardware_complexity_audit":
        execution = payload.get("hardware_execution") or {}
        return (
            _is_complete(payload)
            and _positive(payload.get("rm_per_inference"))
            and _positive(payload.get("bop_per_inference"))
            and _positive(payload.get("nabs_per_inference"))
            and _positive(payload.get("lookup_table_bytes"))
            and payload.get("hardware_claim_allowed") is False
            and execution.get("fpga_execution") is False
            and execution.get("npu_execution") is False
            and execution.get("analog_execution") is False
        )
    if name == "pbit_sampler_portability_packet":
        return (
            _is_complete(payload)
            and payload.get("dual_bram_mapping_ready") is True
            and _nonempty(payload.get("reuse_factor_sweep"))
            and _nonempty(payload.get("dac_bits_sweep"))
            and _is_number(payload.get("kl_to_cpu_gibbs"))
            and payload.get("vivado_required_for_next_step") is True
            and payload.get("hardware_claim_allowed") is False
        )
    if name == "publication_hold_related_work_delta_v11":
        return (
            _is_complete(payload)
            and payload.get("publication_state") == "operator_hold"
            and payload.get("operator_hold_active") is True
            and payload.get("credentialed_submission_attempted") is False
            and payload.get("related_work_delta_written") is True
            and _positive(payload.get("new_references_count"))
        )
    raise AssertionError(f"unknown criterion: {name}")  # pragma: no cover


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
    return MET if _criterion_met(name, payload) else FAILED


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


def _gated_or_missing_tasks(
    sources: Mapping[int, Mapping[str, Any]],
    criteria_results: Mapping[str, str],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for name, exp_id in CRITERIA:
        if exp_id is None or criteria_results[name] == MET:
            continue
        payload = sources.get(exp_id, {})
        tasks.append(
            {
                "experiment_id": exp_id,
                "criterion": name,
                "status": criteria_results[name],
                "source_status": payload.get("status", "missing"),
                "honest_verdict": payload.get("honest_verdict", "missing"),
                "gate_check_summary": payload.get("gate_check_summary")
                or "source artifact missing or planned metric not satisfied",
            }
        )
    return tasks


def _certificate_path_headline_ready(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    source_1312 = sources.get(1312, {})
    source_1314 = sources.get(1314, {})
    return (
        source_1312.get("headline_result_allowed") is True
        and _certificate_parse_gate_open(sources)
        and _validator_gate_open(sources)
        and source_1314.get("headline_result_allowed") is True
        and _nonpositive(source_1314.get("false_acceptance_rate"))
    )


def _sota_runtime_recovered(criteria_results: Mapping[str, str]) -> bool:
    return (
        criteria_results["sota_gguf_pair_resolver_repair"] == MET
        and criteria_results["sota_gguf_llamacpp_smoke_load"] == MET
    )


def _continuous_self_learning_advanced(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    payload = sources.get(1315, {})
    return (
        _at_least(payload.get("nonforgetting_certificate_rate"), 0.9)
        and payload.get("memory_regression_count") == 0
        and _positive(payload.get("self_learning_delta_overall"))
        and _nonpositive(payload.get("accepted_violation_delta"))
    )


def _repair_generalization_advanced(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    payload = sources.get(1318, {})
    split = payload.get("generalization_split") or {}
    return (
        _is_complete(payload)
        and payload.get("learned_stop_policy_written") is True
        and _positive(split.get("held_out_count"))
        and _is_number(payload.get("stop_policy_precision"))
        and _is_number(payload.get("stop_policy_recall"))
    )


def _hardware_claims_honest(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    source_1319 = sources.get(1319, {})
    source_1320 = sources.get(1320, {})
    execution = source_1319.get("hardware_execution") or {}
    return (
        _is_complete(source_1319)
        and _is_complete(source_1320)
        and source_1319.get("hardware_claim_allowed") is False
        and source_1320.get("hardware_claim_allowed") is False
        and execution.get("fpga_execution") is False
        and execution.get("npu_execution") is False
        and execution.get("analog_execution") is False
        and source_1320.get("vivado_required_for_next_step") is True
    )


def _format_number(value: object) -> str:
    return str(value) if _is_number(value) else "missing"


def _carry_forward_tasks(
    sources: Mapping[int, Mapping[str, Any]],
    criteria_results: Mapping[str, str],
) -> list[dict[str, Any]]:
    source_1312 = sources.get(1312, {})
    source_1313 = sources.get(1313, {})
    source_1316 = sources.get(1316, {})
    source_1318 = sources.get(1318, {})
    source_1321 = sources.get(1321, {})
    parse_rate = _format_number(source_1312.get("certificate_parse_rate"))
    tasks: list[dict[str, Any]] = []

    if not _certificate_path_headline_ready(sources):
        tasks.append(
            {
                "task_id": "certificate_path_headline_readiness",
                "reason": (
                    "The SOTA certificate measurement completed, but the parse-rate gate "
                    "stayed below 0.75 and semantic validator/safe-prefix work did not finish."
                ),
                "prior_failures": [
                    {
                        "experiment_id": "exp1312-triggered-certificate-extraction-dccd-gbnf",
                        "verdict": source_1312.get("honest_verdict", "missing"),
                        "addressed_by": (
                            f"Raise certificate_parse_rate from {parse_rate} to at least 0.75, "
                            "then rerun semantic validators and safe-prefix acceptance from the "
                            "parsed certificate corpus."
                        ),
                        "retire_if_same_verdict": False,
                    },
                    {
                        "experiment_id": "exp1313-constrainprompt-nsvif-semantic-validator-mus-repair",
                        "verdict": source_1313.get("honest_verdict", "missing"),
                        "addressed_by": (
                            "Run the validator/MUS repair audit after the certificate parse gate opens."
                        ),
                        "retire_if_same_verdict": False,
                    },
                    {
                        "experiment_id": "exp1314-beaver-lite-cactus-safe-prefix-acceptance",
                        "verdict": criteria_results["beaver_lite_cactus_safe_prefix_acceptance"],
                        "addressed_by": (
                            "Run safe-prefix acceptance only after parseable certificates and "
                            "executable validators are both available."
                        ),
                        "retire_if_same_verdict": False,
                    },
                ],
            }
        )

    if criteria_results["dvi_certificate_tail_online_update"] != MET:
        tasks.append(
            {
                "task_id": "dvi_certificate_tail_update",
                "reason": (
                    "Non-forgetting passed, but DVI certificate-tail updates stayed gated by "
                    "the same certificate parse-rate miss."
                ),
                "prior_failures": [
                    {
                        "experiment_id": "exp1316-dvi-certificate-tail-online-update",
                        "verdict": source_1316.get("honest_verdict", "missing"),
                        "addressed_by": (
                            "Rerun after exp1312.certificate_parse_rate >= 0.75 while preserving "
                            "exp1315.nonforgetting_certificate_rate >= 0.9."
                        ),
                        "retire_if_same_verdict": False,
                    }
                ],
            }
        )

    if "not a broad general stop rule" in _honest_verdict(source_1318):
        tasks.append(
            {
                "task_id": "repair_generalization_breadth",
                "reason": (
                    "The learned stop policy advanced to a held-out replay split but did not "
                    "prove broad repair-policy generalization."
                ),
                "prior_failures": [
                    {
                        "experiment_id": "exp1318-hardnetpp-dsp-learned-stop-policy",
                        "verdict": source_1318.get("honest_verdict", "missing"),
                        "addressed_by": (
                            "Add non-replay or fresh validator-backed repair cases before claiming "
                            "a broad general stop rule."
                        ),
                        "retire_if_same_verdict": False,
                    }
                ],
            }
        )

    if source_1321.get("publication_state", "missing") != "submitted":
        tasks.append(
            {
                "task_id": "publication_operator_hold",
                "reason": "Publication state is terminally recorded, but no credentialed submission occurred.",
                "prior_failures": [
                    {
                        "experiment_id": "exp1321-publication-hold-related-work-delta-v11",
                        "verdict": source_1321.get("honest_verdict", "missing"),
                        "addressed_by": (
                            "Keep future publication tasks explicit about operator hold state and "
                            "avoid credentialed upload attempts without approval."
                        ),
                        "retire_if_same_verdict": False,
                    }
                ],
            }
        )

    return tasks


def _summary_fields(
    sources: Mapping[int, Mapping[str, Any]],
    criteria_results: Mapping[str, str],
) -> dict[str, Any]:
    source_1312 = sources.get(1312, {})
    source_1315 = sources.get(1315, {})
    source_1318 = sources.get(1318, {})
    return {
        "sota_runtime_recovered": _sota_runtime_recovered(criteria_results),
        "certificate_path_headline_ready": _certificate_path_headline_ready(sources),
        "continuous_self_learning_advanced": _continuous_self_learning_advanced(sources),
        "repair_generalization_advanced": _repair_generalization_advanced(sources),
        "hardware_claims_honest": _hardware_claims_honest(sources),
        "publication_state": sources.get(1321, {}).get("publication_state", "missing"),
        "milestone_summary": {
            "certificate_parse_rate": source_1312.get("certificate_parse_rate"),
            "certificate_truthfulness_rate": source_1312.get("certificate_truthfulness_rate"),
            "nonforgetting_certificate_rate": source_1315.get("nonforgetting_certificate_rate"),
            "self_learning_delta_overall": source_1315.get("self_learning_delta_overall"),
            "repair_generalization_scope": (
                "replay_distribution_held_out"
                if "replay-distribution" in _honest_verdict(source_1318)
                else "unknown"
            ),
        },
    }


def build_artifact(
    sources: Mapping[int, Mapping[str, Any]],
    *,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """SCENARIO-REPORT-028: synthesize .102 scoring from source artifacts."""

    criteria_results = _build_criteria_results(sources)
    criteria_met = sum(1 for status in criteria_results.values() if status == MET)
    summary = _summary_fields(sources, criteria_results)
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "metadata": {
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "run_date": run_date,
            "source_roadmap": (
                "research-roadmap.yaml; research-roadmap-next.yaml unavailable at runtime"
            ),
            "change_proposal": "openspec/change-proposals/research-roadmap-vNEXT.md",
        },
        "milestone": MILESTONE,
        "status": "complete",
        "criteria_results": criteria_results,
        "criteria_met": criteria_met,
        "criteria_total": len(CRITERIA),
        "source_artifacts_checked": _source_artifacts_checked(sources),
        "gated_missing_blocked_or_failed_tasks": _gated_or_missing_tasks(
            sources, criteria_results
        ),
        "carry_forward_tasks": _carry_forward_tasks(sources, criteria_results),
        "docs_reconciled": False,
        "docs_reconciliation_note": (
            "ops/status.md, ops/changelog.md, and _bmad/traceability.md were left untouched "
            "because the conductor stop rule delegates ops reconciliation to the following Haiku step."
        ),
        "retro_complete": True,
        "honest_verdict": f"milestone_102_{criteria_met}_of_14_criteria_met",
        **summary,
    }


def run(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """Load .102 source artifacts, write the Exp 1322 result JSON, and return it."""

    results_path = Path(results_dir)
    target = Path(out_path)
    write_in_progress_artifact(target)
    sources = {
        exp_id: loaded
        for exp_id, filename in SOURCE_FILES.items()
        if (loaded := _load_json(results_path / filename)) is not None
    }
    return _write_json(target, build_artifact(sources))
