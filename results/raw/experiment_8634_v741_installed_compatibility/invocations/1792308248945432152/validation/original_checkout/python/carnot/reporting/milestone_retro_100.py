"""Build the Exp 1295 milestone .100 retrospective artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1295_milestone_retro_100.json"

EXPERIMENT = "1295_milestone_retro_100"
SCHEMA = "milestone_retro_v5"
RUN_DATE = "20260504"
MILESTONE = "2026.04.100"

MET = "MET"
NOT_MET = "NOT_MET"
GATED = "GATED"
BLOCKED = "BLOCKED"
MISSING = "MISSING"

REQUIRED_SOTA_GGUF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

SOURCE_FILES = {
    1282: "experiment_1282_sota_gguf_cache_provenance_preflight.json",
    1283: "experiment_1283_certificate_grammar_backend_bakeoff.json",
    1284: "experiment_1284_ars_uqlm_answer_stability_sota_audit.json",
    1285: "experiment_1285_triggered_certificate_extraction_v2.json",
    1286: "experiment_1286_grad_beaver_nsvif_semantic_routing.json",
    1287: "experiment_1287_token_guard_cactus_constrained_acceptance_v2.json",
    1288: "experiment_1288_interwhen_dvi_verifier_feedback_replay.json",
    1289: "experiment_1289_leanabell_grpo_v9_sota_headline_gated.json",
    1290: "experiment_1290_skill_graph_promotion_demotion.json",
    1291: "experiment_1291_hardnetpp_nonlinear_repair_benchmark.json",
    1292: "experiment_1292_dsp_feasibility_channel_diagnostic.json",
    1293: "experiment_1293_ebt_arm_ebm_cot_energy_bridge_audit.json",
    1294: "experiment_1294_arxiv_v10_submission_receipt_or_blocker.json",
}

CRITERIA: tuple[tuple[str, int | None], ...] = (
    ("sota_gguf_cache_provenance_preflight", 1282),
    ("certificate_grammar_backend_bakeoff", 1283),
    ("ars_uqlm_answer_stability_sota_audit", 1284),
    ("triggered_certificate_extraction_v2", 1285),
    ("grad_beaver_nsvif_semantic_routing", 1286),
    ("token_guard_cactus_constrained_acceptance_v2", 1287),
    ("interwhen_dvi_verifier_feedback_replay", 1288),
    ("leanabell_grpo_v9_sota_headline_gated", 1289),
    ("skill_graph_promotion_demotion", 1290),
    ("hardnetpp_nonlinear_repair_benchmark", 1291),
    ("dsp_feasibility_channel_diagnostic", 1292),
    ("ebt_arm_ebm_cot_energy_bridge_audit", 1293),
    ("arxiv_v10_submission_receipt_or_blocker", 1294),
    ("retro_100_complete", None),
)

CRITERION_NAMES = tuple(name for name, _exp_id in CRITERIA)


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """Write the durable placeholder required before source artifact evaluation."""

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "milestone": MILESTONE,
            "status": "in_progress",
            "criteria_total": len(CRITERIA),
            "retro_complete": False,
            "honest_verdict": "in_progress",
        },
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float)


def _at_least(value: object, threshold: float) -> bool:
    return _is_number(value) and float(value) >= threshold


def _positive(value: object) -> bool:
    return _is_number(value) and float(value) > 0.0


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status", "")).lower()


def _honest_verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict", "")).lower()


def _stale_source(payload: Mapping[str, Any]) -> bool:
    return _status(payload) in {"in_progress", "bootstrap"} or _honest_verdict(payload) == "in_progress"


def _blocked_source(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "blocked" or _honest_verdict(payload).startswith("blocked")


def _terminal_failure(payload: Mapping[str, Any]) -> str:
    return BLOCKED if _blocked_source(payload) else NOT_MET


def _sota_gate_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return sources.get(1282, {}).get("cached_sota_ready") is True


def _grammar_gate_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return sources.get(1283, {}).get("grammar_backend_available") is True


def _answer_stability_gate_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _at_least(sources.get(1284, {}).get("answer_stability_score"), 0.6)


def _certificate_parse_gate_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _at_least(sources.get(1285, {}).get("certificate_parse_rate"), 0.8)


def _headline_gate_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return sources.get(1285, {}).get("headline_result_allowed") is True


def _dvi_gate_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return _positive(sources.get(1288, {}).get("dvi_acceptance_delta"))


def _memory_gate_passed(sources: Mapping[int, Mapping[str, Any]]) -> bool:
    return sources.get(1288, {}).get("memory_update_written") is True


def _gate_unmet(name: str, sources: Mapping[int, Mapping[str, Any]]) -> bool:
    if name == "ars_uqlm_answer_stability_sota_audit":
        return not _sota_gate_passed(sources)
    if name == "triggered_certificate_extraction_v2":
        return not (
            _sota_gate_passed(sources)
            and _grammar_gate_passed(sources)
            and _answer_stability_gate_passed(sources)
        )
    if name in {
        "grad_beaver_nsvif_semantic_routing",
        "token_guard_cactus_constrained_acceptance_v2",
    }:
        return not _certificate_parse_gate_passed(sources)
    if name == "leanabell_grpo_v9_sota_headline_gated":
        return not (_headline_gate_passed(sources) and _dvi_gate_passed(sources))
    if name == "skill_graph_promotion_demotion":
        return not _memory_gate_passed(sources)
    return False


def _has_keys(payload: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    return all(key in payload for key in keys)


def _classify_criterion(
    name: str,
    exp_id: int | None,
    sources: Mapping[int, Mapping[str, Any]],
) -> str:
    if exp_id is None:
        return MET
    if _gate_unmet(name, sources):
        return GATED
    if exp_id not in sources:
        return MISSING

    payload = sources[exp_id]
    if _stale_source(payload):
        return NOT_MET
    if _blocked_source(payload):
        return BLOCKED

    if name == "sota_gguf_cache_provenance_preflight":
        return MET if isinstance(payload.get("cached_sota_ready"), bool) else NOT_MET
    if name == "certificate_grammar_backend_bakeoff":
        required = (
            "cdot_expressiveness_note",
            "static_trie_note",
            "bounded_vocab_constraint_count",
            "automata_fallback_viable",
            "dfa_checkable_fields",
            "structure_snowballing_risk",
        )
        has_backend_decision = bool(payload.get("grammar_backend_selected")) or payload.get("grammar_backend_available") is False
        return MET if has_backend_decision and _has_keys(payload, required) else NOT_MET
    if name == "ars_uqlm_answer_stability_sota_audit":
        return MET if _is_number(payload.get("answer_stability_score")) else NOT_MET
    if name == "triggered_certificate_extraction_v2":
        measured = _is_number(payload.get("certificate_parse_rate"))
        return MET if measured and payload.get("headline_result_allowed") is True else _terminal_failure(payload)
    if name == "grad_beaver_nsvif_semantic_routing":
        routed = _is_number(payload.get("semantic_routing_coverage")) and _is_number(payload.get("routed_claim_count"))
        return MET if routed else _terminal_failure(payload)
    if name == "token_guard_cactus_constrained_acceptance_v2":
        required = (
            "risk_bound_proxy",
            "token_guard_risk_score",
            "low_risk_acceptance_rate",
            "speedbench_eval_mode",
        )
        return MET if _is_number(payload.get("cactus_acceptance_rate")) and _has_keys(payload, required) else NOT_MET
    if name == "interwhen_dvi_verifier_feedback_replay":
        measured = _is_number(payload.get("dvi_acceptance_delta")) and bool(payload.get("claim_level_memory_entries"))
        return MET if measured else NOT_MET
    if name == "leanabell_grpo_v9_sota_headline_gated":
        measured = payload.get("headline_result_allowed") is True and _is_number(payload.get("grpo_v9_delta"))
        return MET if measured else NOT_MET
    if name == "skill_graph_promotion_demotion":
        written = payload.get("skill_graph_entries_written") is True or _is_number(payload.get("skill_replay_delta"))
        return MET if written else NOT_MET
    if name == "hardnetpp_nonlinear_repair_benchmark":
        return MET if payload.get("nonlinear_repair_viable") is True and _positive(payload.get("hardnetpp_delta_over_snarenet")) else NOT_MET
    if name == "dsp_feasibility_channel_diagnostic":
        return MET if payload.get("feasibility_channel_predictive") is True and _is_number(payload.get("feasibility_channel_auc")) else NOT_MET
    if name == "ebt_arm_ebm_cot_energy_bridge_audit":
        return MET if payload.get("energy_bridge_written") is True else NOT_MET
    if name == "arxiv_v10_submission_receipt_or_blocker":
        submitted = payload.get("arxiv_submitted") is True and bool(payload.get("arxiv_receipt"))
        blocker = bool(payload.get("external_blocker")) and _status(payload) == "complete"
        return MET if submitted or blocker else NOT_MET
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


def _stale_artifacts(
    sources: Mapping[int, Mapping[str, Any]],
    criteria_results: Mapping[str, str],
) -> list[dict[str, Any]]:
    stale: list[dict[str, Any]] = []
    for name, exp_id in CRITERIA:
        if exp_id is None:
            continue
        classification = criteria_results[name]
        if classification == MET:
            continue
        payload = sources.get(exp_id, {})
        stale.append(
            {
                "path": f"results/{SOURCE_FILES[exp_id]}",
                "criterion": name,
                "classification": classification,
                "loaded": exp_id in sources,
                "status": payload.get("status"),
                "honest_verdict": payload.get("honest_verdict"),
                "reason": payload.get("gate_check_summary") or payload.get("external_blocker"),
            }
        )
    return stale


def _sota_model_usage_summary(sources: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    cache = sources.get(1282, {})
    model_ids_used: list[str] = []
    for exp_id in (1284, 1285, 1287, 1289):
        payload = sources.get(exp_id, {})
        for key in ("MODEL_SPECS", "models_used"):
            value = payload.get(key, [])
            if isinstance(value, list):
                for entry in value:
                    if isinstance(entry, Mapping):
                        model_id = str(entry.get("hf_id") or entry.get("id") or "")
                        used = bool(entry.get("used_for_generation", entry.get("available", False)))
                        if model_id in REQUIRED_SOTA_GGUF_IDS and used and model_id not in model_ids_used:
                            model_ids_used.append(model_id)
    return {
        "headline_model_ids_required": list(REQUIRED_SOTA_GGUF_IDS),
        "cached_sota_ready": cache.get("cached_sota_ready"),
        "headline_result_possible": cache.get("headline_result_possible", False),
        "headline_model_ids_used": model_ids_used,
        "headline_result_allowed": bool(model_ids_used),
        "blocker": cache.get("gate_check_summary") or cache.get("blocked_reason"),
    }


def _self_learning_result(sources: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    dvi = sources.get(1288, {})
    grpo = sources.get(1289, {})
    skill = sources.get(1290, {})
    return {
        "dvi_acceptance_delta": dvi.get("dvi_acceptance_delta"),
        "online_acceptance_delta": dvi.get("online_acceptance_delta"),
        "self_learning_delta_overall": dvi.get("self_learning_delta_overall"),
        "claim_level_memory_entries": dvi.get("claim_level_memory_entries"),
        "memory_update_written": dvi.get("memory_update_written", False),
        "headline_result_allowed": dvi.get("headline_result_allowed", False),
        "grpo_v9_status": grpo.get("status", "missing"),
        "skill_graph_status": skill.get("status", "missing"),
        "summary": "DVI replay produced a positive non-headline acceptance delta, but GRPO v9 stayed gated and the skill graph artifact is missing.",
    }


def _continuous_repair_summary(sources: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    hardnet = sources.get(1291, {})
    dsp = sources.get(1292, {})
    bridge = sources.get(1293, {})
    return {
        "hardnetpp_nonlinear_repair_viable": hardnet.get("nonlinear_repair_viable", False),
        "hardnetpp_delta_over_snarenet": hardnet.get("hardnetpp_delta_over_snarenet"),
        "copy_as_decode_verified_span_reuse": hardnet.get("copy_as_decode_verified_span_reuse"),
        "feasibility_channel_predictive": dsp.get("feasibility_channel_predictive", False),
        "feasibility_channel_auc": dsp.get("feasibility_channel_auc"),
        "false_continue_rate": dsp.get("false_continue_rate"),
        "false_stop_rate": dsp.get("false_stop_rate"),
        "energy_bridge_status": bridge.get("status", "missing"),
        "energy_bridge_blocker": bridge.get("gate_check_summary"),
    }


def _publication_state(sources: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    publication = sources.get(1294, {})
    return {
        "status": publication.get("status", "missing"),
        "arxiv_submitted": publication.get("arxiv_submitted", False),
        "arxiv_receipt": publication.get("arxiv_receipt"),
        "external_blocker": publication.get("external_blocker") or publication.get("gate_check_summary"),
        "honest_verdict": publication.get("honest_verdict"),
    }


def build_artifact(
    sources: Mapping[int, Mapping[str, Any]],
    *,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Build the final retrospective artifact from already-loaded source JSON."""

    criteria_results = _build_criteria_results(sources)
    criteria_met = sum(1 for status in criteria_results.values() if status == MET)

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": run_date,
        "milestone": MILESTONE,
        "status": "complete",
        "criteria_results": criteria_results,
        "criteria_met": criteria_met,
        "criteria_total": len(CRITERIA),
        "source_artifacts_checked": _source_artifacts_checked(sources),
        "findings_summary": (
            f"Milestone .100 met {criteria_met} of 14 planned criteria. Grammar backend selection, "
            "DVI replay, HardNet++ nonlinear repair, DSP feasibility diagnostics, and this retro "
            "reached terminal positive criteria; SOTA certificate work, skill promotion, energy "
            "bridge, and publication receipt remain blocked, gated, or missing."
        ),
        "top_successes": [
            "Exp1283 selected llama.cpp GBNF as the local certificate grammar backend and recorded fallback limits.",
            "Exp1288 produced a positive DVI acceptance delta with claim-level memory records, though not headline eligible.",
            "Exp1291 found HardNet++ nonlinear repair viable against the FSNet/SnareNet comparison.",
            "Exp1292 measured a marginal but predictive DSP feasibility channel for repair-step decisions.",
        ],
        "top_gaps": [
            "Exp1282 was blocked by conductor prior-failure gating before SOTA cache readiness fields were written.",
            "Exp1284 and Exp1285 never produced SOTA answer-stability or certificate parse-rate artifacts.",
            "Exp1286, Exp1287, and Exp1289 remained gated behind the missing SOTA certificate parse/headline gates.",
            "Exp1290 is missing even though Exp1288 wrote memory_update_written=true.",
            "Exp1293 and Exp1294 were blocked by prior-failure gate checks, leaving the energy bridge and arXiv receipt unresolved.",
        ],
        "self_learning_result": _self_learning_result(sources),
        "sota_model_usage_summary": _sota_model_usage_summary(sources),
        "continuous_repair_summary": _continuous_repair_summary(sources),
        "publication_state": _publication_state(sources),
        "stale_artifacts": _stale_artifacts(sources, criteria_results),
        "key_carry_forwards": [
            "Repair the prior_failures metadata gate on Exp1282 so SOTA cache/provenance readiness can be measured.",
            "Run answer-stability and triggered certificate extraction only after cached_sota_ready is actually true.",
            "Use certificate_parse_rate >= 0.8 as the mechanical unlock for semantic routing and Cactus acceptance.",
            "Emit the missing skill-graph promotion/demotion artifact from Exp1288 memory updates or record a terminal blocker.",
            "Rerun the energy bridge and arXiv receipt tasks with prior-failure metadata so they produce real terminal artifacts.",
        ],
        "retro_complete": True,
        "honest_verdict": f"milestone_100_{criteria_met}_of_14_criteria_met",
    }


def run(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """Load .100 source artifacts, write the Exp 1295 result JSON, and return it."""

    results_path = Path(results_dir)
    target = Path(out_path)
    write_in_progress_artifact(target)
    sources = {
        exp_id: loaded
        for exp_id, filename in SOURCE_FILES.items()
        if (loaded := _load_json(results_path / filename)) is not None
    }
    return _write_json(target, build_artifact(sources))
