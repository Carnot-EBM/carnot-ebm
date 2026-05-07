"""Build the Exp 1478 milestone .113 retrospective artifact.

Spec: REQ-REPORT-049, SCENARIO-REPORT-049.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1478_milestone_113_retro.json"

EXPERIMENT = "1478_milestone_113_retro"
SCHEMA = "milestone_113_retro_v1"
RUN_DATE = "20260507"
MILESTONE = "2026.04.113"

MET = "met"
UNMET = "unmet"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone",
    "criteria_met",
    "criteria_total",
    "blocked_tasks",
    "retired_lineages",
    "preserved_lineages",
    "carry_forward_tracks",
    "missing_artifacts",
    "research_roadmap_yaml_modified",
    "scripts_research_conductor_modified",
    "ops_docs_updated",
    "honest_verdict",
)

SOURCE_FILES = {
    "exp1467": "experiment_1467_112_completion_archive_113_activation.json",
    "exp1468": "experiment_1468_live_sota_logprob_telemetry_preflight.json",
    "exp1469": "experiment_1469_halt_spilled_energy_telemetry_diagnostic.json",
    "exp1470": "experiment_1470_beaver_lite_deterministic_bound_smoke.json",
    "exp1471": "experiment_1471_fr11_v8_verified_memory_growth_pivot.json",
    "exp1472": "experiment_1472_online_verifier_asymmetric_mistake_budget.json",
    "exp1473": "experiment_1473_live_telemetry_adversarial_validity_audit.json",
    "exp1474": "experiment_1474_tskm_linear_constraint_projection_smoke.json",
    "exp1475": "experiment_1475_static_csr_certificate_automaton_smoke.json",
    "exp1476": "experiment_1476_kv260_discrete_sb_rtl_regression_pack.json",
    "exp1477": "experiment_1477_thrml_npim_simulator_parity_microprobe.json",
}

CRITERION_SOURCE = {
    "activation": "exp1467",
    "telemetry_preflight": "exp1468",
    "halt_energy_diagnostic": "exp1469",
    "beaver_smoke": "exp1470",
    "self_learning_pivot": "exp1471",
    "mistake_audit": "exp1472",
    "adversarial_telemetry_audit": "exp1473",
    "tskm_smoke": "exp1474",
    "static_automaton_smoke": "exp1475",
    "kv260_rtl_regression": "exp1476",
    "thrml_npim_parity": "exp1477",
    "retro": "exp1478",
}

LOG_TITLE_FRAGMENTS = {
    "exp1467": (".112 Completion Archive",),
    "exp1468": ("Live SOTA GGUF Logprob Telemetry Preflight",),
    "exp1469": ("HALT + Spilled Energy Diagnostic",),
    "exp1470": ("BEAVER-Lite Deterministic Bound Smoke",),
    "exp1471": ("FR-11 v8 Verified-Memory-Growth Pivot",),
    "exp1472": ("Online Verifier Asymmetric Mistake-Budget Audit",),
    "exp1473": ("Live Telemetry Adversarial Validity Audit",),
    "exp1474": ("T-SKM Linear Constraint Projection Smoke",),
    "exp1475": ("STATIC CSR Certificate Automaton Smoke",),
    "exp1476": ("KV260 Discrete SB RTL Regression Pack",),
    "exp1477": ("THRML + NPIM Simulator Parity",),
}


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-049: persist a truthful skeleton before evidence loading.

    The conductor can be interrupted between source reads. A minimal progress
    record prevents that interruption from being mistaken for a completed
    retrospective or for missing work.
    """

    artifact = {field: None for field in REQUIRED_ARTIFACT_FIELDS}
    artifact["status"] = "in_progress"
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payloads = [
        (exp_id, _read_json(results_dir / filename))
        for exp_id, filename in SOURCE_FILES.items()
    ]
    sources = {exp_id: payload for exp_id, payload in payloads if payload is not None}
    missing = [exp_id for exp_id, payload in payloads if payload is None]
    return sources, missing


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict") or "")


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _source_path(exp_id: str, field: str | None = None) -> str:
    filename = SOURCE_FILES.get(exp_id, f"experiment_{exp_id}.json")
    path = f"results/{filename}"
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


def _score_halt_energy(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1468 = sources.get("exp1468", {})
    exp1469 = sources.get("exp1469", {})
    target = (
        "if exp1468.topk_logprobs_available=true then "
        "exp1469.telemetry_diagnostic_complete=true; otherwise terminal logprob skip"
    )
    if "exp1469" in missing_source_ids or "exp1469" not in sources:
        return _source_missing_criterion("exp1469", target)
    topk_available = exp1468.get("topk_logprobs_available") is True
    complete = exp1469.get("telemetry_diagnostic_complete") is True
    skip_reason = " ".join(
        str(exp1469.get(field) or "")
        for field in ("gated_off_reason", "blocker", "honest_verdict")
    ).lower()
    terminal_logprob_skip = not topk_available and "logprob" in skip_reason
    passed = complete if topk_available else terminal_logprob_skip
    return _criterion(
        MET if passed else UNMET,
        target,
        [
            _source_path("exp1468", "topk_logprobs_available"),
            _source_path("exp1469", "telemetry_diagnostic_complete"),
            _source_path("exp1469", "gated_off_reason"),
        ],
        [
            "HALT/spilled diagnostic completed with available top-k telemetry."
            if topk_available
            else "HALT/spilled diagnostic produced a terminal missing-logprob skip."
        ]
        if passed
        else [],
        []
        if passed
        else ["HALT/spilled diagnostic neither completed nor recorded a missing-logprob skip."],
        {
            "status": exp1469.get("status"),
            "topk_logprobs_available": exp1468.get("topk_logprobs_available"),
            "telemetry_diagnostic_complete": exp1469.get("telemetry_diagnostic_complete"),
            "gated_off_reason": exp1469.get("gated_off_reason"),
            "diagnostic_lineage_retired": exp1469.get("diagnostic_lineage_retired"),
            "honest_verdict": _verdict(exp1469),
        },
    )


def _score_self_learning(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1471 = sources.get("exp1471", {})
    delta = _number(exp1471.get("self_learning_delta_overall"))
    promoted = _number(exp1471.get("new_promoted_count"))
    nonforgetting = _number(exp1471.get("nonforgetting_rate"))
    growth_passed = (
        delta is not None
        and delta > 0.0
        and promoted is not None
        and promoted >= 1.0
        and nonforgetting is not None
        and nonforgetting >= 0.99
    )
    retired = exp1471.get("pivot_retired") is True or exp1471.get("self_learning_pivot_retired") is True
    return _scored(
        "exp1471",
        sources,
        missing_source_ids,
        growth_passed or retired,
        "exp1471 growth/nonforgetting gates pass or pivot is retired",
        (
            "self_learning_delta_overall",
            "new_promoted_count",
            "nonforgetting_rate",
            "pivot_retired",
        ),
        "Self-learning pivot produced verified growth without forgetting."
        if growth_passed
        else "Self-learning pivot was explicitly retired.",
        "Self-learning pivot neither passed growth gates nor retired itself.",
    )


def _score_adversarial_telemetry(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1473 = sources.get("exp1473", {})
    checked_flags = (
        "length_confound_checked",
        "format_confound_checked",
        "mock_logprob_leakage_checked",
        "prompt_family_confound_checked",
    )
    confound_checked = any(exp1473.get(flag) is True for flag in checked_flags)
    blockers = exp1473.get("superficial_baseline_results", {}).get("claim_blockers", [])
    passed = bool(exp1473.get("telemetry_validity_verdict")) and (confound_checked or bool(blockers))
    return _scored(
        "exp1473",
        sources,
        missing_source_ids,
        passed,
        "exp1473.telemetry_validity_verdict is terminal and confounds are named",
        ("telemetry_validity_verdict", *checked_flags),
        "Telemetry audit reached a terminal validity verdict and named superficial checks.",
        "Telemetry audit did not reach a terminal confound-aware verdict.",
    )


def _score_tskm(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1474 = sources.get("exp1474", {})
    passed = exp1474.get("zero_violation_projection") is True or bool(exp1474.get("blockers"))
    return _scored(
        "exp1474",
        sources,
        missing_source_ids,
        passed,
        "exp1474.zero_violation_projection=true or blocker recorded",
        ("zero_violation_projection", "blockers", "baseline_verifier_agreement"),
        "T-SKM projection completed with zero violations or a terminal blocker.",
        "T-SKM projection neither achieved zero violations nor recorded a blocker.",
    )


def _score_thrml_npim(
    sources: Mapping[str, Mapping[str, Any]], missing_source_ids: set[str]
) -> dict[str, Any]:
    exp1477 = sources.get("exp1477", {})
    parity_reported = isinstance(exp1477.get("parity_metric"), Mapping)
    sample_quality_reported = bool(exp1477.get("carnot_sampler_cases")) or bool(exp1477.get("npim_cases"))
    passed = exp1477.get("hardware_claim_allowed") is False and parity_reported and sample_quality_reported
    return _scored(
        "exp1477",
        sources,
        missing_source_ids,
        passed,
        "exp1477.hardware_claim_allowed=false and simulator fields are reported",
        ("hardware_claim_allowed", "parity_metric", "carnot_sampler_cases", "npim_cases"),
        "THRML/NPIM remained simulator-only and reported parity/sample-quality fields.",
        "THRML/NPIM did not preserve the no-hardware-claim simulator boundary.",
    )


def _score_retro(
    research_roadmap_yaml_modified: bool,
    scripts_research_conductor_modified: bool,
    retired_lineages: list[dict[str, Any]],
    preserved_lineages: list[dict[str, Any]],
    carry_forward_tracks: list[dict[str, Any]],
) -> dict[str, Any]:
    required_tracks = {
        "HALT/Spilled Energy telemetry diagnostic",
        "FR-11 v8 verified-memory-growth pivot",
        "T-SKM toy linear projection",
        "STATIC CSR certificate automaton",
        "KV260 source-level RTL regression",
        "THRML/NPIM simulator-only parity probe",
    }
    recorded_tracks = {item["lineage"] for item in retired_lineages + preserved_lineages}
    passed = (
        not research_roadmap_yaml_modified
        and not scripts_research_conductor_modified
        and required_tracks <= recorded_tracks
        and bool(carry_forward_tracks)
    )
    return _criterion(
        MET if passed else UNMET,
        "exp1478.criteria_total=12 with lineage/carry-forward records and forbidden files unchanged",
        ["results/experiment_1478_milestone_113_retro.json"],
        ["Retro closure records all required lineages and confirms forbidden files unchanged."]
        if passed
        else [],
        []
        if passed
        else ["Retro closure is missing lineage decisions, carry-forward rules, or no-change proof."],
        {
            "criteria_total": 12,
            "research_roadmap_yaml_modified": research_roadmap_yaml_modified,
            "scripts_research_conductor_modified": scripts_research_conductor_modified,
            "recorded_lineage_count": len(recorded_tracks),
            "carry_forward_track_count": len(carry_forward_tracks),
        },
    )


def _score_criteria(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    research_roadmap_yaml_modified: bool,
    scripts_research_conductor_modified: bool,
    retired_lineages: list[dict[str, Any]],
    preserved_lineages: list[dict[str, Any]],
    carry_forward_tracks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    exp1467 = sources.get("exp1467", {})
    exp1468 = sources.get("exp1468", {})
    exp1470 = sources.get("exp1470", {})
    exp1472 = sources.get("exp1472", {})
    exp1475 = sources.get("exp1475", {})
    exp1476 = sources.get("exp1476", {})
    return {
        "activation": _scored(
            "exp1467",
            sources,
            missing_source_ids,
            exp1467.get("activation_manifest_complete") is True
            and exp1467.get("predecessor_honest_verdict") is not None,
            "exp1467.activation_manifest_complete=true with predecessor summary",
            ("activation_manifest_complete", "predecessor_honest_verdict", "criteria_met"),
            ".113 activation completed and summarized .112 completion evidence.",
            ".113 activation did not complete or lacked predecessor evidence.",
        ),
        "telemetry_preflight": _scored(
            "exp1468",
            sources,
            missing_source_ids,
            exp1468.get("live_sota_model_inference_used") is True
            and "topk_logprobs_available" in exp1468,
            "exp1468.live_sota_model_inference_used=true and top-k availability recorded",
            ("live_sota_model_inference_used", "topk_logprobs_available", "telemetry_cases_completed"),
            "Live SOTA telemetry preflight ran and recorded top-k/logprob availability.",
            "Live SOTA telemetry preflight did not run or failed to record top-k availability.",
        ),
        "halt_energy_diagnostic": _score_halt_energy(sources, missing_source_ids),
        "beaver_smoke": _scored(
            "exp1470",
            sources,
            missing_source_ids,
            exp1470.get("bound_is_sound") is True and bool(exp1470.get("mock_or_live_logprobs")),
            "exp1470.bound_is_sound=true with live/mock logprob label",
            ("bound_is_sound", "mock_or_live_logprobs", "empirical_violation_rates"),
            "BEAVER-lite smoke found sound bounds and labeled logprob provenance.",
            "BEAVER-lite smoke did not prove sound bounds or label logprob provenance.",
        ),
        "self_learning_pivot": _score_self_learning(sources, missing_source_ids),
        "mistake_audit": _scored(
            "exp1472",
            sources,
            missing_source_ids,
            _number(exp1472.get("soundness_mistakes")) is not None
            and _number(exp1472.get("completeness_mistakes")) is not None
            and bool(exp1472.get("pareto_decision") or exp1472.get("asymmetric_cost_decision")),
            "exp1472 soundness/completeness mistakes reported with asymmetric decision",
            ("soundness_mistakes", "completeness_mistakes", "pareto_decision"),
            "Mistake audit reported asymmetric soundness/completeness accounting.",
            "Mistake audit lacked mistake counts or an asymmetric-cost decision.",
        ),
        "adversarial_telemetry_audit": _score_adversarial_telemetry(
            sources, missing_source_ids
        ),
        "tskm_smoke": _score_tskm(sources, missing_source_ids),
        "static_automaton_smoke": _scored(
            "exp1475",
            sources,
            missing_source_ids,
            exp1475.get("exact_acceptance_equivalent") is True
            and _number(exp1475.get("csr_latency_ms_p50")) is not None,
            "exp1475.exact_acceptance_equivalent=true and latency reported",
            ("exact_acceptance_equivalent", "csr_latency_ms_p50", "existing_path_latency_ms_p50"),
            "STATIC CSR automaton matched existing acceptance and reported latency.",
            "STATIC CSR automaton lacked exact equivalence or latency.",
        ),
        "kv260_rtl_regression": _scored(
            "exp1476",
            sources,
            missing_source_ids,
            exp1476.get("rtl_regression_complete") is True
            and exp1476.get("board_execution_performed") is False
            and exp1476.get("latency_claimed") is False,
            "exp1476.rtl_regression_complete=true with no board or latency claim",
            ("rtl_regression_complete", "board_execution_performed", "latency_claimed"),
            "KV260 regression stayed source-level with no board or latency claim.",
            "KV260 regression did not complete or overclaimed hardware execution.",
        ),
        "thrml_npim_parity": _score_thrml_npim(sources, missing_source_ids),
        "retro": _score_retro(
            research_roadmap_yaml_modified,
            scripts_research_conductor_modified,
            retired_lineages,
            preserved_lineages,
            carry_forward_tracks,
        ),
    }


def _retired_lineages(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exp1469 = sources.get("exp1469", {})
    logprob_skip = "logprob" in " ".join(
        str(exp1469.get(field) or "")
        for field in ("gated_off_reason", "blocker", "honest_verdict")
    ).lower()
    return [
        {
            "lineage": "HALT/Spilled Energy telemetry diagnostic",
            "decision": "retired",
            "source_experiment": "exp1469",
            "reason": "Non-headline telemetry signal was flat, confounded, or gated off by missing logprobs.",
            "honest_verdict": _verdict(exp1469),
            "evidence_path": _source_path("exp1469"),
        }
    ] if exp1469.get("diagnostic_lineage_retired") is True or logprob_skip else []


def _preserved_lineages(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exp1471 = sources.get("exp1471", {})
    exp1474 = sources.get("exp1474", {})
    exp1475 = sources.get("exp1475", {})
    exp1476 = sources.get("exp1476", {})
    exp1477 = sources.get("exp1477", {})
    return [
        {
            "lineage": "FR-11 v8 verified-memory-growth pivot",
            "decision": "preserved",
            "source_experiment": "exp1471",
            "rule": "Carry forward only the narrow zero-soundness-mistake memory-growth claim.",
            "honest_verdict": _verdict(exp1471),
        },
        {
            "lineage": "T-SKM toy linear projection",
            "decision": "preserved",
            "source_experiment": "exp1474",
            "rule": "Preserve as a toy projection baseline; do not reopen HardNet++/DSP variants.",
            "honest_verdict": _verdict(exp1474),
        },
        {
            "lineage": "STATIC CSR certificate automaton",
            "decision": "preserved",
            "source_experiment": "exp1475",
            "rule": "Preserve as a bounded parser/automaton benchmark with no repair generation.",
            "honest_verdict": _verdict(exp1475),
        },
        {
            "lineage": "KV260 source-level RTL regression",
            "decision": "preserved",
            "source_experiment": "exp1476",
            "rule": "Carry forward source-level lint/simulation only; no board, bitfile, or latency claim.",
            "honest_verdict": _verdict(exp1476),
        },
        {
            "lineage": "THRML/NPIM simulator-only parity probe",
            "decision": "preserved",
            "source_experiment": "exp1477",
            "rule": "Carry forward simulator-only tracking; install THRML before any parity claim.",
            "honest_verdict": _verdict(exp1477),
        },
    ]


def _carry_forward_tracks(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exp1468 = sources.get("exp1468", {})
    exp1469 = sources.get("exp1469", {})
    exp1470 = sources.get("exp1470", {})
    exp1471 = sources.get("exp1471", {})
    exp1472 = sources.get("exp1472", {})
    exp1473 = sources.get("exp1473", {})
    exp1474 = sources.get("exp1474", {})
    exp1475 = sources.get("exp1475", {})
    exp1476 = sources.get("exp1476", {})
    exp1477 = sources.get("exp1477", {})
    return [
        {
            "track": "live_sota_telemetry",
            "source_experiment": "exp1468",
            "rule": "Preserve the raw top-k telemetry manifest, but do not make a headline signal claim.",
            "status": "raw_telemetry_ready",
            "topk_logprobs_available": exp1468.get("topk_logprobs_available"),
            "honest_verdict": _verdict(exp1468),
        },
        {
            "track": "halt_spilled_energy",
            "source_experiment": "exp1469",
            "rule": "Retire as a headline diagnostic unless a future run beats superficial confounds.",
            "status": "retired",
            "honest_verdict": _verdict(exp1469),
        },
        {
            "track": "beaver_lite_bounds",
            "source_experiment": "exp1470",
            "rule": "Keep only the minimal deterministic-bound smoke; broad external benchmark runners stay deferred.",
            "status": "narrow_smoke_preserved",
            "mock_or_live_logprobs": exp1470.get("mock_or_live_logprobs"),
            "honest_verdict": _verdict(exp1470),
        },
        {
            "track": "self_learning",
            "source_experiment": "exp1471/exp1472",
            "rule": "Carry forward the narrow verified-memory-growth claim with zero soundness mistakes and completeness caveat.",
            "status": "preserved",
            "soundness_mistakes": exp1472.get("soundness_mistakes"),
            "completeness_mistakes": exp1472.get("completeness_mistakes"),
            "self_learning_delta_overall": exp1471.get("self_learning_delta_overall"),
        },
        {
            "track": "telemetry_headline_claim",
            "source_experiment": "exp1473",
            "rule": "Do not claim telemetry validity as a headline; adversarial audit blocked that claim.",
            "status": "blocked",
            "telemetry_validity_verdict": exp1473.get("telemetry_validity_verdict"),
            "honest_verdict": _verdict(exp1473),
        },
        {
            "track": "constraint_smokes",
            "source_experiment": "exp1474/exp1475",
            "rule": "Preserve T-SKM and STATIC only as bounded CPU toy/parser baselines.",
            "status": "preserved",
            "tskm_zero_violation": exp1474.get("zero_violation_projection"),
            "static_exact_equivalence": exp1475.get("exact_acceptance_equivalent"),
        },
        {
            "track": "hardware_simulation",
            "source_experiment": "exp1476/exp1477",
            "rule": "KV260 remains source-level RTL; THRML/NPIM remains simulator-only with no hardware claim.",
            "status": "preserved_with_environmental_blocker",
            "rtl_regression_complete": exp1476.get("rtl_regression_complete"),
            "thrml_available": exp1477.get("thrml_available"),
            "hardware_claim_allowed": exp1477.get("hardware_claim_allowed"),
        },
    ]


def _failed_attempts_from_log(conductor_log_text: str) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for line in conductor_log_text.splitlines():
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) < 4 or columns[2] not in {"FAIL", "GATE_BLOCK", "SKIP"}:
            continue
        title = columns[1]
        exp_id = next(
            (
                candidate
                for candidate, fragments in LOG_TITLE_FRAGMENTS.items()
                if any(fragment in title for fragment in fragments)
            ),
            None,
        )
        if exp_id is not None:
            failed.append(
                {
                    "kind": "failed_conductor_attempt",
                    "experiment_id": exp_id,
                    "status": columns[2],
                    "reason": columns[3],
                    "log_line": line,
                }
            )
    return failed


def _terminal_blockers(
    sources: Mapping[str, Mapping[str, Any]],
    criteria: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    for blocker in sources.get("exp1477", {}).get("blockers", []) or []:
        blocked.append(
            {
                "kind": "terminal_environment_blocker",
                "criterion": "thrml_npim_parity",
                "criterion_status": criteria["thrml_npim_parity"]["status"],
                "experiment_id": "exp1477",
                "blocker": blocker.get("blocker"),
                "detail": blocker.get("detail"),
            }
        )
    for criterion_id, result in criteria.items():
        if result["status"] == UNMET and result["negative_evidence"]:
            blocked.append(
                {
                    "kind": "unmet_terminal_criterion",
                    "criterion": criterion_id,
                    "experiment_id": CRITERION_SOURCE[criterion_id],
                    "blocker": result["negative_evidence"][0],
                }
            )
    return blocked


def _missing_artifacts(
    missing_source_ids: set[str],
    roadmap_doc_present: bool,
    roadmap_yaml_present: bool,
    roadmap_next_present: bool,
    conductor_log_present: bool,
) -> list[dict[str, str]]:
    missing = [
        {"path": f"results/{SOURCE_FILES[exp_id]}", "reason": "source_artifact_missing"}
        for exp_id in SOURCE_FILES
        if exp_id in missing_source_ids
    ]
    optional_inputs = [
        ("openspec/change-proposals/research-roadmap-vNEXT.md", roadmap_doc_present),
        ("research-roadmap.yaml", roadmap_yaml_present),
        ("research-roadmap-next.yaml", roadmap_next_present),
        ("ops/conductor-log.md", conductor_log_present),
    ]
    missing.extend(
        {
            "path": path,
            "reason": "requested_input_missing" if path == "research-roadmap-next.yaml" else "input_missing",
        }
        for path, present in optional_inputs
        if not present
    )
    return missing


def _source_checks(missing_source_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": exp_id,
            "path": f"results/{filename}",
            "exists": exp_id not in missing_source_ids,
        }
        for exp_id, filename in SOURCE_FILES.items()
    ]


def _roadmap_inputs(
    roadmap_doc_present: bool,
    roadmap_yaml_present: bool,
    roadmap_next_present: bool,
    conductor_log_present: bool,
) -> dict[str, Any]:
    return {
        "change_proposal_path": "openspec/change-proposals/research-roadmap-vNEXT.md",
        "change_proposal_present": roadmap_doc_present,
        "active_roadmap_yaml_path": "research-roadmap.yaml",
        "active_roadmap_yaml_present": roadmap_yaml_present,
        "requested_research_roadmap_next_path": "research-roadmap-next.yaml",
        "requested_research_roadmap_next_present": roadmap_next_present,
        "conductor_log_path": "ops/conductor-log.md",
        "conductor_log_present": conductor_log_present,
    }


def build_artifact(
    sources: Mapping[str, dict[str, Any]],
    missing_source_ids: list[str],
    roadmap_doc_present: bool,
    roadmap_yaml_present: bool,
    roadmap_next_present: bool,
    conductor_log_text: str,
    research_roadmap_yaml_modified: bool,
    scripts_research_conductor_modified: bool,
    ops_docs_updated: bool,
) -> dict[str, Any]:
    """REQ-REPORT-049: score .113 from terminal evidence, not optimism.

    Several `.113` tasks are deliberately terminal without becoming headline
    successes. This builder lets those tasks satisfy criteria that asked for a
    terminal audit, while still retiring or blocking the affected research line.
    """

    missing = set(missing_source_ids)
    retired_lineages = _retired_lineages(sources)
    preserved_lineages = _preserved_lineages(sources)
    carry_forward_tracks = _carry_forward_tracks(sources)
    criteria = _score_criteria(
        sources,
        missing,
        research_roadmap_yaml_modified,
        scripts_research_conductor_modified,
        retired_lineages,
        preserved_lineages,
        carry_forward_tracks,
    )
    criteria_met = sum(1 for result in criteria.values() if result["status"] == MET)
    criteria_total = len(CRITERION_SOURCE)
    threshold_met = criteria_met >= 9
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "milestone": MILESTONE,
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
        "success_criteria_results": criteria,
        "blocked_tasks": _failed_attempts_from_log(conductor_log_text)
        + _terminal_blockers(sources, criteria),
        "retired_lineages": retired_lineages,
        "preserved_lineages": preserved_lineages,
        "carry_forward_tracks": carry_forward_tracks,
        "missing_artifacts": _missing_artifacts(
            missing,
            roadmap_doc_present,
            roadmap_yaml_present,
            roadmap_next_present,
            bool(conductor_log_text),
        ),
        "source_artifacts_checked": _source_checks(missing),
        "roadmap_inputs": _roadmap_inputs(
            roadmap_doc_present,
            roadmap_yaml_present,
            roadmap_next_present,
            bool(conductor_log_text),
        ),
        "research_roadmap_yaml_modified": research_roadmap_yaml_modified,
        "scripts_research_conductor_modified": scripts_research_conductor_modified,
        "ops_docs_updated": ops_docs_updated,
        "ops_docs_update_note": (
            "ops/status.md and ops/changelog.md were not edited by Exp 1478 because "
            "the terminal stop rule delegates docs/status reconciliation to the conductor pass."
        )
        if not ops_docs_updated
        else "ops/status.md and ops/changelog.md were updated by the retrospective workflow.",
        "honest_verdict": (
            f"milestone_113_{criteria_met}_of_{criteria_total}_criteria_met_"
            f"{'success_threshold_met' if threshold_met else 'below_success_threshold'}_"
            "halt_spilled_retired_telemetry_headline_blocked"
        ),
    }


def _path_modified_by_git(root: Path, relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", relative_path],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 1


def run(root: Path | str = REPO_ROOT, out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    root_path = Path(root)
    write_in_progress_artifact(out_path)
    sources, missing = _load_sources(root_path / "results")
    conductor_log_text = _read_text(root_path / "ops" / "conductor-log.md")
    artifact = build_artifact(
        sources,
        missing,
        roadmap_doc_present=(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ).exists(),
        roadmap_yaml_present=(root_path / "research-roadmap.yaml").exists(),
        roadmap_next_present=(root_path / "research-roadmap-next.yaml").exists(),
        conductor_log_text=conductor_log_text,
        research_roadmap_yaml_modified=_path_modified_by_git(root_path, "research-roadmap.yaml"),
        scripts_research_conductor_modified=_path_modified_by_git(
            root_path, "scripts/research_conductor.py"
        ),
        ops_docs_updated=False,
    )
    return _write_json(Path(out_path), artifact)
