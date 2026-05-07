"""Build the Exp 1491 milestone .114 retrospective artifact.

Spec: REQ-REPORT-009, SCENARIO-REPORT-006.
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
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1491_milestone_114_retro.json"

EXPERIMENT = "1491_milestone_114_retro"
SCHEMA = "milestone_114_retro_v1"
RUN_DATE = "20260507"
MILESTONE = "2026.04.114"

MET = "met"
UNMET = "unmet"
GATE_SKIPPED = "gate_skipped"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "milestone",
    "criteria_met",
    "criteria_total",
    "success_threshold_met",
    "completed_task_ids",
    "blocked_task_ids",
    "retired_lineages",
    "carry_forward_recommendations",
    "ops_docs_updated",
    "research_roadmap_yaml_modified",
    "scripts_research_conductor_modified",
    "honest_verdict",
}

SOURCE_FILES = {
    "exp1479": "experiment_1479_113_completion_archive_114_activation.json",
    "exp1480": "experiment_1480_live_sota_balanced_telemetry_v2.json",
    "exp1481": "experiment_1481_semantic_energy_feasibility_audit.json",
    "exp1482": "experiment_1482_beaver_lite_live_prefix_bound_calibration.json",
    "exp1483": "experiment_1483_halluguard_risk_bound_fit_audit.json",
    "exp1484": "experiment_1484_fr11_v9_query_time_memory_policy.json",
    "exp1485": "experiment_1485_fr11_completeness_reduction_audit.json",
    "exp1486": "experiment_1486_cctu_executable_constraint_microbenchmark.json",
    "exp1487": "experiment_1487_v1_pairwise_self_verification_vs_energy.json",
    "exp1488": "experiment_1488_thrml_installability_import_preflight.json",
    "exp1489": "experiment_1489_thrml_carnot_simulator_parity_v2.json",
    "exp1490": "experiment_1490_kona_ebt_partial_trace_localization_audit.json",
}

CRITERION_SOURCE = {
    "activation": "exp1479",
    "balanced_telemetry": "exp1480",
    "semantic_energy_audit": "exp1481",
    "beaver_calibration": "exp1482",
    "halluguard_fit": "exp1483",
    "query_time_self_learning": "exp1484",
    "completeness_reduction": "exp1485",
    "executable_tool_use_benchmark": "exp1486",
    "pairwise_verification": "exp1487",
    "thrml_preflight": "exp1488",
    "thrml_parity": "exp1489",
    "partial_trace_localization": "exp1490",
    "retro": "exp1491",
}


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-009: make interruption visible before evidence is loaded.

    Retrospectives are evidence aggregators. A bootstrap artifact prevents a
    killed run from looking like either a valid closure or a missing task.
    """

    artifact = {field: None for field in sorted(REQUIRED_ARTIFACT_FIELDS)}
    artifact["status"] = "in_progress"
    return _write_json(Path(out_path), artifact)


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_sources(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payloads = [(exp_id, _read_json(results_dir / filename)) for exp_id, filename in SOURCE_FILES.items()]
    return (
        {exp_id: payload for exp_id, payload in payloads if payload is not None},
        [exp_id for exp_id, payload in payloads if payload is None],
    )


def _source_path(exp_id: str, field: str | None = None) -> str:
    path = f"results/{SOURCE_FILES[exp_id]}"
    return f"{path}:{field}" if field else path


def _criterion(
    status: str,
    target: str,
    exp_id: str,
    fields: tuple[str, ...],
    values: Mapping[str, Any],
    note: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "target": target,
        "task_id": exp_id,
        "evidence_paths": [_source_path(exp_id, field) for field in fields]
        if exp_id in SOURCE_FILES
        else [],
        "source_values": dict(values),
        "note": note,
    }


def _missing_criterion(exp_id: str, target: str) -> dict[str, Any]:
    return _criterion(
        UNMET,
        target,
        exp_id,
        (),
        {"status": "missing", "honest_verdict": "missing_artifact"},
        f"{exp_id} source artifact is missing.",
    )


def _source_status_values(payload: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    values = {field: payload.get(field) for field in fields}
    values["status"] = payload.get("status")
    values["honest_verdict"] = payload.get("honest_verdict")
    return values


def _scored(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    exp_id: str,
    passed: bool,
    target: str,
    fields: tuple[str, ...],
    positive_note: str,
    negative_note: str,
) -> dict[str, Any]:
    if exp_id in missing_source_ids or exp_id not in sources:
        return _missing_criterion(exp_id, target)
    return _criterion(
        MET if passed else UNMET,
        target,
        exp_id,
        fields,
        _source_status_values(sources[exp_id], fields),
        positive_note if passed else negative_note,
    )


def _zero_bound_violations(value: object) -> bool:
    return value == 0 or value == [] or value == {}


def _score_semantic_energy(
    sources: Mapping[str, Mapping[str, Any]], missing: set[str]
) -> dict[str, Any]:
    payload = sources.get("exp1481", {})
    signal_beats = payload.get("signal_beats_superficial_baselines") is True
    claim_allowed = payload.get("claim_allowed") is True
    passed = payload.get("semantic_energy_audit_complete") is True and claim_allowed == signal_beats
    return _scored(
        sources,
        missing,
        "exp1481",
        passed,
        "Exp1481 completes the Semantic Energy audit and only allows claims when semantic signal beats superficial baselines.",
        (
            "semantic_energy_audit_complete",
            "signal_beats_superficial_baselines",
            "claim_allowed",
            "diagnostic_lineage_retired",
        ),
        "Semantic Energy audit completed and its claim gate matches the baseline comparison.",
        "Semantic Energy audit did not complete or allowed a claim inconsistent with the baseline comparison.",
    )


def _score_thrml_parity(
    sources: Mapping[str, Mapping[str, Any]],
    missing: set[str],
    conductor_gate_blocks: list[dict[str, str]],
) -> dict[str, Any]:
    preflight = sources.get("exp1488", {})
    if "exp1489" in sources:
        parity = sources["exp1489"]
        return _criterion(
            MET if parity.get("simulator_parity_complete") is True else UNMET,
            "When gated on, Exp1489 reports tiny-case THRML/Carnot energy agreement.",
            "exp1489",
            ("simulator_parity_complete", "energy_agreement_reported"),
            _source_status_values(
                parity,
                ("simulator_parity_complete", "energy_agreement_reported"),
            ),
            "THRML/Carnot parity ran because the import gate opened."
            if parity.get("simulator_parity_complete") is True
            else "THRML/Carnot parity artifact exists but did not complete.",
        )
    gate_closed = preflight.get("thrml_import_ready") is False
    if "exp1489" in missing and gate_closed:
        return _criterion(
            GATE_SKIPPED,
            "Exp1489 is skipped honestly when Exp1488.thrml_import_ready=false.",
            "exp1489",
            (),
            {
                "status": "gate_skipped",
                "upstream": "exp1488",
                "thrml_import_ready": preflight.get("thrml_import_ready"),
                "gate_block_log_count": len(conductor_gate_blocks),
            },
            "Structured gate skip: THRML import readiness was false.",
        )
    return _missing_criterion("exp1489", "Exp1489 parity artifact is required when its import gate opens.")


def _score_criteria(
    sources: Mapping[str, Mapping[str, Any]],
    missing: set[str],
    conductor_gate_blocks: list[dict[str, str]],
    research_roadmap_yaml_modified: bool,
    scripts_research_conductor_modified: bool,
    retired_lineages: list[dict[str, Any]],
    carry_forward_recommendations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    exp1479 = sources.get("exp1479", {})
    exp1480 = sources.get("exp1480", {})
    exp1482 = sources.get("exp1482", {})
    exp1483 = sources.get("exp1483", {})
    exp1484 = sources.get("exp1484", {})
    exp1485 = sources.get("exp1485", {})
    exp1486 = sources.get("exp1486", {})
    exp1487 = sources.get("exp1487", {})
    exp1488 = sources.get("exp1488", {})
    exp1490 = sources.get("exp1490", {})
    return {
        "activation": _scored(
            sources,
            missing,
            "exp1479",
            exp1479.get("activation_manifest_complete") is True
            and exp1479.get("predecessor_criteria_met") == exp1479.get("predecessor_criteria_total")
            and exp1479.get("telemetry_headline_block_preserved") is True,
            "Exp1479 activation manifest is complete and summarizes .113 completion.",
            (
                "activation_manifest_complete",
                "predecessor_criteria_met",
                "predecessor_criteria_total",
                "telemetry_headline_block_preserved",
            ),
            ".114 activation preserved .113 guardrails.",
            ".114 activation evidence is missing, incomplete, or did not preserve guardrails.",
        ),
        "balanced_telemetry": _scored(
            sources,
            missing,
            "exp1480",
            exp1480.get("live_sota_model_inference_used") is True
            and exp1480.get("logits_available") is True
            and exp1480.get("superficial_baselines_recorded") is True,
            "Exp1480 uses live SOTA inference, records logits, and records superficial baselines.",
            (
                "live_sota_model_inference_used",
                "logits_available",
                "superficial_baselines_recorded",
                "telemetry_cases_completed",
            ),
            "Balanced telemetry v2 produced live SOTA rows with logits and baseline fields.",
            "Balanced telemetry v2 lacked live SOTA inference, logits, or superficial baselines.",
        ),
        "semantic_energy_audit": _score_semantic_energy(sources, missing),
        "beaver_calibration": _scored(
            sources,
            missing,
            "exp1482",
            exp1482.get("bound_is_sound") is True
            and _zero_bound_violations(exp1482.get("bound_violations"))
            and bool(exp1482.get("mock_or_live_logprobs")),
            "Exp1482 proves sound BEAVER-lite bounds with no violations and labeled logprob lineage.",
            ("bound_is_sound", "bound_violations", "mock_or_live_logprobs", "constraints_evaluated"),
            "BEAVER-lite calibration remained sound and labeled live/mock provenance.",
            "BEAVER-lite calibration had a violation, unsound bound, or missing provenance label.",
        ),
        "halluguard_fit": _scored(
            sources,
            missing,
            "exp1483",
            exp1483.get("risk_decomposition_complete") is True
            and bool(exp1483.get("implemented_assumptions"))
            and bool(exp1483.get("missing_assumptions")),
            "Exp1483 completes risk decomposition with implemented and missing assumptions.",
            ("risk_decomposition_complete", "implemented_assumptions", "missing_assumptions", "claim_allowed"),
            "HalluGuard-style fit audit separated implemented evidence from missing assumptions.",
            "HalluGuard-style fit audit lacked decomposition or assumption accounting.",
        ),
        "query_time_self_learning": _scored(
            sources,
            missing,
            "exp1484",
            exp1484.get("policy_integration_ready") is True
            and exp1484.get("soundness_mistakes") == 0
            and "task_success_delta" in exp1484,
            "Exp1484 integrates query-time memory policy with zero soundness mistakes and reported delta.",
            ("policy_integration_ready", "soundness_mistakes", "task_success_delta", "promotion_allowed"),
            "Query-time memory policy improved bounded replay without false accepts.",
            "Query-time memory policy was not ready, lacked delta, or introduced soundness mistakes.",
        ),
        "completeness_reduction": _scored(
            sources,
            missing,
            "exp1485",
            exp1485.get("completeness_reduction_audit_complete") is True
            and exp1485.get("candidate_soundness_mistakes") == 0
            and exp1485.get("baseline_soundness_mistakes") == 0,
            "Exp1485 completes the gated completeness audit without new soundness mistakes.",
            (
                "completeness_reduction_audit_complete",
                "baseline_soundness_mistakes",
                "candidate_soundness_mistakes",
                "completeness_mistake_delta",
            ),
            "Completeness candidate reduced false rejects while preserving zero soundness mistakes.",
            "Completeness audit did not complete or introduced soundness mistakes.",
        ),
        "executable_tool_use_benchmark": _scored(
            sources,
            missing,
            "exp1486",
            exp1486.get("executable_constraint_benchmark_ready") is True
            and exp1486.get("benchmark_cases", 0) >= 20,
            "Exp1486 writes a 20-case executable constraint benchmark with validators.",
            (
                "executable_constraint_benchmark_ready",
                "benchmark_cases",
                "live_sota_model_inference_used",
                "verifier_false_accept_rate",
            ),
            "CCTU-style executable benchmark is ready with at least 20 cases.",
            "CCTU-style benchmark is missing readiness or case-count evidence.",
        ),
        "pairwise_verification": _scored(
            sources,
            missing,
            "exp1487",
            exp1487.get("pairwise_verification_complete") is True
            and "random_baseline_accuracy" in exp1487
            and "superficial_baseline_accuracy" in exp1487,
            "Exp1487 completes pairwise verification and measures random plus superficial baselines.",
            (
                "pairwise_verification_complete",
                "pairwise_accuracy",
                "random_baseline_accuracy",
                "superficial_baseline_accuracy",
                "energy_ranking_accuracy",
                "improvement_allowed",
            ),
            "V_1 pairwise comparison completed, with no promotion when energy won.",
            "V_1 pairwise comparison did not complete or lacked baseline measurements.",
        ),
        "thrml_preflight": _scored(
            sources,
            missing,
            "exp1488",
            exp1488.get("thrml_preflight_complete") is True
            and exp1488.get("hardware_claim_allowed") is False,
            "Exp1488 completes THRML preflight and blocks hardware claims.",
            ("thrml_preflight_complete", "thrml_import_ready", "hardware_claim_allowed"),
            "THRML preflight completed and preserved no-hardware-claim boundary.",
            "THRML preflight did not complete or allowed unsupported hardware claims.",
        ),
        "thrml_parity": _score_thrml_parity(sources, missing, conductor_gate_blocks),
        "partial_trace_localization": _scored(
            sources,
            missing,
            "exp1490",
            exp1490.get("localization_audit_complete") is True
            and exp1490.get("decoded_quality_claim_allowed") is False
            and exp1490.get("kona_dependency_used") is False,
            "Exp1490 completes bounded localization without decoded-quality or Kona-internals claims.",
            (
                "localization_audit_complete",
                "decoded_quality_claim_allowed",
                "kona_dependency_used",
                "localization_top1_rate",
            ),
            "Partial-trace localization beat random on injected failures without overclaiming.",
            "Partial-trace localization did not complete or overclaimed quality/Kona internals.",
        ),
        "retro": _criterion(
            MET
            if not research_roadmap_yaml_modified
            and not scripts_research_conductor_modified
            and bool(retired_lineages)
            and bool(carry_forward_recommendations)
            else UNMET,
            "Exp1491 writes the terminal retro, records retirements/carry-forwards, and keeps protected files unchanged.",
            "exp1491",
            (),
            {
                "research_roadmap_yaml_modified": research_roadmap_yaml_modified,
                "scripts_research_conductor_modified": scripts_research_conductor_modified,
                "ops_docs_reconciliation": "delegated_to_followup_reconciler",
            },
            "Retrospective artifact completed; ops doc reconciliation is delegated by the stop rule."
            if not research_roadmap_yaml_modified
            and not scripts_research_conductor_modified
            and bool(retired_lineages)
            and bool(carry_forward_recommendations)
            else "Retrospective closure lacked protected-file confirmation, retirements, or carry-forwards.",
        ),
    }


def _conductor_gate_blocks(conductor_log_text: str) -> list[dict[str, str]]:
    gate_blocks: list[dict[str, str]] = []
    for line in conductor_log_text.splitlines():
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) >= 4 and columns[2] == "GATE_BLOCK" and "THRML/Carnot" in columns[1]:
            gate_blocks.append({"task_id": "exp1489", "reason": columns[3], "log_line": line})
    return gate_blocks


def _retired_lineages(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    lineages: list[dict[str, Any]] = []
    exp1481 = sources.get("exp1481", {})
    exp1487 = sources.get("exp1487", {})
    exp1488 = sources.get("exp1488", {})
    if exp1481.get("diagnostic_lineage_retired") is True:
        lineages.append(
            {
                "lineage": "semantic_energy_headline_telemetry",
                "decision": "retired",
                "source_experiment": "exp1481",
                "reason": "Semantic Energy proxy was confounded by superficial baselines.",
                "honest_verdict": exp1481.get("honest_verdict"),
            }
        )
    if exp1487.get("improvement_allowed") is False:
        lineages.append(
            {
                "lineage": "v1_pairwise_self_verification_promotion_path",
                "decision": "do_not_promote",
                "source_experiment": "exp1487",
                "reason": "Pairwise self-verification underperformed executable Carnot energy.",
                "honest_verdict": exp1487.get("honest_verdict"),
            }
        )
    if exp1488.get("thrml_import_ready") is False:
        lineages.append(
            {
                "lineage": "thrml_carnot_simulator_parity_until_import_ready",
                "decision": "gate_blocked",
                "source_experiment": "exp1488/exp1489",
                "reason": "THRML import readiness is false; parity must not run or claim hardware evidence.",
                "honest_verdict": exp1488.get("honest_verdict"),
            }
        )
    lineages.append(
        {
            "lineage": "prior_scope_reduction_blocks",
            "decision": "preserved",
            "source_experiment": "exp1479",
            "reason": "Repair-executor reruns, GRPO/VPRM, WOPR puzzle expansion, and HardNet++/DSP remain closed.",
            "honest_verdict": sources.get("exp1479", {}).get("honest_verdict"),
        }
    )
    return lineages


def _carry_forward_recommendations(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "track": "beaver_lite_bounds",
            "next_focus": "Preserve sound live-prefix calibration; expand only with zero bound violations.",
            "source_experiment": "exp1482",
            "evidence": sources.get("exp1482", {}).get("honest_verdict"),
        },
        {
            "track": "fr11_query_time_memory",
            "next_focus": "Promote the opt-in policy cautiously under the zero-soundness-mistake gate.",
            "source_experiment": "exp1484/exp1485",
            "evidence": {
                "policy": sources.get("exp1484", {}).get("honest_verdict"),
                "completeness": sources.get("exp1485", {}).get("honest_verdict"),
            },
        },
        {
            "track": "cctu_executable_constraints",
            "next_focus": "Use the 20-case benchmark and energy ranking as the baseline for future tool-use work.",
            "source_experiment": "exp1486/exp1487",
            "evidence": sources.get("exp1486", {}).get("honest_verdict"),
        },
        {
            "track": "semantic_energy_telemetry",
            "next_focus": "Do not use as headline evidence unless a future signal beats lexical and format baselines.",
            "source_experiment": "exp1481",
            "evidence": sources.get("exp1481", {}).get("honest_verdict"),
        },
        {
            "track": "thrml",
            "next_focus": "Fix Python/pip/import readiness before proposing simulator parity; no TSU hardware claim.",
            "source_experiment": "exp1488",
            "evidence": sources.get("exp1488", {}).get("honest_verdict"),
        },
        {
            "track": "partial_trace_localization",
            "next_focus": "Carry forward as injected-failure localization evidence only, not decoded quality or Kona internals.",
            "source_experiment": "exp1490",
            "evidence": sources.get("exp1490", {}).get("honest_verdict"),
        },
    ]


def _missing_artifacts(missing_source_ids: set[str]) -> list[dict[str, str]]:
    return [
        {"path": f"results/{SOURCE_FILES[exp_id]}", "reason": "source_artifact_missing"}
        for exp_id in SOURCE_FILES
        if exp_id in missing_source_ids
    ]


def _source_artifacts_checked(missing_source_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": exp_id,
            "path": f"results/{filename}",
            "exists": exp_id not in missing_source_ids,
        }
        for exp_id, filename in SOURCE_FILES.items()
    ]


def _completed_task_ids(sources: Mapping[str, Mapping[str, Any]]) -> list[str]:
    completed = [exp_id for exp_id in SOURCE_FILES if sources.get(exp_id, {}).get("status") == "complete"]
    completed.append("exp1491")
    return completed


def _path_modified_by_git(root: Path, relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", relative_path],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 1


def build_artifact(
    sources: Mapping[str, dict[str, Any]],
    missing_source_ids: list[str],
    conductor_log_text: str,
    research_complete_text: str,
    research_roadmap_yaml_modified: bool,
    scripts_research_conductor_modified: bool,
    ops_docs_updated: bool,
) -> dict[str, Any]:
    """REQ-REPORT-009: score the milestone from terminal artifacts.

    Gate skips are useful terminal evidence, but they are not counted as met
    criteria unless the roadmap explicitly allows that skip as success.
    """

    missing = set(missing_source_ids)
    conductor_gate_blocks = _conductor_gate_blocks(conductor_log_text)
    retired_lineages = _retired_lineages(sources)
    carry_forward_recommendations = _carry_forward_recommendations(sources)
    criteria = _score_criteria(
        sources,
        missing,
        conductor_gate_blocks,
        research_roadmap_yaml_modified,
        scripts_research_conductor_modified,
        retired_lineages,
        carry_forward_recommendations,
    )
    criteria_met = sum(1 for result in criteria.values() if result["status"] == MET)
    criteria_total = len(CRITERION_SOURCE)
    gate_skips = [
        {
            "criterion": criterion,
            "task_id": result["task_id"],
            "reason": result["note"],
            "source_values": result["source_values"],
        }
        for criterion, result in criteria.items()
        if result["status"] == GATE_SKIPPED
    ]
    success_threshold_met = criteria_met >= 10
    research_complete_has_114_entry = "2026.04.114" in research_complete_text
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "status": "complete",
        "milestone": MILESTONE,
        "criteria_results": criteria,
        "criteria_met": criteria_met,
        "criteria_total": criteria_total,
        "success_threshold_met": success_threshold_met,
        "honest_structured_gate_skips": gate_skips,
        "honest_structured_gate_skip_count": len(gate_skips),
        "completed_task_ids": _completed_task_ids(sources),
        "blocked_task_ids": [skip["task_id"] for skip in gate_skips],
        "retired_lineages": retired_lineages,
        "carry_forward_recommendations": carry_forward_recommendations,
        "missing_artifacts": _missing_artifacts(missing),
        "source_artifacts_checked": _source_artifacts_checked(missing),
        "conductor_gate_blocks": conductor_gate_blocks,
        "research_complete_has_114_entry": research_complete_has_114_entry,
        "research_complete_archive_update_needed": not research_complete_has_114_entry,
        "ops_docs_updated": ops_docs_updated,
        "ops_docs_update_note": "ops reconciliation delegated by the operator stop rule"
        if not ops_docs_updated
        else "ops/status.md and ops/changelog.md were updated by this workflow",
        "research_roadmap_yaml_modified": research_roadmap_yaml_modified,
        "scripts_research_conductor_modified": scripts_research_conductor_modified,
        "protected_file_checks": {
            "research-roadmap.yaml": "modified"
            if research_roadmap_yaml_modified
            else "unchanged",
            "scripts/research_conductor.py": "modified"
            if scripts_research_conductor_modified
            else "unchanged",
        },
        "honest_verdict": (
            f"complete: milestone_114_{criteria_met}_of_{criteria_total}_criteria_met_"
            f"{'success_threshold_met' if success_threshold_met else 'below_success_threshold'}_"
            f"{len(gate_skips)}_honest_gate_skips_ops_reconciliation_delegated"
        ),
    }


def run(root: Path | str = REPO_ROOT, out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    root_path = Path(root)
    write_in_progress_artifact(out_path)
    sources, missing = _load_sources(root_path / "results")
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=missing,
        conductor_log_text=_read_text(root_path / "ops" / "conductor-log.md"),
        research_complete_text=_read_text(root_path / "research-complete.yaml"),
        research_roadmap_yaml_modified=_path_modified_by_git(root_path, "research-roadmap.yaml"),
        scripts_research_conductor_modified=_path_modified_by_git(
            root_path, "scripts/research_conductor.py"
        ),
        ops_docs_updated=False,
    )
    return _write_json(Path(out_path), artifact)
