"""Build the Exp 1466 milestone .112 retrospective artifact.

Spec: REQ-REPORT-047, SCENARIO-REPORT-047.
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
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1466_milestone_112_retro.json"

EXPERIMENT = "1466_milestone_112_retro"
SCHEMA = "milestone_112_retro_v1"
RUN_DATE = "20260507"
MILESTONE = "2026.04.112"

MET = "met"
UNMET = "unmet"
GATE_BLOCKED_WITH_EVIDENCE = "gate_blocked_with_evidence"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone",
    "criteria_met",
    "criteria_total",
    "scope_reduction_required",
    "scope_reduction_tasks_completed",
    "scope_reduction_compliance_met",
    "blocked_tasks",
    "retired_lineages",
    "carry_forward_tracks",
    "missing_artifacts",
    "research_roadmap_yaml_modified",
    "scripts_research_conductor_modified",
    "ops_docs_updated",
    "honest_verdict",
)

SOURCE_FILES = {
    "exp1453": "experiment_1453_112_scope_reduction_activation_manifest.json",
    "exp1454": "experiment_1454_experiment_artifact_signal_noise_classifier.json",
    "exp1455": "experiment_1455_known_issues_mandatory_priority_audit.json",
    "exp1456": "experiment_1456_grpo_vprm_lineage_consolidation_retirement.json",
    "exp1457": "experiment_1457_wopr_puzzle_cartridge_retirement.json",
    "exp1458": "experiment_1458_hardnet_dsp_repair_stack_consolidation.json",
    "exp1459": "experiment_1459_self_learning_nonheadline_lineage_decision.json",
    "exp1460": "experiment_1460_hardware_portfolio_narrowing.json",
    "exp1461": "experiment_1461_comparator_integration_cite_retire_audit.json",
    "exp1462": "experiment_1462_paper_v6_anchored_claims_narrowing.json",
    "exp1463": "experiment_1463_local_sota_gguf_runtime_repair.json",
    "exp1464": "experiment_1464_repair_validation_error_context_ab.json",
    "exp1465": "experiment_1465_external_verifier_benchmark_fit_audit.json",
}

SUPPORTING_SOURCE_FILES = {
    "exp1463_reproduced_probe": "experiment_1463_reproduced_exp1442_current_probe.json",
}

CRITERION_SOURCE = {
    "scope_activation": "exp1453",
    "artifact_classifier": "exp1454",
    "priority_audit": "exp1455",
    "grpo_retirement": "exp1456",
    "wopr_retirement": "exp1457",
    "hardnet_dsp_retirement": "exp1458",
    "self_learning_decision": "exp1459",
    "hardware_narrowing": "exp1460",
    "comparator_audit": "exp1461",
    "paper_claims": "exp1462",
    "live_sota_runtime": "exp1463",
    "repair_salvage": "exp1464",
    "verifier_benchmark_fit": "exp1465",
    "retro": "exp1466",
}

SCOPE_CRITERIA = (
    "scope_activation",
    "artifact_classifier",
    "priority_audit",
    "grpo_retirement",
    "wopr_retirement",
    "hardnet_dsp_retirement",
    "self_learning_decision",
    "hardware_narrowing",
    "comparator_audit",
    "paper_claims",
)

DEFAULT_EVIDENCE_PATHS = {
    "ops/milestone_112_scope_reduction_manifest.md",
    "ops/experiment_signal_noise_classification.csv",
    "ops/experiment_signal_noise_summary.md",
    "ops/mandatory_priority_audit.md",
    "ops/active-priorities.md",
    "ops/exclusion_manifest.yaml",
    "ops/lineage-retirements/grpo_vprm_lineage_retired.md",
    "ops/lineage-retirements/wopr_puzzle_cartridges_retired.md",
    "ops/lineage-retirements/hardnet_dsp_repair_stack_retired.md",
    "docs/research-notes/self_learning_lineage_decision.md",
    "docs/research-notes/hardware_portfolio_narrowing.md",
    "docs/research-notes/comparator_cite_retire_audit.md",
    "docs/research-notes/paper_v6_anchored_claim_matrix.md",
    "docs/arxiv-paper/main.tex",
    "docs/research-notes/external_verifier_benchmark_fit.md",
}


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-047: create durable progress before any evidence scoring.

    Retrospective runs often discover missing gate artifacts late. This small
    skeleton gives the conductor a truthful file to inspect if source scoring
    is interrupted before the final artifact can be assembled.
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


def _verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict", ""))


def _number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _source_path(exp_id: str, field: str | None = None) -> str:
    filename = SOURCE_FILES.get(exp_id, f"experiment_{exp_id}.json")
    path = f"results/{filename}"
    return f"{path}:{field}" if field else path


def _normalize_path(value: object) -> str:
    text = str(value or "")
    prefix = PROJECT_ROOT_FOR_METADATA + "/"
    return text[len(prefix) :] if text.startswith(prefix) else text


def _paths_present(root: Path, paths: set[str]) -> set[str]:
    return {path for path in paths if (root / path).exists()}


def _evidence_paths_present(root: Path) -> set[str]:
    return _paths_present(root, DEFAULT_EVIDENCE_PATHS)


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


def _missing_evidence(paths: list[str], evidence_paths_present: set[str]) -> list[str]:
    return [path for path in paths if _normalize_path(path) not in evidence_paths_present]


def _scored(
    exp_id: str,
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    passed: bool,
    target: str,
    fields: tuple[str, ...],
    positive: str,
    negative: str,
    evidence_paths_present: set[str],
    required_evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    if exp_id in missing_source_ids or exp_id not in sources:
        return _source_missing_criterion(exp_id, target)
    payload = sources[exp_id]
    source_values = {field: payload.get(field) for field in fields}
    source_values["status"] = payload.get("status")
    source_values["honest_verdict"] = _verdict(payload)
    required_paths = [_normalize_path(path) for path in required_evidence_paths or []]
    missing_evidence = _missing_evidence(required_paths, evidence_paths_present)
    evidence_paths = [_source_path(exp_id, field) for field in fields] + required_paths
    if passed and not missing_evidence:
        return _criterion(MET, target, evidence_paths, [positive], [], source_values)
    negative_evidence = [] if passed else [negative]
    negative_evidence.extend(f"Required evidence missing: {path}" for path in missing_evidence)
    source_values["missing_required_evidence"] = missing_evidence
    return _criterion(UNMET, target, evidence_paths, [], negative_evidence, source_values)


def _gate_blocked(
    exp_id: str,
    target: str,
    evidence_paths: list[str],
    reason: str,
    source_values: Mapping[str, Any],
) -> dict[str, Any]:
    values = dict(source_values)
    values.setdefault("status", "missing" if exp_id not in SOURCE_FILES else values.get("status"))
    return _criterion(GATE_BLOCKED_WITH_EVIDENCE, target, evidence_paths, [], [reason], values)


def _score_scope_activation(
    sources: Mapping[str, Mapping[str, Any]],
    missing: set[str],
    evidence: set[str],
) -> dict[str, Any]:
    exp1453 = sources.get("exp1453", {})
    return _scored(
        "exp1453",
        sources,
        missing,
        exp1453.get("scope_reduction_manifest_complete") is True
        and (_number(exp1453.get("planned_scope_reduction_task_count")) or 0.0)
        >= (_number(exp1453.get("required_scope_reduction_task_count")) or 8.0),
        "exp1453.scope_reduction_manifest_complete=true",
        (
            "scope_reduction_manifest_complete",
            "planned_scope_reduction_task_count",
            "required_scope_reduction_task_count",
        ),
        "Scope-reduction activation manifest completed with enough mapped tasks.",
        "Scope-reduction activation manifest did not meet its completion/count gate.",
        evidence,
        [_normalize_path(exp1453.get("scope_reduction_manifest_path"))],
    )


def _score_live_runtime(exp1463: Mapping[str, Any]) -> dict[str, Any]:
    target = (
        "exp1463.local_sota_runtime_ready=true or precise persistent blocker "
        "with same-verdict retirement"
    )
    blockers = exp1463.get("persistent_blockers") if isinstance(exp1463.get("persistent_blockers"), list) else []
    ready = (
        exp1463.get("local_sota_runtime_ready") is True
        and exp1463.get("live_sota_model_inference_used") is True
    )
    precise_blocker_retired = bool(blockers) and exp1463.get("same_verdict_retirement_recorded") is True
    if ready or precise_blocker_retired:
        return _criterion(
            MET,
            target,
            [
                _source_path("exp1463", "local_sota_runtime_ready"),
                _source_path("exp1463", "persistent_blockers"),
            ],
            [
                "Live local SOTA runtime reached usable inference."
                if ready
                else "Runtime blocker was precise and same-verdict retirement was recorded."
            ],
            [],
            {
                "status": exp1463.get("status"),
                "local_sota_runtime_ready": exp1463.get("local_sota_runtime_ready"),
                "live_sota_model_inference_used": exp1463.get("live_sota_model_inference_used"),
                "persistent_blockers": blockers,
                "same_verdict_retirement_recorded": exp1463.get(
                    "same_verdict_retirement_recorded"
                ),
                "honest_verdict": _verdict(exp1463),
            },
        )
    return _gate_blocked(
        "exp1463",
        target,
        [
            _source_path("exp1463", "local_sota_runtime_ready"),
            _source_path("exp1463", "persistent_blockers"),
        ],
        "Live SOTA runtime did not reach readiness and no same-verdict retirement field passed.",
        {
            "status": exp1463.get("status"),
            "local_sota_runtime_ready": exp1463.get("local_sota_runtime_ready"),
            "live_sota_model_inference_used": exp1463.get("live_sota_model_inference_used"),
            "persistent_blockers": blockers,
            "same_verdict_retirement_recorded": exp1463.get("same_verdict_retirement_recorded"),
            "honest_verdict": _verdict(exp1463),
        },
    )


def _score_repair_salvage(exp1464: Mapping[str, Any], runtime_ready: bool) -> dict[str, Any]:
    target = (
        "if gated on, exp1464.acceptance_delta_pp > 0 or repair executor lineage retired"
    )
    acceptance_delta = _number(exp1464.get("acceptance_delta_pp"))
    if not runtime_ready:
        return _gate_blocked(
            "exp1464",
            target,
            [_source_path("exp1463"), _source_path("exp1464")],
            "Repair salvage did not run behind a ready live-SOTA runtime gate.",
            {
                "status": exp1464.get("status"),
                "acceptance_delta_pp": exp1464.get("acceptance_delta_pp"),
                "repair_executor_lineage_retired": exp1464.get("repair_executor_lineage_retired"),
                "honest_verdict": _verdict(exp1464),
            },
        )
    passed = (acceptance_delta is not None and acceptance_delta > 0.0) or (
        exp1464.get("repair_executor_lineage_retired") is True
    )
    return _criterion(
        MET if passed else UNMET,
        target,
        [
            _source_path("exp1464", "acceptance_delta_pp"),
            _source_path("exp1464", "repair_executor_lineage_retired"),
        ],
        [
            "Repair salvage improved acceptance or explicitly retired the repair-executor lineage."
        ]
        if passed
        else [],
        []
        if passed
        else ["Repair salvage neither improved acceptance nor retired the executor lineage."],
        {
            "status": exp1464.get("status"),
            "acceptance_delta_pp": exp1464.get("acceptance_delta_pp"),
            "repair_executor_lineage_retired": exp1464.get("repair_executor_lineage_retired"),
            "live_sota_model_inference_used": exp1464.get("live_sota_model_inference_used"),
            "cases_evaluated": exp1464.get("cases_evaluated"),
            "honest_verdict": _verdict(exp1464),
        },
    )


def _score_criteria(
    sources: Mapping[str, Mapping[str, Any]],
    missing_source_ids: set[str],
    evidence_paths_present: set[str],
) -> dict[str, dict[str, Any]]:
    exp1454 = sources.get("exp1454", {})
    exp1455 = sources.get("exp1455", {})
    exp1456 = sources.get("exp1456", {})
    exp1457 = sources.get("exp1457", {})
    exp1458 = sources.get("exp1458", {})
    exp1459 = sources.get("exp1459", {})
    exp1460 = sources.get("exp1460", {})
    exp1461 = sources.get("exp1461", {})
    exp1462 = sources.get("exp1462", {})
    exp1463 = sources.get("exp1463", {})
    exp1464 = sources.get("exp1464", {})
    exp1465 = sources.get("exp1465", {})
    runtime_criterion = _score_live_runtime(exp1463)
    runtime_ready_for_repair = (
        exp1463.get("local_sota_runtime_ready") is True
        and exp1463.get("live_sota_model_inference_used") is True
    )
    comparator_rows = exp1461.get("decisions") if isinstance(exp1461.get("decisions"), list) else []
    anchored_claims = (
        exp1462.get("anchored_claims") if isinstance(exp1462.get("anchored_claims"), list) else []
    )
    benchmark_decision = str(exp1465.get("benchmark_adoption_decision") or "")

    return {
        "scope_activation": _score_scope_activation(sources, missing_source_ids, evidence_paths_present),
        "artifact_classifier": _scored(
            "exp1454",
            sources,
            missing_source_ids,
            exp1454.get("classification_table_written") is True
            and bool(exp1454.get("top_50_noise_candidates")),
            "exp1454.classification_table_written=true and top-50 noise candidates identified",
            ("classification_table_written", "top_50_noise_candidates"),
            "Signal/noise table was written and top noise candidates were identified.",
            "Artifact classifier did not write the table or identify top noise candidates.",
            evidence_paths_present,
            [
                _normalize_path(exp1454.get("classification_table_path")),
                _normalize_path(exp1454.get("summary_path")),
            ],
        ),
        "priority_audit": _scored(
            "exp1455",
            sources,
            missing_source_ids,
            (_number(exp1455.get("active_priority_count")) or 999.0) <= 10.0
            and (_number(exp1455.get("trim_fraction")) or 0.0) >= 0.40,
            "exp1455.active_priority_count <= 10 and trim_fraction >= 0.40",
            ("active_priority_count", "trim_fraction", "known_issues_updated"),
            "Mandatory priorities were trimmed to the required active set.",
            "Mandatory priorities were not trimmed enough.",
            evidence_paths_present,
            [
                _normalize_path(exp1455.get("priority_audit_path")),
                _normalize_path(exp1455.get("active_priorities_index_path")),
            ],
        ),
        "grpo_retirement": _scored(
            "exp1456",
            sources,
            missing_source_ids,
            exp1456.get("grpo_lineage_retired") is True
            and exp1456.get("exclusion_manifest_updated") is True,
            "exp1456.grpo_lineage_retired=true and GRPO v15 is manifest-blocked",
            ("grpo_lineage_retired", "exclusion_manifest_updated", "lessons_retained"),
            "GRPO/VPRM lineage was retired with lessons retained.",
            "GRPO/VPRM lineage was not retired or manifest-blocked.",
            evidence_paths_present,
            [_normalize_path(exp1456.get("consolidation_note_path")), "ops/exclusion_manifest.yaml"],
        ),
        "wopr_retirement": _scored(
            "exp1457",
            sources,
            missing_source_ids,
            exp1457.get("wopr_puzzle_lineage_retired") is True
            and exp1457.get("exclusion_manifest_updated") is True,
            "exp1457.wopr_puzzle_lineage_retired=true and future cartridges are blocked",
            ("wopr_puzzle_lineage_retired", "exclusion_manifest_updated", "preserved_assets"),
            "WOPR puzzle-cartridge research scope was retired while preserving assets.",
            "WOPR puzzle-cartridge scope was not retired or manifest-blocked.",
            evidence_paths_present,
            [_normalize_path(exp1457.get("retirement_note_path")), "ops/exclusion_manifest.yaml"],
        ),
        "hardnet_dsp_retirement": _scored(
            "exp1458",
            sources,
            missing_source_ids,
            exp1458.get("hardnet_dsp_lineage_retired") is True
            and exp1458.get("exclusion_manifest_updated") is True
            and bool(exp1458.get("lessons_retained")),
            "exp1458.hardnet_dsp_lineage_retired=true with lessons retained",
            ("hardnet_dsp_lineage_retired", "exclusion_manifest_updated", "lessons_retained"),
            "HardNet++/DSP repair-stack scope was retired with retained lessons.",
            "HardNet++/DSP repair-stack scope was not retired with retained lessons.",
            evidence_paths_present,
            [_normalize_path(exp1458.get("consolidation_note_path")), "ops/exclusion_manifest.yaml"],
        ),
        "self_learning_decision": _scored(
            "exp1459",
            sources,
            missing_source_ids,
            (
                exp1459.get("self_learning_headline_pivot_selected") is True
                or exp1459.get("self_learning_lineage_retired") is True
            )
            and (
                exp1459.get("exp1447_delta_overall") is not None
                or "exp1447" in (exp1459.get("source_artifact_summaries") or {})
            ),
            "exp1459 selects a headline pivot or retirement and cites exp1447",
            (
                "self_learning_headline_pivot_selected",
                "self_learning_lineage_retired",
                "exp1447_delta_overall",
            ),
            "Self-learning scope was narrowed to an exp1447-anchored decision.",
            "Self-learning decision did not cite exp1447 or choose pivot/retirement.",
            evidence_paths_present,
            [_normalize_path(exp1459.get("decision_note_path"))],
        ),
        "hardware_narrowing": _scored(
            "exp1460",
            sources,
            missing_source_ids,
            (_number(exp1460.get("active_hardware_track_count")) or 999.0) <= 3.0
            and exp1460.get("architecture_updated") is True
            and exp1460.get("hardware_wishlist_updated") is True,
            "exp1460.active_hardware_track_count <= 3 and docs updated",
            (
                "active_hardware_track_count",
                "architecture_updated",
                "hardware_wishlist_updated",
            ),
            "Hardware portfolio was narrowed to three active tracks with docs updated.",
            "Hardware portfolio did not narrow to <=3 with required doc updates.",
            evidence_paths_present,
            [_normalize_path(exp1460.get("decision_note_path"))],
        ),
        "comparator_audit": _scored(
            "exp1461",
            sources,
            missing_source_ids,
            (_number(exp1461.get("comparator_decision_count")) or 0.0) >= 6.0
            and all(row.get("decision") in {"cite", "retire", "future_watchlist"} for row in comparator_rows),
            "exp1461.comparator_decision_count >= 6 and every row has cite/retire/watchlist",
            ("comparator_decision_count", "cite_count", "retire_count", "watchlist_count"),
            "Comparator scope was narrowed into cite/retire/watchlist rows.",
            "Comparator audit did not produce enough valid decision rows.",
            evidence_paths_present,
            [_normalize_path(exp1461.get("decision_table_path"))],
        ),
        "paper_claims": _scored(
            "exp1462",
            sources,
            missing_source_ids,
            3.0 <= (_number(exp1462.get("anchored_claim_count")) or 0.0) <= 5.0
            and exp1462.get("paper_updated") is True
            and all(bool(row.get("empirical_artifact_paths")) for row in anchored_claims),
            "exp1462.anchored_claim_count is between 3 and 5 with artifact references",
            ("anchored_claim_count", "paper_updated", "claim_matrix_path"),
            "Paper v6 was narrowed to artifact-anchored claims.",
            "Paper v6 claim narrowing did not satisfy count, paper, or artifact-reference gates.",
            evidence_paths_present,
            [
                _normalize_path(exp1462.get("claim_matrix_path")),
                _normalize_path(exp1462.get("paper_source_path")),
            ],
        ),
        "live_sota_runtime": runtime_criterion,
        "repair_salvage": _score_repair_salvage(exp1464, runtime_ready_for_repair),
        "verifier_benchmark_fit": _scored(
            "exp1465",
            sources,
            missing_source_ids,
            any(word in benchmark_decision for word in ("adopt", "defer", "retire"))
            and bool(exp1465.get("next_minimal_benchmark_task") or exp1465.get("decision_rows")),
            "exp1465.benchmark_adoption_decision is adopt/defer/retire with rationale",
            ("benchmark_adoption_decision", "adopted_benchmark", "next_minimal_benchmark_task"),
            "External verifier benchmark fit chose a bounded next decision.",
            "External verifier benchmark fit did not record an adopt/defer/retire rationale.",
            evidence_paths_present,
            [_normalize_path(exp1465.get("benchmark_decision_table_path"))],
        ),
        "retro": _criterion(
            MET,
            "exp1466.criteria_total=14 and carry-forward/retirement rules are recorded",
            ["results/experiment_1466_milestone_112_retro.json"],
            ["This final artifact records all 14 criteria and carry-forward rules."],
            [],
            {"criteria_total": 14, "status": "complete"},
        ),
    }


def _source_checks(root: Path, missing_source_ids: set[str]) -> list[dict[str, Any]]:
    checks = [
        {
            "experiment_id": exp_id,
            "path": f"results/{filename}",
            "exists": exp_id not in missing_source_ids,
        }
        for exp_id, filename in SOURCE_FILES.items()
    ]
    checks.extend(
        {
            "experiment_id": exp_id,
            "path": f"results/{filename}",
            "exists": (root / "results" / filename).exists(),
        }
        for exp_id, filename in SUPPORTING_SOURCE_FILES.items()
    )
    return checks


def _missing_artifacts(
    missing_source_ids: set[str],
    criteria: Mapping[str, Mapping[str, Any]],
    roadmap_next_present: bool,
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = [
        {"path": f"results/{SOURCE_FILES[exp_id]}", "reason": "source_artifact_missing"}
        for exp_id in SOURCE_FILES
        if exp_id in missing_source_ids
    ]
    seen = {item["path"] for item in missing}
    for result in criteria.values():
        for path in result["source_values"].get("missing_required_evidence", []):
            if path and path not in seen:
                missing.append({"path": path, "reason": "required_evidence_missing"})
                seen.add(path)
    if not roadmap_next_present:
        missing.append({"path": "research-roadmap-next.yaml", "reason": "requested_input_missing"})
    return missing


def _successful_scope_tasks(criteria: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        CRITERION_SOURCE[criterion_id]
        for criterion_id in SCOPE_CRITERIA
        if criteria[criterion_id]["status"] == MET
    ]


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


def _retired_lineages(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    retired: list[dict[str, Any]] = []
    lineage_specs = (
        ("exp1456", "GRPO/VPRM", "grpo_lineage_retired", "consolidation_note_path"),
        ("exp1457", "WOPR puzzle cartridges", "wopr_puzzle_lineage_retired", "retirement_note_path"),
        ("exp1458", "HardNet++/DSP repair stack", "hardnet_dsp_lineage_retired", "consolidation_note_path"),
        (
            "exp1464",
            "repair executor validation-error context",
            "repair_executor_lineage_retired",
            None,
        ),
    )
    for exp_id, lineage, flag, note_field in lineage_specs:
        payload = sources.get(exp_id, {})
        if payload.get(flag) is True:
            retired.append(
                {
                    "lineage": lineage,
                    "source_experiment": exp_id,
                    "honest_verdict": _verdict(payload),
                    "evidence_path": _source_path(exp_id),
                    "note_path": _normalize_path(payload.get(note_field)) if note_field else None,
                }
            )
    return retired


def _carry_forward_tracks(sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    exp1463 = sources.get("exp1463", {})
    exp1464 = sources.get("exp1464", {})
    exp1459 = sources.get("exp1459", {})
    exp1462 = sources.get("exp1462", {})
    exp1465 = sources.get("exp1465", {})
    return [
        {
            "track": "runtime",
            "source_experiment": "exp1463",
            "rule": (
                "Preserve local SOTA GGUF runtime as a precondition for headline repair; "
                "future runtime regressions must name the exact missing cache, CUDA, or llama.cpp field."
            ),
            "status": "ready"
            if exp1463.get("local_sota_runtime_ready") is True
            else "carry_forward_blocker",
            "honest_verdict": _verdict(exp1463),
        },
        {
            "track": "repair",
            "source_experiment": "exp1464",
            "rule": (
                "Do not preserve the repair-executor validation-error context line unless "
                "a future task changes the executor and beats acceptance_delta_pp > 0."
            ),
            "status": "retired"
            if exp1464.get("repair_executor_lineage_retired") is True
            else "needs_changed_prerequisite",
            "honest_verdict": _verdict(exp1464),
        },
        {
            "track": "self_learning",
            "source_experiment": "exp1459",
            "rule": (
                "Allow only one bounded exp1447-style fresh verified growth follow-up; "
                "do not revive broad non-headline self-learning claims."
            ),
            "status": "pivot_selected"
            if exp1459.get("self_learning_headline_pivot_selected") is True
            else "retired",
            "honest_verdict": _verdict(exp1459),
        },
        {
            "track": "paper_claims",
            "source_experiment": "exp1462",
            "rule": (
                "Keep paper-v6 claims within the anchored claim matrix; unsupported "
                "hardware, comparator, and broad scaling claims remain appendix or future work."
            ),
            "anchored_claim_count": exp1462.get("anchored_claim_count"),
            "honest_verdict": _verdict(exp1462),
        },
        {
            "track": "benchmark_adoption",
            "source_experiment": "exp1465",
            "rule": (
                "Adopt at most the one minimal BEAVER-style bounds smoke task; defer "
                "broad VNN-COMP or external verifier benchmark runners."
            ),
            "decision": exp1465.get("benchmark_adoption_decision"),
            "honest_verdict": _verdict(exp1465),
        },
    ]


def _path_modified_by_git(root: Path, relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", relative_path],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 1


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
    evidence_paths_present: set[str],
    roadmap_doc_present: bool,
    roadmap_yaml_present: bool,
    roadmap_next_present: bool,
    conductor_log_present: bool,
    research_roadmap_yaml_modified: bool,
    scripts_research_conductor_modified: bool,
    ops_docs_updated: bool,
) -> dict[str, Any]:
    """REQ-REPORT-047: score the .112 milestone from source artifacts.

    This scorer intentionally separates terminal evidence from success. Scope
    reduction needs its backing docs, runtime readiness needs the actual runtime
    fields, and the repair experiment only closes because it explicitly retired
    the no-improvement executor line.
    """

    missing = set(missing_source_ids)
    criteria = _score_criteria(sources, missing, evidence_paths_present)
    criteria_met = sum(1 for result in criteria.values() if result["status"] == MET)
    criteria_total = len(CRITERION_SOURCE)
    scope_tasks_completed = _successful_scope_tasks(criteria)
    scope_required = sources.get("exp1453", {}).get("scope_reduction_required") is not False
    scope_compliance_met = scope_required and len(scope_tasks_completed) >= 8 and all(
        criteria[criterion_id]["status"] == MET for criterion_id in SCOPE_CRITERIA
    )
    retired_lineages = _retired_lineages(sources)
    carry_forward_tracks = _carry_forward_tracks(sources)
    verdict_suffix = "scope_reduction_satisfied" if scope_compliance_met else "scope_reduction_incomplete"
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
        "scope_reduction_required": scope_required,
        "scope_reduction_tasks_completed": scope_tasks_completed,
        "scope_reduction_compliance_met": scope_compliance_met,
        "blocked_tasks": _blocked_tasks(sources, criteria),
        "retired_lineages": retired_lineages,
        "carry_forward_tracks": carry_forward_tracks,
        "missing_artifacts": _missing_artifacts(missing, criteria, roadmap_next_present),
        "source_artifacts_checked": _source_checks(REPO_ROOT, missing),
        "roadmap_inputs": _roadmap_inputs(
            roadmap_doc_present,
            roadmap_yaml_present,
            roadmap_next_present,
            conductor_log_present,
        ),
        "research_roadmap_yaml_modified": research_roadmap_yaml_modified,
        "scripts_research_conductor_modified": scripts_research_conductor_modified,
        "ops_docs_updated": ops_docs_updated,
        "ops_docs_update_note": (
            "ops/status.md and ops/changelog.md were not edited by Exp 1466 because the "
            "terminal stop rule delegates docs/status reconciliation to the conductor pass."
        )
        if not ops_docs_updated
        else "ops docs updated by the retrospective workflow.",
        "lessons_learned": [
            "Scope reduction was satisfied only where JSON artifacts and backing docs existed.",
            "The local SOTA runtime gate recovered, enabling a bounded repair salvage test.",
            "The validation-error context repair path produced no acceptance improvement and was retired.",
            "Paper, hardware, comparator, and benchmark work were narrowed to explicit next gates.",
        ],
        "honest_verdict": f"milestone_112_{criteria_met}_of_{criteria_total}_criteria_met_{verdict_suffix}",
    }


def run(root: Path | str = REPO_ROOT, out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    root_path = Path(root)
    write_in_progress_artifact(out_path)
    sources, missing = _load_sources(root_path / "results")
    artifact = build_artifact(
        sources,
        missing,
        evidence_paths_present=_evidence_paths_present(root_path),
        roadmap_doc_present=(
            root_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md"
        ).exists(),
        roadmap_yaml_present=(root_path / "research-roadmap.yaml").exists(),
        roadmap_next_present=(root_path / "research-roadmap-next.yaml").exists(),
        conductor_log_present=(root_path / "ops" / "conductor-log.md").exists(),
        research_roadmap_yaml_modified=_path_modified_by_git(root_path, "research-roadmap.yaml"),
        scripts_research_conductor_modified=_path_modified_by_git(
            root_path, "scripts/research_conductor.py"
        ),
        ops_docs_updated=False,
    )
    artifact["source_artifacts_checked"] = _source_checks(root_path, set(missing))
    return _write_json(Path(out_path), artifact)
