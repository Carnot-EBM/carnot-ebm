"""Tests for the Exp 1466 milestone .112 retrospective.

Spec: REQ-REPORT-047, SCENARIO-REPORT-047.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_112 as retro112
from carnot.reporting.milestone_retro_112 import (
    GATE_BLOCKED_WITH_EVIDENCE,
    MET,
    REQUIRED_ARTIFACT_FIELDS,
    SOURCE_FILES,
    UNMET,
    build_artifact,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scenario_sources() -> dict[str, dict[str, object]]:
    return {
        "exp1453": {
            "status": "complete",
            "scope_reduction_required": True,
            "scope_reduction_manifest_complete": True,
            "scope_reduction_manifest_path": "ops/milestone_112_scope_reduction_manifest.md",
            "planned_scope_reduction_task_count": 10,
            "planned_scope_task_ids": [
                "exp1453",
                "exp1454",
                "exp1455",
                "exp1456",
                "exp1457",
                "exp1458",
                "exp1459",
                "exp1460",
                "exp1461",
                "exp1462",
            ],
            "honest_verdict": "scope_activation_complete",
        },
        "exp1454": {
            "status": "complete",
            "classification_table_written": True,
            "classification_table_path": "ops/experiment_signal_noise_classification.csv",
            "summary_written": True,
            "summary_path": "ops/experiment_signal_noise_summary.md",
            "top_50_noise_candidates": [{"experiment_id": "1087"}],
            "honest_verdict": "signal_noise_ledger_written",
        },
        "exp1455": {
            "status": "complete",
            "active_priority_count": 7,
            "trim_fraction": 0.7083,
            "priority_audit_path": "ops/mandatory_priority_audit.md",
            "known_issues_updated": True,
            "active_priorities_index_path": "ops/active-priorities.md",
            "honest_verdict": "priorities_trimmed",
        },
        "exp1456": {
            "status": "complete",
            "grpo_lineage_retired": True,
            "exclusion_manifest_updated": True,
            "consolidation_note_path": "ops/lineage-retirements/grpo_vprm_lineage_retired.md",
            "lessons_retained": ["early process reward signal retained"],
            "honest_verdict": "grpo_vprm_lineage_retired_no_v15_without_operator_reopen",
        },
        "exp1457": {
            "status": "complete",
            "wopr_puzzle_lineage_retired": True,
            "exclusion_manifest_updated": True,
            "retirement_note_path": "ops/lineage-retirements/wopr_puzzle_cartridges_retired.md",
            "preserved_assets": ["python/carnot/games/hex.py"],
            "honest_verdict": "wopr_puzzle_lineage_retired_demo_assets_preserved",
        },
        "exp1458": {
            "status": "complete",
            "hardnet_dsp_lineage_retired": True,
            "exclusion_manifest_updated": True,
            "consolidation_note_path": "ops/lineage-retirements/hardnet_dsp_repair_stack_retired.md",
            "lessons_retained": ["hard projection remains useful"],
            "honest_verdict": "hardnet_dsp_lineage_retired",
        },
        "exp1459": {
            "status": "complete",
            "self_learning_headline_pivot_selected": True,
            "self_learning_lineage_retired": False,
            "exp1447_delta_overall": 156,
            "decision_note_path": "docs/research-notes/self_learning_lineage_decision.md",
            "source_artifact_summaries": {"exp1447": {"honest_verdict": "positive_growth"}},
            "next_allowed_experiment_shape": {"allowed_count": 1},
            "honest_verdict": "self_learning_headline_pivot_selected_exp1447_verified_growth_only",
        },
        "exp1460": {
            "status": "complete",
            "active_hardware_track_count": 3,
            "architecture_updated": True,
            "hardware_wishlist_updated": True,
            "decision_note_path": "docs/research-notes/hardware_portfolio_narrowing.md",
            "active_hardware_tracks": [{"track_id": "dual_rtx3090_live_sota_runtime"}],
            "honest_verdict": "active_tracks_narrowed_to_3",
        },
        "exp1461": {
            "status": "complete",
            "comparator_decision_count": 10,
            "decision_table_path": "docs/research-notes/comparator_cite_retire_audit.md",
            "decisions": [
                {"comparator": "Abstract-CoT", "decision": "cite"},
                {"comparator": "Meta-Harness", "decision": "cite"},
                {"comparator": "Autodata", "decision": "cite"},
                {"comparator": "LARQL", "decision": "future_watchlist"},
                {"comparator": "Skillify", "decision": "future_watchlist"},
                {"comparator": "GStack", "decision": "future_watchlist"},
            ],
            "references_updated": True,
            "honest_verdict": "comparator_scope_narrowed",
        },
        "exp1462": {
            "status": "complete",
            "anchored_claim_count": 4,
            "anchored_claims": [
                {"claim_id": "CLAIM-1", "empirical_artifact_paths": ["results/a.json"]},
                {"claim_id": "CLAIM-2", "empirical_artifact_paths": ["results/b.json"]},
                {"claim_id": "CLAIM-3", "empirical_artifact_paths": ["results/c.json"]},
                {"claim_id": "CLAIM-4", "empirical_artifact_paths": ["results/d.json"]},
            ],
            "claim_matrix_path": "docs/research-notes/paper_v6_anchored_claim_matrix.md",
            "paper_updated": True,
            "paper_source_path": "docs/arxiv-paper/main.tex",
            "honest_verdict": "paper_v6_narrowed_to_4_anchored_claims",
        },
        "exp1463": {
            "status": "complete",
            "local_sota_runtime_ready": True,
            "live_sota_model_inference_used": True,
            "persistent_blockers": [],
            "models_missing_from_cache": [],
            "honest_verdict": "local_sota_runtime_ready",
        },
        "exp1464": {
            "status": "complete",
            "acceptance_delta_pp": 0.0,
            "repair_executor_lineage_retired": True,
            "repair_executor_lineage_preserved": False,
            "live_sota_model_inference_used": True,
            "cases_evaluated": 1,
            "honest_verdict": "complete_no_retry_context_improvement_repair_executor_retired",
        },
        "exp1465": {
            "status": "complete",
            "benchmark_adoption_decision": "adopt_beaver_style_deterministic_bounds_smoke",
            "benchmark_decision_table_path": "docs/research-notes/external_verifier_benchmark_fit.md",
            "adopted_benchmark": "BEAVER-style deterministic bounds",
            "next_minimal_benchmark_task": {"benchmark_family": "BEAVER-style deterministic bounds"},
            "honest_verdict": "adopt_one_minimal_beaver_bounds_smoke",
        },
    }


def _evidence_paths() -> set[str]:
    return {
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


def test_req_report_047_scores_all_112_criteria_with_backing_docs() -> None:
    """REQ-REPORT-047: all 14 roadmap criteria are scored from exact source fields."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=[],
        evidence_paths_present=_evidence_paths(),
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=False,
        conductor_log_present=True,
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    assert artifact["criteria_total"] == 14
    assert artifact["criteria_met"] == 14
    assert artifact["scope_reduction_required"] is True
    assert artifact["scope_reduction_compliance_met"] is True
    assert artifact["scope_reduction_tasks_completed"] == [
        "exp1453",
        "exp1454",
        "exp1455",
        "exp1456",
        "exp1457",
        "exp1458",
        "exp1459",
        "exp1460",
        "exp1461",
        "exp1462",
    ]
    assert artifact["missing_artifacts"] == [
        {"path": "research-roadmap-next.yaml", "reason": "requested_input_missing"}
    ]
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["scripts_research_conductor_modified"] is False
    assert artifact["ops_docs_updated"] is False
    assert artifact["success_criteria_results"]["repair_salvage"]["status"] == MET
    assert artifact["honest_verdict"] == "milestone_112_14_of_14_criteria_met_scope_reduction_satisfied"


def test_scenario_report_047_scope_reduction_requires_artifacts_and_docs() -> None:
    """SCENARIO-REPORT-047: scope-reduction criteria need both JSON and doc evidence."""

    evidence = _evidence_paths() - {"ops/lineage-retirements/grpo_vprm_lineage_retired.md"}

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=[],
        evidence_paths_present=evidence,
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=True,
        conductor_log_present=True,
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    criteria = artifact["success_criteria_results"]
    assert criteria["grpo_retirement"]["status"] == UNMET
    assert "exp1456" not in artifact["scope_reduction_tasks_completed"]
    assert artifact["scope_reduction_compliance_met"] is False
    assert artifact["criteria_met"] == 13
    assert artifact["missing_artifacts"] == [
        {
            "path": "ops/lineage-retirements/grpo_vprm_lineage_retired.md",
            "reason": "required_evidence_missing",
        }
    ]


def test_req_report_047_runtime_and_repair_exact_gates_do_not_get_inferred() -> None:
    """REQ-REPORT-047: runtime and repair gate failures are evidence, not wins."""

    sources = _scenario_sources()
    sources["exp1463"] = {
        "status": "complete",
        "local_sota_runtime_ready": False,
        "live_sota_model_inference_used": False,
        "persistent_blockers": ["libcudart_missing"],
        "same_verdict_retirement_recorded": False,
        "honest_verdict": "blocked_no_live_sota_runtime",
    }
    sources["exp1464"] = {
        "status": "blocked",
        "acceptance_delta_pp": None,
        "repair_executor_lineage_retired": False,
        "live_sota_model_inference_used": False,
        "honest_verdict": "blocked_runtime_gate_failed",
    }

    artifact = build_artifact(
        sources,
        missing_source_ids=[],
        evidence_paths_present=_evidence_paths(),
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=True,
        conductor_log_present=True,
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    criteria = artifact["success_criteria_results"]
    assert criteria["live_sota_runtime"]["status"] == GATE_BLOCKED_WITH_EVIDENCE
    assert criteria["repair_salvage"]["status"] == GATE_BLOCKED_WITH_EVIDENCE
    assert artifact["criteria_met"] == 12
    assert artifact["blocked_tasks"] == [
        {
            "criterion": "live_sota_runtime",
            "experiment_id": "exp1463",
            "honest_verdict": "blocked_no_live_sota_runtime",
            "blocker": "Live SOTA runtime did not reach readiness and no same-verdict retirement field passed.",
        },
        {
            "criterion": "repair_salvage",
            "experiment_id": "exp1464",
            "honest_verdict": "blocked_runtime_gate_failed",
            "blocker": "Repair salvage did not run behind a ready live-SOTA runtime gate.",
        },
    ]


def test_req_report_047_retired_lineages_and_carry_forward_rules_are_recorded() -> None:
    """REQ-REPORT-047: the retro preserves retired lineages and carry-forward rules."""

    artifact = build_artifact(
        _scenario_sources(),
        missing_source_ids=[],
        evidence_paths_present=_evidence_paths(),
        roadmap_doc_present=True,
        roadmap_yaml_present=True,
        roadmap_next_present=False,
        conductor_log_present=True,
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    retired = {item["lineage"]: item for item in artifact["retired_lineages"]}
    assert set(retired) == {
        "GRPO/VPRM",
        "WOPR puzzle cartridges",
        "HardNet++/DSP repair stack",
        "repair executor validation-error context",
    }
    assert retired["repair executor validation-error context"]["source_experiment"] == "exp1464"
    carry_forward = {item["track"]: item for item in artifact["carry_forward_tracks"]}
    assert set(carry_forward) == {
        "runtime",
        "repair",
        "self_learning",
        "paper_claims",
        "benchmark_adoption",
    }
    assert carry_forward["repair"]["rule"].startswith("Do not preserve the repair-executor")
    assert carry_forward["benchmark_adoption"]["source_experiment"] == "exp1465"


def test_req_report_047_step0_skeleton_has_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-047: STEP 0 writes an auditable in-progress skeleton first."""

    out_path = tmp_path / "results" / "experiment_1466_milestone_112_retro.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["status"] == "in_progress"
    assert set(written) == set(REQUIRED_ARTIFACT_FIELDS)


def test_req_report_047_run_writes_step0_before_loading_sources(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-047: run writes status=in_progress before reading source artifacts."""

    observations: dict[str, object] = {}
    out_path = tmp_path / "results" / "experiment_1466_milestone_112_retro.json"

    def fake_load_sources(_results_dir: Path):
        observations["step0"] = json.loads(out_path.read_text(encoding="utf-8"))
        return _scenario_sources(), []

    monkeypatch.setattr(retro112, "_load_sources", fake_load_sources)
    monkeypatch.setattr(retro112, "_evidence_paths_present", lambda _root: _evidence_paths())

    artifact = run(root=tmp_path, out_path=out_path)

    step0 = observations["step0"]
    assert isinstance(step0, dict)
    assert step0["status"] == "in_progress"
    assert set(step0) == set(REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"


def test_req_report_047_run_reads_realistic_source_files(tmp_path: Path) -> None:
    """SCENARIO-REPORT-047: the runner loads .112 source files and writes the final JSON."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1466_milestone_112_retro.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    for evidence_path in _evidence_paths():
        (tmp_path / evidence_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / evidence_path).write_text("evidence\n", encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# roadmap\n",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.04.112\n", encoding="utf-8")
    (tmp_path / "ops" / "conductor-log.md").write_text("exp1465 OK\n", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["criteria_met"] == 14
    assert written["source_artifacts_checked"][0] == {
        "experiment_id": "exp1453",
        "path": "results/experiment_1453_112_scope_reduction_activation_manifest.json",
        "exists": True,
    }
    assert written["roadmap_inputs"]["requested_research_roadmap_next_present"] is False


def test_req_report_047_run_reports_missing_source_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-047: missing source artifacts are explicit unmet criteria."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1466_milestone_112_retro.json"
    for exp_id, payload in _scenario_sources().items():
        if exp_id != "exp1465":
            _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    for evidence_path in _evidence_paths():
        (tmp_path / evidence_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / evidence_path).write_text("evidence\n", encoding="utf-8")
    (tmp_path / "openspec" / "change-proposals").mkdir(parents=True)
    (tmp_path / "openspec" / "change-proposals" / "research-roadmap-vNEXT.md").write_text(
        "# roadmap\n",
        encoding="utf-8",
    )
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.04.112\n", encoding="utf-8")
    (tmp_path / "ops" / "conductor-log.md").write_text("exp1464 OK\n", encoding="utf-8")

    artifact = run(root=tmp_path, out_path=out_path)

    assert artifact["success_criteria_results"]["verifier_benchmark_fit"]["status"] == UNMET
    assert artifact["criteria_met"] == 13
    assert {
        "path": "results/experiment_1465_external_verifier_benchmark_fit_audit.json",
        "reason": "source_artifact_missing",
    } in artifact["missing_artifacts"]
