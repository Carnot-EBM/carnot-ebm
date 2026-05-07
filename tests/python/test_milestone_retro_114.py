"""Tests for the Exp 1491 milestone .114 retrospective.

Spec: REQ-REPORT-009, SCENARIO-REPORT-006.
"""

from __future__ import annotations

import json
from pathlib import Path

import carnot.reporting.milestone_retro_114 as retro114
from carnot.reporting.milestone_retro_114 import (
    GATE_SKIPPED,
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
        "exp1479": {
            "status": "complete",
            "milestone": "2026.04.114",
            "predecessor_criteria_met": 12,
            "predecessor_criteria_total": 12,
            "activation_manifest_complete": True,
            "telemetry_headline_block_preserved": True,
            "self_learning_followup_allowed": True,
            "research_roadmap_yaml_modified": False,
            "scripts_research_conductor_modified": False,
            "honest_verdict": "milestone_114_activation_complete_113_archived_guardrails_preserved",
        },
        "exp1480": {
            "status": "complete",
            "live_sota_model_inference_used": True,
            "logits_available": True,
            "superficial_baselines_recorded": True,
            "telemetry_cases_completed": 36,
            "balanced_label_counts": {"correct": 22, "incorrect": 14},
            "honest_verdict": "balanced_live_sota_telemetry_ready",
        },
        "exp1481": {
            "status": "complete",
            "semantic_energy_audit_complete": True,
            "signal_beats_superficial_baselines": False,
            "claim_allowed": False,
            "diagnostic_lineage_retired": True,
            "honest_verdict": "retired_semantic_energy_confounded_by_superficial_baseline",
        },
        "exp1482": {
            "status": "complete",
            "bound_is_sound": True,
            "bound_violations": [],
            "mock_or_live_logprobs": "live_exp1480_plus_exp1468",
            "constraints_evaluated": 18,
            "honest_verdict": "sound_bound_live_exp1480_plus_exp1468_calibrated",
        },
        "exp1483": {
            "status": "complete",
            "risk_decomposition_complete": True,
            "implemented_assumptions": ["BEAVER-lite sound prefix-bound checks"],
            "missing_assumptions": ["HalluGuard NTK feature construction"],
            "claim_allowed": False,
            "honest_verdict": "halluguard_style_fit_audit_only_no_full_reproduction",
        },
        "exp1484": {
            "status": "complete",
            "policy_integration_ready": True,
            "soundness_mistakes": 0,
            "task_success_delta": 0.5,
            "promotion_allowed": True,
            "honest_verdict": (
                "query_time_memory_policy_improves_bounded_replay_without_false_accepts"
            ),
        },
        "exp1485": {
            "status": "complete",
            "completeness_reduction_audit_complete": True,
            "baseline_soundness_mistakes": 0,
            "candidate_soundness_mistakes": 0,
            "completeness_mistake_delta": -12,
            "policy_change_allowed": True,
            "honest_verdict": "completeness_reduction_candidate_allowed_zero_soundness",
        },
        "exp1486": {
            "status": "complete",
            "live_sota_model_inference_used": True,
            "executable_constraint_benchmark_ready": True,
            "benchmark_cases": 20,
            "verifier_false_accept_rate": 0.0,
            "honest_verdict": (
                "complete: executable CCTU micro-benchmark ready with live local SOTA GGUF inference"
            ),
        },
        "exp1487": {
            "status": "complete",
            "pairwise_verification_complete": True,
            "pairwise_accuracy": 0.05,
            "random_baseline_accuracy": 0.5,
            "superficial_baseline_accuracy": 1.0,
            "energy_ranking_accuracy": 1.0,
            "improvement_allowed": False,
            "honest_verdict": "complete: no V_1 pairwise improvement over executable Carnot energy",
        },
        "exp1488": {
            "status": "complete",
            "thrml_preflight_complete": True,
            "thrml_import_ready": False,
            "hardware_claim_allowed": False,
            "simulator_lane_allowed": True,
            "honest_verdict": "thrml_not_importable_bounded_install_probe_blocked_simulator_only",
        },
        "exp1490": {
            "status": "complete",
            "localization_audit_complete": True,
            "decoded_quality_claim_allowed": False,
            "kona_dependency_used": False,
            "localization_top1_rate": 1.0,
            "random_baseline_rate": 0.16,
            "honest_verdict": (
                "bounded_injected_failure_localization_beats_random_no_decoded_quality_claim"
            ),
        },
    }


def _conductor_log_text() -> str:
    return "\n".join(
        [
            "| 2026-05-07 12:17 UTC | .113 Completion Archive + .114 Activation Manifest | OK | 81 passed |",
            "| 2026-05-07 15:46 UTC | THRML/Carnot Simulator Parity v2 - Gated on Exp148 | GATE_BLOCK | Pre-emptive skip: upstream retired |",
            "| 2026-05-07 16:17 UTC | THRML/Carnot Simulator Parity v2 - Gated on Exp148 | GATE_BLOCK | Pre-emptive skip: upstream retired |",
        ]
    )


def test_req_report_009_scores_114_criteria_and_counts_gate_skip_separately() -> None:
    """REQ-REPORT-009: .114 criteria are scored from source artifact fields."""

    artifact = build_artifact(
        sources=_scenario_sources(),
        missing_source_ids=["exp1489"],
        conductor_log_text=_conductor_log_text(),
        research_complete_text="- id: 2026.04.114\n",
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["criteria_total"] == 13
    assert artifact["criteria_met"] == 12
    assert artifact["success_threshold_met"] is True
    assert artifact["criteria_results"]["thrml_parity"]["status"] == GATE_SKIPPED
    assert artifact["honest_structured_gate_skip_count"] == 1
    assert artifact["completed_task_ids"] == [
        "exp1479",
        "exp1480",
        "exp1481",
        "exp1482",
        "exp1483",
        "exp1484",
        "exp1485",
        "exp1486",
        "exp1487",
        "exp1488",
        "exp1490",
        "exp1491",
    ]
    assert artifact["blocked_task_ids"] == ["exp1489"]
    assert artifact["ops_docs_updated"] is False
    assert "delegated" in artifact["ops_docs_update_note"]
    assert {item["lineage"] for item in artifact["retired_lineages"]} >= {
        "semantic_energy_headline_telemetry",
        "thrml_carnot_simulator_parity_until_import_ready",
    }
    assert len(artifact["carry_forward_recommendations"]) >= 5
    assert artifact["research_complete_has_114_entry"] is True
    assert artifact["honest_verdict"].startswith("complete: milestone_114_12_of_13")


def test_scenario_report_006_missing_open_gate_artifact_is_unmet_not_skipped() -> None:
    """SCENARIO-REPORT-006: missing gated artifacts only skip when the gate is closed."""

    sources = _scenario_sources()
    sources["exp1488"]["thrml_import_ready"] = True
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=["exp1489"],
        conductor_log_text="",
        research_complete_text="",
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    assert artifact["criteria_results"]["thrml_parity"]["status"] == UNMET
    assert artifact["honest_structured_gate_skip_count"] == 0
    assert artifact["criteria_met"] == 12
    assert {"path": "results/experiment_1489_thrml_carnot_simulator_parity_v2.json"} in [
        {"path": item["path"]} for item in artifact["missing_artifacts"]
    ]


def test_req_report_009_scores_present_thrml_parity_when_gate_opens() -> None:
    """REQ-REPORT-009: present gated artifacts are scored from their own fields."""

    sources = _scenario_sources()
    sources["exp1488"]["thrml_import_ready"] = True
    sources["exp1489"] = {
        "status": "complete",
        "simulator_parity_complete": True,
        "energy_agreement_reported": True,
        "honest_verdict": "complete: tiny THRML/Carnot parity matched",
    }
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=[],
        conductor_log_text="",
        research_complete_text="",
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    assert artifact["criteria_results"]["thrml_parity"]["status"] == MET
    assert artifact["criteria_met"] == 13
    assert "exp1489" in artifact["completed_task_ids"]


def test_req_report_009_missing_non_gated_artifact_is_unmet() -> None:
    """REQ-REPORT-009: missing non-gated source artifacts are not inferred."""

    sources = _scenario_sources()
    del sources["exp1486"]
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=["exp1486", "exp1489"],
        conductor_log_text=_conductor_log_text(),
        research_complete_text="",
        research_roadmap_yaml_modified=False,
        scripts_research_conductor_modified=False,
        ops_docs_updated=False,
    )

    assert artifact["criteria_results"]["executable_tool_use_benchmark"]["status"] == UNMET
    assert artifact["criteria_results"]["executable_tool_use_benchmark"]["source_values"] == {
        "status": "missing",
        "honest_verdict": "missing_artifact",
    }


def test_req_report_009_source_field_failures_are_unmet() -> None:
    """REQ-REPORT-009: false source fields are not promoted to success."""

    sources = _scenario_sources()
    sources["exp1480"]["superficial_baselines_recorded"] = False
    sources["exp1481"]["claim_allowed"] = True
    artifact = build_artifact(
        sources=sources,
        missing_source_ids=["exp1489"],
        conductor_log_text="",
        research_complete_text="",
        research_roadmap_yaml_modified=True,
        scripts_research_conductor_modified=True,
        ops_docs_updated=False,
    )

    assert artifact["criteria_results"]["balanced_telemetry"]["status"] == UNMET
    assert artifact["criteria_results"]["semantic_energy_audit"]["status"] == UNMET
    assert artifact["criteria_results"]["retro"]["status"] == UNMET
    assert artifact["criteria_met"] == 9
    assert artifact["success_threshold_met"] is False
    assert artifact["research_roadmap_yaml_modified"] is True
    assert artifact["scripts_research_conductor_modified"] is True


def test_req_report_009_step0_skeleton_has_required_fields(tmp_path: Path) -> None:
    """REQ-REPORT-009: STEP 0 writes a non-terminal in-progress skeleton."""

    out_path = tmp_path / "results" / "experiment_1491_milestone_114_retro.json"

    artifact = write_in_progress_artifact(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert written["status"] == "in_progress"
    assert set(written) == set(REQUIRED_ARTIFACT_FIELDS)


def test_req_report_009_run_writes_step0_before_loading_sources(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-009: run writes status=in_progress before source reads."""

    out_path = tmp_path / "results" / "experiment_1491_milestone_114_retro.json"
    observations: dict[str, object] = {}

    def fake_load_sources(_results_dir: Path):
        observations["step0"] = json.loads(out_path.read_text(encoding="utf-8"))
        return _scenario_sources(), ["exp1489"]

    monkeypatch.setattr(retro114, "_load_sources", fake_load_sources)
    monkeypatch.setattr(retro114, "_read_text", lambda _path: _conductor_log_text())
    monkeypatch.setattr(retro114, "_path_modified_by_git", lambda _root, _relative: False)

    artifact = run(root=tmp_path, out_path=out_path)

    step0 = observations["step0"]
    assert isinstance(step0, dict)
    assert step0["status"] == "in_progress"
    assert set(step0) == set(REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"


def test_scenario_report_006_run_reads_real_source_files(tmp_path: Path) -> None:
    """SCENARIO-REPORT-006: the runner loads .114 sources and writes JSON."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1491_milestone_114_retro.json"
    for exp_id, payload in _scenario_sources().items():
        _write_json(results_dir / SOURCE_FILES[exp_id], payload)
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "conductor-log.md").write_text(_conductor_log_text(), encoding="utf-8")
    (tmp_path / "research-complete.yaml").write_text(
        "- id: 2026.04.113\n",
        encoding="utf-8",
    )

    artifact = run(root=tmp_path, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["criteria_met"] == 12
    assert written["source_artifacts_checked"][-1] == {
        "experiment_id": "exp1490",
        "path": "results/experiment_1490_kona_ebt_partial_trace_localization_audit.json",
        "exists": True,
    }
    assert written["research_complete_has_114_entry"] is False
    assert written["research_complete_archive_update_needed"] is True
